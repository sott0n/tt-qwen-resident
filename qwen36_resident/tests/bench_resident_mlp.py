# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident decode design: the MLP half of a Qwen3.6-27B decoder layer, resident on QB2.

One program per chip runs `layers` residual MLP blocks back to back, x <- x + mlp(rmsnorm(x) * gamma),
with gate|up (bfp4) and down (bfp8) TP-sharded over the chips of a 1 x N mesh. Per chip:
  - 16 streamer cores (2 per DRAM bank) hold a replica of x, compute rmsnorm themselves, stream their
    column slices of gate|up and down through the weight ring, and exchange the intermediate activation
    slices directly with each other;
  - one hub core gathers the down-projection partials, all-reduces them over fabric and multicasts the
    per-chip partial slots back to the streamers (chip 0 folds x into its partial).
There is no host involvement and no op boundary between layers. The run is checked against a torch
reference built from the same (dequantized) weights, and timed per layer (slope over the layer count).
"""
import json
import math
import os
import struct
import time

import pytest
import torch
from loguru import logger

import ttnn

OUT = os.environ.get("BENCH_OUT", "/tmp/resident_mlp.jsonl")
KDIR = "models/experimental/qwen36_resident/kernels/"
TILE = 32
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
HIDDEN = 5120
INTER = 17408  # full MLP width; each chip holds INTER / num_chips
EPS = 1e-6
PER_BANK = int(os.environ.get("RESIDENT_PER_BANK", "4"))  # streamer cores per DRAM bank
L1_WEIGHT_BUDGET = 1_000_000
SEM_SLOTS, SEM_ACT, SEM_GATHER, SEM_FLAG = 0, 1, 2, 3
GU_DT, DOWN_DT = ttnn.bfloat4_b, ttnn.bfloat8_b


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


MAX_PAGE = int(os.environ.get("RESIDENT_MAX_PAGE", "8192"))  # DRAM read packet size cap


def block_geometry(Kt, dtype, max_block=20_000, max_page=None):
    max_page = max_page or MAX_PAGE
    tb = TILE_BYTES[dtype]
    sb = max(d for d in range(2, Kt + 1, 2) if Kt % d == 0 and d * tb <= max_block)
    block = sb * tb
    page = (max_page // tb) * tb
    while block % page:
        page -= tb
    return sb, block // page, page, block


def interleaved_bank_layout(w, core_cols, sb):
    """DRAM layout of a weight [K, N] for WIDTH_SHARDED banks read by PER_BANK cores each.

    core_cols[bank][j] lists the tile columns of w that core j of the bank streams, in order. Each core
    reads its columns as K blocks of sb tiles; the bank's cores' blocks are interleaved, slot
    r * PER_BANK + j holding block r of core j, so the cores of a bank read neighbouring addresses at
    the same time (disjoint per-core regions cost ~30% of the bank's bandwidth). Slots of cores with
    fewer columns stay zero. Returns [K, banks * width] whose row-major tilized shards hold the slots
    in order, width = PER_BANK x the largest per-core column count.
    """
    K, _ = w.shape
    Kt = K // TILE
    nkb = Kt // sb
    tiles = w.reshape(Kt, TILE, -1, TILE).permute(0, 2, 1, 3)  # [Kt, Nt, 32, 32]
    width = PER_BANK * max(len(c) for cols in core_cols for c in cols)
    zero = torch.zeros(TILE, TILE)
    shards = []
    for cols in core_cols:
        lin = [zero] * (Kt * width)
        for j, cj in enumerate(cols):
            for t, col in enumerate(cj):
                for kb in range(nkb):
                    slot = (t * nkb + kb) * PER_BANK + j
                    for i in range(sb):
                        lin[slot * sb + i] = tiles[kb * sb + i, col]
        grid = torch.stack(lin).reshape(Kt, width, TILE, TILE).permute(0, 2, 1, 3)
        shards.append(grid.reshape(K, width * TILE))
    return torch.cat(shards, dim=1), width


def bank_core_cols(total_tiles, banks, offset=0, first_extra=0):
    """core_cols for column tiles [0, total_tiles) split bank-major, then contiguously over PER_BANK cores."""
    per_bank = total_tiles // banks
    return [
        [
            list(range(offset + b * per_bank + f, offset + b * per_bank + f + c))
            for c, f in split(per_bank, PER_BANK, first_extra)
        ]
        for b in range(banks)
    ]


def gate_up_core_cols(it_total, banks, first_extra=0):
    """core_cols of cat(G, U): each core streams its gate tiles, then its up tiles."""
    g = bank_core_cols(it_total, banks, first_extra=first_extra)
    u = bank_core_cols(it_total, banks, offset=it_total, first_extra=first_extra)
    return [[gj + uj for gj, uj in zip(gb, ub)] for gb, ub in zip(g, u)]


def split(n, parts, first_extra=0):
    """(count, first) of `parts` contiguous pieces of n; the n % parts larger pieces start at part first_extra
    (cyclically), so entries with a remainder can put it on different cores."""
    q, r = divmod(n, parts)
    out, first = [], 0
    for j in range(parts):
        c = q + ((j - first_extra) % parts < r)
        out.append((c, first))
        first += c
    return out


def streamer_cores(device):
    """PER_BANK cores per DRAM bank: the bank-adjacent core plus its nearest free cores (column first)."""
    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    grid = device.compute_with_storage_grid_size()
    used = {(c.x, c.y) for c in primary}
    groups = []
    for c in primary:
        g = [ttnn.CoreCoord(c.x, c.y)]
        near = sorted(
            ((x, y) for x in range(grid.x) for y in range(grid.y) if (x, y) not in used),
            key=lambda p: (abs(p[0] - c.x) * 2 + abs(p[1] - c.y), p),
        )
        for x, y in near[: PER_BANK - 1]:
            used.add((x, y))
            g.append(ttnn.CoreCoord(x, y))
        assert len(g) == PER_BANK
        groups.append(g)
    return groups, used


def quantize(w, dtype):
    return ttnn.to_torch(ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def make_weights(num_chips, sets, banks, seed=0):
    """Per set and chip: dequantized G, U [H, I_c], D [I_c, H] and gamma [H]."""
    g = torch.Generator().manual_seed(seed)
    ic = INTER // num_chips
    out = []
    for _ in range(sets):
        chips = []
        for _ in range(num_chips):
            G = torch.randn(HIDDEN, ic, generator=g) / math.sqrt(HIDDEN)
            U = torch.randn(HIDDEN, ic, generator=g) / math.sqrt(HIDDEN)
            D = torch.randn(ic, HIDDEN, generator=g) * (0.5 / math.sqrt(ic))
            chips.append((quantize(G, GU_DT), quantize(U, GU_DT), quantize(D, DOWN_DT)))
        gamma = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g)).bfloat16().float()
        out.append((chips, gamma))
    return out


def torch_reference(x0, weights, layers):
    x = x0.clone()
    for l in range(layers):
        chips, gamma = weights[l % len(weights)]
        h = (x * torch.rsqrt(x.pow(2).mean() + EPS) * gamma).bfloat16().float()
        x = x + sum((torch.nn.functional.silu(h @ G) * (h @ U)) @ D for G, U, D in chips)
    return x


def build(mesh, weights, x0, layers, packet_bytes=4096, dbg=0):
    n = mesh.get_num_devices()
    assert tuple(mesh.shape)[0] == 1
    banks = mesh.dram_grid_size().x
    groups, used = streamer_cores(mesh)
    cores = [c for grp in groups for c in grp]
    S = len(cores)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    # hub: a free core inside the streamers' bounding box (the slot multicast loops back through it)
    xs, ys = [c.x for c in cores], [c.y for c in cores]
    x0r, x1r, y0r, y1r = min(xs), max(xs), min(ys), max(ys)
    hub = next(ttnn.CoreCoord(x, y) for x in range(x0r, x1r + 1) for y in range(y0r, y1r + 1) if (x, y) not in used)
    hub_grid = ttnn.CoreRangeSet([ttnn.CoreRange(hub, hub)])
    all_grid = grid.merge(hub_grid)
    mc_dests = (x1r - x0r + 1) * (y1r - y0r + 1)
    p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0r, y0r))
    p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x1r, y1r))
    hub_phys = mesh.worker_core_from_logical_core(hub)
    phys = [mesh.worker_core_from_logical_core(c) for c in cores]

    Ht, ic = HIDDEN // TILE, INTER // n
    It = ic // TILE
    it_bank, ht_bank = It // banks, Ht // banks
    tiny = ttnn.Tile([1, TILE])
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def l1_tiny(t, width, per_core_grid):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            tile=tiny,
            mesh_mapper=rep,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(per_core_grid, [1, width], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )

    sets = len(weights)
    slots0 = torch.zeros(S, n * HIDDEN)
    slots0[:, :HIDDEN] = x0
    slots_t = l1_tiny(slots0, n * HIDDEN, grid)
    slots_host_t = ttnn.from_torch(slots0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=tiny, mesh_mapper=rep)
    act_t = l1_tiny(torch.zeros(S, ic), ic, grid)
    gamma_t = l1_tiny(torch.cat([gm for _, gm in weights]).repeat(S, 1), sets * HIDDEN, grid)
    hub_slots = ttnn.from_torch(
        torch.zeros(2 * n, HIDDEN),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=rep,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(hub_grid, [2 * n, HIDDEN], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    ccl_sem = ttnn.create_global_semaphore(mesh, hub_grid, 0)

    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})

    geo = [block_geometry(Ht, GU_DT), block_geometry(It, DOWN_DT)]

    def dram_weight(per_chip, dtype, core_cols, sb):
        laid = [interleaved_bank_layout(w, core_cols, sb) for w in per_chip]
        k, width = per_chip[0].shape[0], laid[0][1]
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [k, width * TILE], ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.from_torch(
            torch.stack([t for t, _ in laid]).unsqueeze(1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )

    gu_cols, d_cols = gate_up_core_cols(It, banks), bank_core_cols(Ht, banks)
    w_gu = [
        dram_weight([torch.cat([G, U], dim=1) for G, U, _ in chips], GU_DT, gu_cols, geo[0][0]) for chips, _ in weights
    ]
    w_d = [dram_weight([D for _, _, D in chips], DOWN_DT, d_cols, geo[1][0]) for chips, _ in weights]

    lcm = math.lcm(*TILE_BYTES.values())
    ring_bytes = (L1_WEIGHT_BUDGET // lcm) * lcm
    ng_max = max(c for c, _ in split(it_bank, PER_BANK))
    nd_max = max(c for c, _ in split(ht_bank, PER_BANK))

    def cb(idx, pages, page=64, dtype=ttnn.bfloat16, tiled=True):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=idx,
            data_format=dtype,
            page_size=page,
            **({"tile": ttnn.TileDescriptor(1, TILE)} if tiled else {}),
        )
        return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid, format_descriptors=[fmt])

    full_page = TILE * TILE * 2

    def aliased(tiny_idx, full_idx, tiles):
        # one allocation seen as 1x32 tiles and as 32x32 tiles (same bytes)
        return ttnn.CBDescriptor(
            total_size=tiles * 64,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=tiny_idx, data_format=ttnn.bfloat16, page_size=64, tile=ttnn.TileDescriptor(1, TILE)
                ),
                ttnn.CBFormatDescriptor(buffer_index=full_idx, data_format=ttnn.bfloat16, page_size=full_page),
            ],
        )

    gamma_cb = ttnn.cb_descriptor_from_sharded_tensor(11, gamma_t)
    gamma_cb.format_descriptors = list(gamma_cb.format_descriptors) + [
        ttnn.CBFormatDescriptor(buffer_index=14, data_format=ttnn.bfloat16, page_size=full_page)
    ]
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, slots_t),
        ttnn.CBDescriptor(
            total_size=ring_bytes,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1, data_format=GU_DT, page_size=TILE_BYTES[GU_DT]),
                ttnn.CBFormatDescriptor(buffer_index=2, data_format=DOWN_DT, page_size=TILE_BYTES[DOWN_DT]),
            ],
        ),
        aliased(3, 12, Ht),
        aliased(4, 13, Ht),
        cb(5, 2 * ng_max),
        cb(6, ng_max),
        cb(7, 1, page=16, dtype=ttnn.uint32, tiled=False),
        ttnn.cb_descriptor_from_sharded_tensor(8, act_t),
        cb(9, nd_max),
        cb(10, nd_max),
        gamma_cb,
    ]
    sems = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=all_grid, initial_value=0)
        for i in (SEM_SLOTS, SEM_ACT, SEM_GATHER, SEM_FLAG)
    ]

    mesh_pd = ttnn.MeshProgramDescriptor()
    for chip in range(n):
        coord = ttnn.MeshCoordinate(0, chip)
        ct = [
            layers,
            sets,
            n,
            S,
            Ht,
            It,
            chip,
            ring_bytes,
            f32_bits(EPS),
            f32_bits(1 / math.sqrt(HIDDEN)),
            SEM_SLOTS,
            SEM_ACT,
            SEM_GATHER,
        ]
        for Kt, (sb, pages, page, block) in ((Ht, geo[0]), (It, geo[1])):
            ct += [Kt, sb, pages, page, block]
        ct += [dbg, ng_max, nd_max]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        peers = [v for p in phys for v in (p.x, p.y)]
        for i, c in enumerate(cores):
            bank, j = divmod(i, PER_BANK)
            ng, gfirst = split(it_bank, PER_BANK)[j]
            nd, dfirst = split(ht_bank, PER_BANK)[j]
            reader_rt[c.x][c.y] = [bank, i & 0x3, j, PER_BANK, 2 * ng, nd] + [
                t.buffer_address() for s in range(sets) for t in (w_gu[s], w_d[s])
            ]
            writer_rt[c.x][c.y] = [
                ng,
                bank * it_bank + gfirst,
                nd,
                bank * ht_bank + dfirst,
                hub_phys.x,
                hub_phys.y,
                hub_slots.buffer_address(),
                act_t.buffer_address(),
            ] + peers
            compute_rt[c.x][c.y] = [ng, nd, bank * ht_bank + dfirst]

        def dm(src, rt, proc, noc, core_ranges, ctargs):
            return ttnn.KernelDescriptor(
                kernel_source=KDIR + src,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=ctargs,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
            )

        hub_rt = ttnn.RuntimeArgs()
        hub_rt[hub.x][hub.y] = [
            hub_slots.buffer_address(),
            ttnn.get_global_semaphore_address(ccl_sem),
            slots_t.buffer_address(),
        ]
        hub_ct = [
            n,
            chip,
            HIDDEN * 2,
            packet_bytes,
            layers,
            S,
            SEM_GATHER,
            SEM_SLOTS,
            p0.x,
            p0.y,
            p1.x,
            p1.y,
            mc_dests,
            SEM_FLAG,
        ]
        compute = ttnn.KernelDescriptor(
            kernel_source=KDIR + "mlp_streamer_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=ct,
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(),
        )
        compute.config.math_fidelity = ttnn.MathFidelity.LoFi
        kernels = [
            dm(
                "mlp_streamer_reader.cpp",
                reader_rt,
                ttnn.DataMovementProcessor.RISCV_1,
                ttnn.NOC.RISCV_0_default,
                grid,
                ct,
            ),
            dm(
                "mlp_streamer_writer.cpp",
                writer_rt,
                ttnn.DataMovementProcessor.RISCV_0,
                ttnn.NOC.RISCV_1_default,
                grid,
                ct,
            ),
            compute,
            dm("mlp_hub.cpp", hub_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default, hub_grid, hub_ct),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        args = program.kernels[3].runtime_args[hub.x][hub.y]
        for nb in (chip + 1, chip - 1):
            if 0 <= nb < n:
                me = mesh.get_fabric_node_id(coord)
                args.append(1)
                args.extend(
                    ttnn.setup_fabric_connection(
                        me, mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)), 0, program, hub
                    )
                )
            else:
                args.append(0)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program

    io = [slots_t, act_t, gamma_t, hub_slots] + w_gu + w_d
    bytes_per_layer = HIDDEN * 2 * ic * 0.5625 + ic * HIDDEN * 1.0625

    # the hub multicasts overwrite the streamers' slots during a run: restore layer 0's input each launch

    def run(keep=(ccl_sem,)):
        ttnn.copy_host_to_device_tensor(slots_host_t, slots_t)
        ttnn.generic_op(io, mesh_pd)

    def result():
        got = ttnn.to_torch(hub_slots, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
        p = (layers - 1) & 1
        return got[: 2 * n][p * n : (p + 1) * n].sum(0)  # chip 0's copy of the last layer's slots

    info = dict(
        chips=n,
        streamers=S,
        hub=(hub.x, hub.y),
        mc_rect=(x0r, y0r, x1r, y1r),
        ring_bytes=ring_bytes,
        bytes_per_layer_per_chip=bytes_per_layer,
    )
    return run, result, info


def timed(mesh, fn, reps=3):
    fn()
    ttnn.synchronize_device(mesh)
    t = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        t += time.perf_counter() - t0
    return t / reps


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def run_case(mesh, layers_short, layers_long, sets=2):
    n = mesh.get_num_devices()
    banks = mesh.dram_grid_size().x
    weights = make_weights(n, sets, banks)
    x0 = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(1)).bfloat16().float()
    run_s, res_s, info = build(mesh, weights, x0, layers_short)
    run_l, res_l, _ = build(mesh, weights, x0, layers_long)
    logger.info(info)
    ts = timed(mesh, run_s)
    tl = timed(mesh, run_l)
    per_layer = (tl - ts) / (layers_long - layers_short)
    checks = {}
    for L, res in ((layers_short, res_s), (layers_long, res_l)):
        ref = torch_reference(x0, weights, L)
        got = res()
        checks[L] = dict(pcc=pcc(got, ref), rel_err=((got - ref).norm() / ref.norm()).item())
    rec = dict(
        info,
        us_per_layer=per_layer * 1e6,
        weight_us_at_365=info["bytes_per_layer_per_chip"] / 365e9 * 1e6,
        GBps_per_chip=info["bytes_per_layer_per_chip"] / per_layer / 1e9,
        checks=checks,
    )
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_resident_mlp_1chip(mesh_device):
    rec = run_case(mesh_device, 2, 10)
    assert all(c["pcc"] > 0.99 for c in rec["checks"].values())


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_mlp_4chip(mesh_device):
    rec = run_case(mesh_device, 2, 18)
    assert all(c["pcc"] > 0.99 for c in rec["checks"].values())
