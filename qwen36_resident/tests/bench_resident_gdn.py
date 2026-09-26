# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident decode design: a full Qwen3.6-27B linear-attention (Gated DeltaNet) decoder layer on QB2.

One program per chip runs `layers` decoder layers back to back, each an attention half and an MLP half
with a fabric all-reduce after each, TP-sharded over a 1 x N mesh. Per chip:
  - 16 streamer cores (2 per DRAM bank) hold a replica of x, run rmsnorm, stream their column slices
    of qkvzab / out / gate|up / down through the weight ring, apply conv1d + silu to their q|k|v
    columns (per-core conv state), and send the qkvzab row to the head cores;
  - one head core per local value head runs the delta-rule recurrence (fp32 state resident in its L1),
    the gated rmsnorm and silu(z) gate, and writes its output into every streamer;
  - one hub core all-reduces the partials over fabric and multicasts the slots back.
Checked against a torch reference built from the same (dequantized) weights and states.
"""
import gc
import json
import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.qwen36_resident.tests.bench_resident_mlp import (
    DOWN_DT,
    EPS,
    GU_DT,
    HIDDEN,
    INTER,
    L1_WEIGHT_BUDGET,
    PER_BANK,
    TILE,
    TILE_BYTES,
    bank_core_cols,
    block_geometry,
    f32_bits,
    gate_up_core_cols,
    interleaved_bank_layout,
    pcc,
    quantize,
    split,
    streamer_cores,
    timed,
)

OUT = os.environ.get("BENCH_OUT", "/tmp/resident_gdn.jsonl")
KDIR = "models/experimental/qwen36_resident/kernels/"
NK, NV, DK, DV, CONV_K = 16, 48, 128, 128, 4
QKVZ_DT, OUT_DT = ttnn.bfloat8_b, ttnn.bfloat8_b
SEM_SLOTS, SEM_ACT, SEM_GATHER, SEM_FLAG, SEM_HEADS, SEM_ROWS, SEM_STATE = range(7)
HEAD_FP32_CBS = (5, 6, 7, 15, 30, 31)
FP32_ACC = os.environ.get("RESIDENT_FP32_ACC", "1") == "1"
STREAMER_FIDELITY = getattr(ttnn.MathFidelity, os.environ.get("RESIDENT_FIDELITY", "HiFi4"))


class Dims:
    def __init__(self, n, banks):
        self.n, self.banks = n, banks
        self.nk, self.nv = NK // n, NV // n
        self.qd, self.vd = self.nk * DK, self.nv * DV
        self.conv_ch = 2 * self.qd + self.vd
        self.z0 = self.conv_ch
        self.a0 = self.z0 + self.vd
        self.b0 = self.a0 + self.nv
        width = self.b0 + self.nv
        step = TILE * banks
        self.row_cols = (width + step - 1) // step * step
        self.row_tiles = self.row_cols // TILE
        self.conv_tiles = self.conv_ch // TILE
        self.Ot = self.vd // TILE
        self.ic = INTER // n
        self.It = self.ic // TILE
        self.Ht = HIDDEN // TILE


def make_weights(d, sets, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    out = []
    for _ in range(sets):
        chips = []
        for _ in range(d.n):
            Wq = torch.zeros(HIDDEN, d.row_cols)
            Wq[:, : d.b0 + d.nv] = rn(HIDDEN, d.b0 + d.nv) / math.sqrt(HIDDEN)
            chips.append(
                dict(
                    Wq=quantize(Wq, QKVZ_DT),
                    taps=(0.5 * rn(d.conv_ch, CONV_K)).bfloat16().float(),
                    dt=0.5 * rn(d.nv),
                    neg_a=-torch.exp(torch.rand(d.nv, generator=g) * 2 - 1),
                    state=0.05 * rn(d.nv, DK, DV),
                    Wo=quantize(rn(d.vd, HIDDEN) * (0.5 / math.sqrt(d.vd)), OUT_DT),
                    G=quantize(rn(HIDDEN, d.ic) / math.sqrt(HIDDEN), GU_DT),
                    U=quantize(rn(HIDDEN, d.ic) / math.sqrt(HIDDEN), GU_DT),
                    D=quantize(rn(d.ic, HIDDEN) * (0.5 / math.sqrt(d.ic)), DOWN_DT),
                )
            )
        out.append(
            dict(
                chips=chips,
                gamma_in=(1 + 0.1 * rn(HIDDEN)).bfloat16().float(),
                gamma_post=(1 + 0.1 * rn(HIDDEN)).bfloat16().float(),
                norm_w=(1 + 0.1 * rn(DV)).bfloat16().float(),
            )
        )
    return out


def bf16(t):
    return t.bfloat16().float()


def rmsnorm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def torch_reference(x0, d, weights, layers, bf16_dataflow=False):
    """fp32 reference; bf16_dataflow rounds x and the chip partials to bf16 like the device buffers."""
    r = bf16 if bf16_dataflow else (lambda t: t)
    sets = len(weights)
    states = [[c["state"].clone() for c in w["chips"]] for w in weights]
    hist = [[torch.zeros(CONV_K - 1, d.conv_ch) for _ in range(d.n)] for _ in range(sets)]
    x = x0.clone()
    scale = DK**-0.5
    for l in range(layers):
        s = l % sets
        w = weights[s]
        h = bf16(rmsnorm(x, w["gamma_in"]))
        parts = []
        for c, cw in enumerate(w["chips"]):
            y = bf16(h @ cw["Wq"])
            qkv = y[: d.conv_ch]
            window = torch.cat([hist[s][c], qkv[None]], 0)  # [K, C], oldest first
            hist[s][c] = window[1:]
            conv = bf16(torch.nn.functional.silu((window * cw["taps"].T).sum(0)))
            q, k, v = conv[: d.qd], conv[d.qd : 2 * d.qd], conv[2 * d.qd : d.conv_ch]
            z, a, b = y[d.z0 : d.a0], y[d.a0 : d.b0], y[d.b0 : d.b0 + d.nv]
            outs = []
            for hh in range(d.nv):
                kh = hh // (d.nv // d.nk)
                qn = q[kh * DK : (kh + 1) * DK]
                qn = qn * torch.rsqrt(qn.pow(2).sum() + 1e-6) * scale
                kn = k[kh * DK : (kh + 1) * DK]
                kn = kn * torch.rsqrt(kn.pow(2).sum() + 1e-6)
                decay = torch.exp(cw["neg_a"][hh] * torch.nn.functional.softplus(a[hh] + cw["dt"][hh]))
                beta = torch.sigmoid(b[hh])
                S = states[s][c][hh]
                delta = beta * (v[hh * DV : (hh + 1) * DV] - (decay * kn) @ S)
                S = decay * S + torch.outer(kn, delta)
                states[s][c][hh] = S
                o = qn @ S
                on = rmsnorm(o, w["norm_w"])
                outs.append(bf16(on * torch.nn.functional.silu(z[hh * DV : (hh + 1) * DV])))
            parts.append(r(torch.cat(outs) @ cw["Wo"] + (x if c == 0 else 0)))
        x = r(sum(parts))
        h = bf16(rmsnorm(x, w["gamma_post"]))
        x = r(
            sum(
                r(bf16(torch.nn.functional.silu(h @ cw["G"]) * (h @ cw["U"])) @ cw["D"] + (x if c == 0 else 0))
                for c, cw in enumerate(w["chips"])
            )
        )
    return x


def build(mesh, d, weights, x0, layers, packet_bytes=4096, timeline=False):
    n, banks = d.n, d.banks
    groups, used = streamer_cores(mesh)
    cores = [c for grp in groups for c in grp]
    S = len(cores)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    xs, ys = [c.x for c in cores], [c.y for c in cores]
    x0r, x1r, y0r, y1r = min(xs), max(xs), min(ys), max(ys)
    hub = next(ttnn.CoreCoord(x, y) for x in range(x0r, x1r + 1) for y in range(y0r, y1r + 1) if (x, y) not in used)
    used.add((hub.x, hub.y))
    cgrid = mesh.compute_with_storage_grid_size()
    free = [ttnn.CoreCoord(x, y) for y in range(cgrid.y) for x in range(cgrid.x) if (x, y) not in used]
    heads = free[: d.nv]
    hub_grid = ttnn.CoreRangeSet([ttnn.CoreRange(hub, hub)])
    head_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in heads])
    all_grid = grid.merge(hub_grid).merge(head_grid)
    mc_dests = (x1r - x0r + 1) * (y1r - y0r + 1)
    p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0r, y0r))
    p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x1r, y1r))
    hub_phys = mesh.worker_core_from_logical_core(hub)
    phys = [mesh.worker_core_from_logical_core(c) for c in cores]
    head_phys = [mesh.worker_core_from_logical_core(c) for c in heads]
    sets = len(weights)
    tiny = ttnn.Tile([1, TILE])
    rep = ttnn.ReplicateTensorToMesh(mesh)
    per_chip = ttnn.ShardTensorToMesh(mesh, dim=0)

    def sharded(t, grid_, shard, dtype=ttnn.bfloat16, tile=tiny, layout=ttnn.TILE_LAYOUT, mapper=rep):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=mesh,
            tile=tile,
            mesh_mapper=mapper,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid_, shard, ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )

    # ---- per-core column assignment
    q_bank, ht_bank, it_bank = d.row_tiles // banks, d.Ht // banks, d.It // banks
    # the remainder columns of qkvzab and gate|up go to different cores of the bank: a bank's cores share
    # its bandwidth evenly, so the core with the most bytes per layer paces the whole layer
    g_extra = q_bank % PER_BANK
    nq_split, nd_split = split(q_bank, PER_BANK), split(ht_bank, PER_BANK)
    ng_split = split(it_bank, PER_BANK, g_extra)
    nq_max, nd_max, ng_max = (max(c for c, _ in sp) for sp in (nq_split, nd_split, ng_split))

    # ---- streamer tensors
    slots0 = torch.zeros(S, n * HIDDEN)
    slots0[:, :HIDDEN] = x0
    slots_t = sharded(slots0, grid, [1, n * HIDDEN])
    slots_host = ttnn.from_torch(slots0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=tiny, mesh_mapper=rep)
    act_t = sharded(torch.zeros(S, d.ic), grid, [1, d.ic])
    o_in_t = sharded(torch.zeros(S, d.vd), grid, [1, d.vd])
    gamma_t = sharded(
        torch.cat([torch.cat([w["gamma_in"], w["gamma_post"]]) for w in weights]).repeat(S, 1),
        grid,
        [1, sets * 2 * HIDDEN],
    )
    blk = TILE * TILE  # one 32x32 tile's worth of 1x32 tiles
    conv_w = torch.zeros(n, S, sets * CONV_K * blk)
    for c in range(n):
        for i in range(S):
            bank, j = divmod(i, PER_BANK)
            cnt, first = nq_split[j]
            for s, w in enumerate(weights):
                taps = w["chips"][c]["taps"]
                for t in range(cnt):
                    g = bank * q_bank + first + t
                    if g >= d.conv_tiles:
                        continue
                    for tap in range(CONV_K):
                        o = (s * CONV_K + tap) * blk + t * TILE
                        conv_w[c, i, o : o + TILE] = taps[g * TILE : (g + 1) * TILE, tap]
    conv_w_t = sharded(conv_w.reshape(n * S, -1), grid, [1, sets * CONV_K * blk], mapper=per_chip)
    hist_t = sharded(torch.zeros(S, sets * 3 * blk), grid, [1, sets * 3 * blk])

    # ---- head tensors
    rows_t = sharded(torch.zeros(d.nv, d.row_cols), head_grid, [1, d.row_cols])
    state = torch.zeros(n, d.nv, TILE, sets * 16 * TILE)
    norm_w = torch.zeros(n, d.nv, TILE, sets * 4 * TILE)
    for c in range(n):
        for hh in range(d.nv):
            for s, w in enumerate(weights):
                S_ = w["chips"][c]["state"][hh]
                for kt in range(4):
                    for vt in range(4):
                        o = (s * 16 + kt * 4 + vt) * TILE
                        state[c, hh, :, o : o + TILE] = S_[kt * TILE : (kt + 1) * TILE, vt * TILE : (vt + 1) * TILE]
                for vt in range(4):
                    o = (s * 4 + vt) * TILE
                    norm_w[c, hh, 0, o : o + TILE] = w["norm_w"][vt * TILE : (vt + 1) * TILE]
    full = ttnn.Tile([TILE, TILE])
    state_t = sharded(
        state.reshape(n * d.nv * TILE, -1),
        head_grid,
        [TILE, sets * 16 * TILE],
        dtype=ttnn.float32,
        tile=full,
        mapper=per_chip,
    )
    norm_w_t = sharded(
        norm_w.reshape(n * d.nv * TILE, -1), head_grid, [TILE, sets * 4 * TILE], tile=full, mapper=per_chip
    )

    # optional per-core timeline (16 uint32 per layer, see the streamer reader/writer)
    ts_t = None
    if timeline:
        ts_t = ttnn.from_torch(
            torch.zeros(S, layers * 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [1, layers * 16], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
    ts_addr = ts_t.buffer_address() if timeline else 0
    hts_t = None
    if timeline:
        hts_t = ttnn.from_torch(
            torch.zeros(d.nv, layers * 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(head_grid, [1, layers * 4], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
    hts_addr = hts_t.buffer_address() if timeline else 0

    # ---- hub
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

    # ---- DRAM weights (per chip), column slices bank-major
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})

    entries = [(d.Ht, QKVZ_DT), (d.Ot, OUT_DT), (d.Ht, GU_DT), (d.It, DOWN_DT)]
    geo = [block_geometry(Kt, dt) for Kt, dt in entries]

    def dram_weight(mats, dtype, core_cols, sb):
        laid = [interleaved_bank_layout(m, core_cols, sb) for m in mats]
        k, width = mats[0].shape[0], laid[0][1]
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
            mesh_mapper=per_chip,
        )

    cols = [
        bank_core_cols(d.row_tiles, banks),
        bank_core_cols(d.Ht, banks),
        gate_up_core_cols(d.It, banks, g_extra),
        bank_core_cols(d.Ht, banks),
    ]
    w_dram = []
    for w in weights:
        ch = w["chips"]
        mats = [
            [c["Wq"] for c in ch],
            [c["Wo"] for c in ch],
            [torch.cat([c["G"], c["U"]], dim=1) for c in ch],
            [c["D"] for c in ch],
        ]
        w_dram.append([dram_weight(m, dt, cc, g[0]) for m, (_, dt), cc, g in zip(mats, entries, cols, geo)])

    lcm = math.lcm(*TILE_BYTES.values())
    ring_bytes = (L1_WEIGHT_BUDGET // lcm) * lcm

    # ---- CBs
    def tiny_cb(idx, pages, grid_=grid):
        return ttnn.CBDescriptor(
            total_size=pages * 64,
            core_ranges=grid_,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=ttnn.bfloat16, page_size=64, tile=ttnn.TileDescriptor(1, TILE)
                )
            ],
        )

    full_page = TILE * TILE * 2

    def aliased(tiny_idx, full_idx, tiles):
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

    def with_full_view(desc, full_idx):
        desc.format_descriptors = list(desc.format_descriptors) + [
            ttnn.CBFormatDescriptor(buffer_index=full_idx, data_format=ttnn.bfloat16, page_size=full_page)
        ]
        return desc

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
                ttnn.CBFormatDescriptor(buffer_index=1 + e, data_format=dt, page_size=TILE_BYTES[dt])
                for e, (_, dt) in enumerate(entries)
            ],
        ),
        aliased(5, 12, d.Ht),
        aliased(6, 13, d.Ht),
        ttnn.CBDescriptor(
            total_size=16,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=7, data_format=ttnn.uint32, page_size=16)],
        ),
        ttnn.cb_descriptor_from_sharded_tensor(8, act_t),
        tiny_cb(9, nd_max),
        tiny_cb(10, nd_max),
        gamma_cb,
        aliased(15, 27, TILE),  # gate block | 32x32 view
        aliased(26, 28, TILE),  # up block
        aliased(16, 29, TILE),  # silu(g) * u block
        aliased(17, 22, TILE),  # qkvzab columns
        aliased(18, 23, TILE),  # after conv1d + silu
        with_full_view(ttnn.cb_descriptor_from_sharded_tensor(19, conv_w_t), 24),
        with_full_view(ttnn.cb_descriptor_from_sharded_tensor(20, hist_t), 25),
        ttnn.cb_descriptor_from_sharded_tensor(21, o_in_t),
    ]
    bf, f32 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
    head_cb_spec = {
        0: (bf, 4),
        1: (bf, 4),
        2: (bf, 4),
        3: (bf, 4),
        4: (bf, 4),
        5: (f32, 16),
        6: (f32, 1),
        7: (f32, 1),
        8: (f32, 1),
        9: (f32, 1),
        10: (f32, 4),
        11: (f32, 4),
        12: (f32, 1),
        13: (f32, 4),
        14: (f32, 4),
        15: (f32, 1),
        16: (f32, 1),
        17: (f32, 16),
        18: (f32, 4),
        19: (f32, 1),
        20: (f32, 4),
        21: (f32, 4),
        23: (f32, 16),
        24: (f32, 16),
        25: (f32, 4),
        26: (f32, 4),
        27: (f32, 4),
        28: (bf, 4),
        30: (f32, 1),
        31: (f32, 1),
    }
    for idx, ((dt, page), cnt) in head_cb_spec.items():
        cbs.append(
            ttnn.CBDescriptor(
                total_size=cnt * page,
                core_ranges=head_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
            )
        )
    cbs.append(
        ttnn.CBDescriptor(
            total_size=4 * 64,
            core_ranges=head_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=22, data_format=ttnn.bfloat16, page_size=4 * 64)],
        )
    )
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_grid, initial_value=0) for i in range(7)]

    head_unpack = type(ttnn.ComputeConfigDescriptor().unpack_to_dest_mode)([ttnn.UnpackToDestMode.Default] * 64)
    for i in HEAD_FP32_CBS:
        head_unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32

    def kernel(src, core_ranges, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    def dm(proc, noc):
        return ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)

    mesh_pd = ttnn.MeshProgramDescriptor()
    for chip in range(n):
        coord = ttnn.MeshCoordinate(0, chip)
        ct = [
            layers,
            sets,
            n,
            S,
            d.Ht,
            d.It,
            chip,
            ring_bytes,
            f32_bits(EPS),
            f32_bits(1 / math.sqrt(HIDDEN)),
            SEM_SLOTS,
            SEM_ACT,
            SEM_GATHER,
        ]
        for (Kt, _), (sb, pages, page, block) in zip(entries, geo):
            ct += [Kt, sb, pages, page, block]
        ct += [ng_max, nd_max, nq_max, d.conv_tiles, d.Ot, d.nv, SEM_HEADS, d.row_tiles, SEM_ROWS, d.nv]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        peers = [v for p in phys for v in (p.x, p.y)]
        head_xy = [v for p in head_phys for v in (p.x, p.y)]
        for i, c in enumerate(cores):
            bank, j = divmod(i, PER_BANK)
            nq, qf = nq_split[j]
            nd, dfirst = nd_split[j]
            ng, gfirst = ng_split[j]
            reader_rt[c.x][c.y] = (
                [bank, i & 0x3, j, PER_BANK, nq, nd, 2 * ng, nd]
                + [t.buffer_address() for s in range(sets) for t in w_dram[s]]
                + [ts_addr]
            )
            writer_rt[c.x][c.y] = (
                [
                    ng,
                    bank * it_bank + gfirst,
                    nd,
                    bank * ht_bank + dfirst,
                    hub_phys.x,
                    hub_phys.y,
                    hub_slots.buffer_address(),
                    act_t.buffer_address(),
                    nq,
                    bank * q_bank + qf,
                    rows_t.buffer_address(),
                ]
                + head_xy
                + peers
                + [ts_addr]
            )
            n_conv = max(0, min(nq, d.conv_tiles - (bank * q_bank + qf)))
            compute_rt[c.x][c.y] = [ng, nd, bank * ht_bank + dfirst, nq, n_conv]

        head_r, head_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        group = d.nv // d.nk
        for hh, c in enumerate(heads):
            kh = hh // group
            a_col, b_col = d.a0 + hh, d.b0 + hh
            gates = []
            for w in weights:
                cw = w["chips"][chip]
                gates += [f32_bits(float(cw["dt"][hh])), f32_bits(float(cw["neg_a"][hh]))]
            head_r[c.x][c.y] = (
                [
                    rows_t.buffer_address(),
                    state_t.buffer_address(),
                    norm_w_t.buffer_address(),
                    kh * 4,
                    d.qd // TILE + kh * 4,
                    2 * d.qd // TILE + hh * 4,
                    d.z0 // TILE + hh * 4,
                    a_col // TILE,
                    a_col % TILE,
                    b_col // TILE,
                    b_col % TILE,
                ]
                + gates
                + [hts_addr]
            )
            head_w[c.x][c.y] = [state_t.buffer_address(), o_in_t.buffer_address(), hh] + peers + [hts_addr]

        compute = ttnn.ComputeConfigDescriptor()
        # custom_mm is LoFi by construction (no fidelity phases); the fidelity only applies to the
        # rmsnorm multiplies
        compute.math_fidelity = STREAMER_FIDELITY
        compute.fp32_dest_acc_en = FP32_ACC
        compute.dst_full_sync_en = FP32_ACC  # keeps 8 DST tiles (rmsnorm holds 5 full tiles)
        head_compute = ttnn.ComputeConfigDescriptor()
        head_compute.math_fidelity = ttnn.MathFidelity.HiFi4
        head_compute.fp32_dest_acc_en = True
        head_compute.unpack_to_dest_mode = head_unpack
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
            2 * layers,
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
        head_ct = [4, 4, layers, sets, S]
        head_compute_ct = [
            d.nk,
            d.nv,
            4,
            4,
            group,
            f32_bits(DK**-0.5),
            f32_bits(1e-6),
            f32_bits(1e-6),
            f32_bits(1.0 / DV),
            layers,
        ]
        kernels = [
            kernel(
                "gdn_streamer_reader.cpp",
                grid,
                ct,
                reader_rt,
                dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_0_default),
            ),
            kernel(
                "gdn_streamer_writer.cpp",
                grid,
                ct,
                writer_rt,
                dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_1_default),
            ),
            kernel("gdn_streamer_compute.cpp", grid, ct, compute_rt, compute),
            kernel(
                "mlp_hub.cpp",
                hub_grid,
                hub_ct,
                hub_rt,
                dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
            ),
            kernel(
                "gdn_head_reader.cpp",
                head_grid,
                head_ct + [SEM_ROWS, SEM_STATE],
                head_r,
                dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
            ),
            kernel(
                "gdn_head_writer.cpp",
                head_grid,
                head_ct + [SEM_HEADS, SEM_STATE],
                head_w,
                dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
            ),
            kernel("gdn_head_compute.cpp", head_grid, head_compute_ct, ttnn.RuntimeArgs(), head_compute),
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

    io = (
        ([ts_t, hts_t] if timeline else [])
        + [slots_t, act_t, o_in_t, gamma_t, conv_w_t, hist_t, rows_t, state_t, norm_w_t, hub_slots]
        + [t for ws in w_dram for t in ws]
    )
    state_host = ttnn.from_torch(
        state.reshape(n * d.nv * TILE, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, tile=full, mesh_mapper=per_chip
    )
    hist_host = ttnn.from_torch(
        torch.zeros(S, sets * 3 * blk),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=tiny,
        mesh_mapper=rep,
    )

    def run(reset=True, keep=(ccl_sem,)):
        # a checked launch starts from the same x, recurrent states and conv history; timed launches
        # skip the (large, variable) host uploads
        if reset:
            ttnn.copy_host_to_device_tensor(slots_host, slots_t)
            ttnn.copy_host_to_device_tensor(state_host, state_t)
            ttnn.copy_host_to_device_tensor(hist_host, hist_t)
        ttnn.generic_op(io, mesh_pd)

    def result():
        got = ttnn.to_torch(hub_slots, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
        p = (2 * layers - 1) & 1
        return got[: 2 * n][p * n : (p + 1) * n].sum(0)

    weight_bytes = sum(
        k * cols * TILE_BYTES[dt] / (TILE * TILE)
        for (k, cols, dt) in (
            (HIDDEN, d.row_cols, QKVZ_DT),
            (d.vd, HIDDEN, OUT_DT),
            (HIDDEN, 2 * d.ic, GU_DT),
            (d.ic, HIDDEN, DOWN_DT),
        )
    )
    info = dict(chips=n, streamers=S, heads=len(heads), hub=(hub.x, hub.y), bytes_per_layer_per_chip=weight_bytes)
    if timeline:
        info["timeline"] = lambda: ttnn.to_torch(ts_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        info["head_timeline"] = lambda: ttnn.to_torch(hts_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return run, result, info


def run_case(mesh, layers_short, layers_long, sets=2):
    n = mesh.get_num_devices()
    d = Dims(n, mesh.dram_grid_size().x)
    weights = make_weights(d, sets)
    x0 = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(1)).bfloat16().float()
    times, checks = {}, {}
    for L in (layers_short, layers_long):
        # one build at a time: the L1 tensors of a build are only freed with its closures
        run, res, info = build(mesh, d, weights, x0, L)
        times[L] = timed(mesh, lambda: run(reset=False), reps=5)
        run()
        ttnn.synchronize_device(mesh)
        got = res()
        run()
        ttnn.synchronize_device(mesh)
        repeat_exact = torch.equal(got, res())
        ref = torch_reference(x0, d, weights, L)
        ref_bf = torch_reference(x0, d, weights, L, bf16_dataflow=True)
        checks[L] = dict(
            pcc=pcc(got, ref),
            rel_err=((got - ref).norm() / ref.norm()).item(),
            rel_err_vs_bf16_dataflow=((got - ref_bf).norm() / ref_bf.norm()).item(),
            repeat_exact=repeat_exact,
        )
        del run, res
        gc.collect()
    logger.info(info)
    per_layer = (times[layers_long] - times[layers_short]) / (layers_long - layers_short)
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


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_gdn_4chip(mesh_device):
    rec = run_case(mesh_device, 2, 10)
    assert all(c["pcc"] > 0.99 and c["repeat_exact"] for c in rec["checks"].values())
