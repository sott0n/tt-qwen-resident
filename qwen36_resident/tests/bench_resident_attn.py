# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident decode design: a full Qwen3.6-27B full-attention (gated attention) decoder layer on QB2.

One program per chip runs `layers` decoder layers back to back at one decode position, each an
attention half and an MLP half with a fabric all-reduce after each, TP-sharded over a 1 x N mesh
(per chip 6 query heads and 1 KV head). Per chip:
  - 32 streamer cores (the GDN-layer streamer kernels with no conv columns) run rmsnorm, stream
    their column slices of the [q | gate | k | v] projection, out-proj, gate|up and down through the
    weight ring, and send the projection row to the attention leader;
  - the attention leader normalizes and rotates q and k, puts the new k / v row into the KV tile of
    this position (read from and written back to the DRAM cache), multicasts q, combines the partials
    and applies the output gate;
  - attention workers each own a chunk of the KV cache in DRAM (bf8) and return a partial;
  - one hub core all-reduces the partials over fabric and multicasts the slots back.
Checked against a torch reference built from the same (dequantized) weights and caches.
"""
import gc
import json
import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.qwen36_resident.tests.bench_resident_gdn import FP32_ACC, STREAMER_FIDELITY, bf16, rmsnorm
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

OUT = os.environ.get("BENCH_OUT", "/tmp/resident_attn.jsonl")
KDIR = "models/experimental/qwen36_resident/kernels/"
NQ, NKV, HD, ROT = 24, 4, 256, 64
ROPE_THETA = 10_000_000.0
QKVG_DT, OUT_DT = ttnn.bfloat8_b, ttnn.bfloat8_b
WORKERS = int(os.environ.get("RESIDENT_ATTN_WORKERS", "32"))
FANIN = int(os.environ.get("RESIDENT_ATTN_FANIN", "8"))  # workers per reduction group
(SEM_SLOTS, SEM_ACT, SEM_GATHER, SEM_FLAG, SEM_HEADS, SEM_ROWS, SEM_Q, SEM_m, SEM_M, SEM_PART, SEM_ADDR) = range(11)


class Dims:
    def __init__(self, n, banks):
        self.n, self.banks = n, banks
        self.h = NQ // n  # query heads per chip
        assert NKV // n == 1, "one KV head per chip"
        self.Dt = HD // TILE
        self.qd = self.h * HD
        self.gate0, self.k0 = self.qd, 2 * self.qd
        self.v0 = self.k0 + HD
        self.row_cols = self.v0 + HD
        assert self.row_cols % (TILE * banks) == 0
        self.row_tiles = self.row_cols // TILE
        self.Ot = self.qd // TILE
        self.ic = INTER // n
        self.It = self.ic // TILE
        self.Ht = HIDDEN // TILE


def rope_tables(pos):
    inv = 1.0 / ROPE_THETA ** (torch.arange(0, ROT, 2, dtype=torch.float64) / ROT)
    ang = pos * inv
    return torch.cos(ang).float(), torch.sin(ang).float()  # [ROT / 2]


def cache_rows(t, pos):
    t[pos:] = 0
    return t


def make_weights(d, sets, pos, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    tp, r = divmod(pos, TILE)
    out = []
    for _ in range(sets):
        chips = []
        for _ in range(d.n):
            chips.append(
                dict(
                    Wq=quantize(rn(HIDDEN, d.row_cols) / math.sqrt(HIDDEN), QKVG_DT),
                    Wo=quantize(rn(d.qd, HIDDEN) * (0.5 / math.sqrt(d.qd)), OUT_DT),
                    G=quantize(rn(HIDDEN, d.ic) / math.sqrt(HIDDEN), GU_DT),
                    U=quantize(rn(HIDDEN, d.ic) / math.sqrt(HIDDEN), GU_DT),
                    D=quantize(rn(d.ic, HIDDEN) * (0.5 / math.sqrt(d.ic)), DOWN_DT),
                    # bf8 cache of the positions before this one (later rows zero)
                    K=cache_rows(quantize(rn(tp * TILE + TILE, HD), ttnn.bfloat8_b), pos),
                    V=cache_rows(quantize(rn(tp * TILE + TILE, HD), ttnn.bfloat8_b), pos),
                )
            )
        out.append(
            dict(
                chips=chips,
                gamma_in=bf16(1 + 0.1 * rn(HIDDEN)),
                gamma_post=bf16(1 + 0.1 * rn(HIDDEN)),
                wq=bf16(1 + 0.1 * rn(HD)),  # (1 + q_norm.weight)
                wk=bf16(1 + 0.1 * rn(HD)),
            )
        )
    return out


def rope(t, cos, sin):
    h = ROT // 2
    t = t.clone()
    a, b = t[..., :h].clone(), t[..., h:ROT].clone()
    t[..., :h] = a * cos - b * sin
    t[..., h:ROT] = b * cos + a * sin
    return t


def torch_reference(x0, d, weights, layers, pos, bf16_dataflow=False, attn_out=None):
    """x after `layers` layers; attn_out (a list) receives the last layer's gated attention output per chip."""
    r_ = bf16 if bf16_dataflow else (lambda t: t)
    tp, r = divmod(pos, TILE)
    cos, sin = rope_tables(pos)
    x = x0.clone()
    for l in range(layers):
        w = weights[l % len(weights)]
        h = bf16(rmsnorm(x, w["gamma_in"]))
        parts = []
        for c, cw in enumerate(w["chips"]):
            y = bf16(h @ cw["Wq"])
            q = y[: d.qd].view(d.h, HD)
            gate = y[d.gate0 : d.k0].view(d.h, HD)
            k, v = y[d.k0 : d.v0], y[d.v0 : d.row_cols]
            qn = rope(rmsnorm(q, w["wq"] / math.sqrt(HD)), cos, sin)
            kn = bf16(rope(rmsnorm(k, w["wk"]), cos, sin))
            K = torch.cat([cw["K"][:pos], kn[None]])
            V = torch.cat([cw["V"][:pos], v[None]])
            p = torch.softmax(qn @ K.T, dim=-1)
            o = bf16((p @ V) * torch.sigmoid(gate)).flatten()
            if attn_out is not None and l == layers - 1:
                attn_out.append(o)
            parts.append(r_(o @ cw["Wo"] + (x if c == 0 else 0)))
        x = r_(sum(parts))
        h = bf16(rmsnorm(x, w["gamma_post"]))
        x = r_(
            sum(
                r_(bf16(torch.nn.functional.silu(h @ cw["G"]) * (h @ cw["U"])) @ cw["D"] + (x if c == 0 else 0))
                for c, cw in enumerate(w["chips"])
            )
        )
    return x


def worker_rect(mesh, used, count):
    """The first rectangle of `count` free cores (row-major search, widest first)."""
    g = mesh.compute_with_storage_grid_size()
    for w in range(min(count, g.x), 0, -1):
        if count % w:
            continue
        hgt = count // w
        for y0 in range(g.y - hgt + 1):
            for x0 in range(g.x - w + 1):
                cells = [(x, y) for y in range(y0, y0 + hgt) for x in range(x0, x0 + w)]
                if not any(c in used for c in cells):
                    return [ttnn.CoreCoord(x, y) for x, y in cells]
    raise RuntimeError(f"no free rectangle of {count} cores")


def build(mesh, d, weights, x0, layers, pos, packet_bytes=4096, timeline=False):
    n, banks = d.n, d.banks
    tp, r = divmod(pos, TILE)
    groups, used = streamer_cores(mesh)
    cores = [c for grp in groups for c in grp]
    S = len(cores)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    xs, ys = [c.x for c in cores], [c.y for c in cores]
    x0r, x1r, y0r, y1r = min(xs), max(xs), min(ys), max(ys)
    hub = next(ttnn.CoreCoord(x, y) for x in range(x0r, x1r + 1) for y in range(y0r, y1r + 1) if (x, y) not in used)
    used.add((hub.x, hub.y))
    workers = worker_rect(mesh, used, WORKERS)
    used |= {(c.x, c.y) for c in workers}
    cgrid = mesh.compute_with_storage_grid_size()
    leader = next(ttnn.CoreCoord(x, y) for y in range(cgrid.y) for x in range(cgrid.x) if (x, y) not in used)
    hub_grid = ttnn.CoreRangeSet([ttnn.CoreRange(hub, hub)])
    lead_grid = ttnn.CoreRangeSet([ttnn.CoreRange(leader, leader)])
    work_grid = ttnn.CoreRangeSet([ttnn.CoreRange(workers[0], workers[-1])])
    tail_core = workers[-1]  # the last core of the rectangle is the tail core, the rest chunk workers
    tail_grid = ttnn.CoreRangeSet([ttnn.CoreRange(tail_core, tail_core)])
    chunk_grid = work_grid.subtract(tail_grid)
    prep_grid = lead_grid.merge(tail_grid)  # receive the projection row
    attn_grid = work_grid.merge(lead_grid)
    all_grid = grid.merge(hub_grid).merge(attn_grid)
    mc_dests = (x1r - x0r + 1) * (y1r - y0r + 1)
    p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0r, y0r))
    p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x1r, y1r))
    hub_phys = mesh.worker_core_from_logical_core(hub)
    lead_phys = mesh.worker_core_from_logical_core(leader)
    tail_phys = mesh.worker_core_from_logical_core(tail_core)
    w0p, w1p = (mesh.worker_core_from_logical_core(c) for c in (workers[0], workers[-1]))
    phys = [mesh.worker_core_from_logical_core(c) for c in cores]
    sets = len(weights)
    tiny = ttnn.Tile([1, TILE])
    full = ttnn.Tile([TILE, TILE])
    rep = ttnn.ReplicateTensorToMesh(mesh)
    per_chip = ttnn.ShardTensorToMesh(mesh, dim=0)

    def sharded(t, grid_, shard, dtype=ttnn.bfloat16, tile=tiny, mapper=rep):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            tile=tile,
            mesh_mapper=mapper,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid_, shard, ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )

    # ---- per-core column assignment (remainders of the projection and gate|up on different cores)
    q_bank, ht_bank, it_bank = d.row_tiles // banks, d.Ht // banks, d.It // banks
    g_extra = q_bank % PER_BANK
    nq_split, nd_split = split(q_bank, PER_BANK), split(ht_bank, PER_BANK)
    ng_split = split(it_bank, PER_BANK, g_extra)
    nq_max, nd_max, ng_max = (max(c for c, _ in sp) for sp in (nq_split, nd_split, ng_split))
    assert nq_max <= TILE and ng_max <= TILE

    # ---- streamer tensors
    slots0 = torch.zeros(S, n * HIDDEN)
    slots0[:, :HIDDEN] = x0
    slots_t = sharded(slots0, grid, [1, n * HIDDEN])
    slots_host = ttnn.from_torch(slots0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=tiny, mesh_mapper=rep)
    act_t = sharded(torch.zeros(S, d.ic), grid, [1, d.ic])
    o_in_t = sharded(torch.zeros(S, d.qd), grid, [1, d.qd])
    gamma_t = sharded(
        torch.cat([torch.cat([w["gamma_in"], w["gamma_post"]]) for w in weights]).repeat(S, 1),
        grid,
        [1, sets * 2 * HIDDEN],
    )
    blk = TILE * TILE
    conv_w_t = sharded(torch.zeros(S, sets * 4 * blk), grid, [1, sets * 4 * blk])
    hist_t = sharded(torch.zeros(S, sets * 3 * blk), grid, [1, sets * 3 * blk])

    # optional per-streamer timeline (16 uint32 per layer, see the GDN streamer reader / writer)
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
    lts_t = None
    if timeline:
        lts_t = ttnn.from_torch(
            torch.zeros(1, layers * 8, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(leader, leader)]),
                    [1, layers * 8],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
    lts_addr = lts_t.buffer_address() if timeline else 0
    wts_t = None
    if timeline:
        wts_t = ttnn.from_torch(
            torch.zeros(WORKERS, layers * 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(workers[0], workers[-1])]),
                    [1, layers * 4],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
    wts_addr = wts_t.buffer_address() if timeline else 0

    # ---- attention tensors
    Dt, heads = d.Dt, d.h
    # chunk workers: spread over the banks holding complete position tiles; the J workers of a bank
    # interleave its rows
    active = min(WORKERS - 1, tp)
    per_bank_rows = [len(range(b, tp, banks)) for b in range(banks)]
    live = [b for b in range(banks) if per_bank_rows[b]]
    chunks = []  # (bank, j, J, rows) per active worker
    for w_i in range(active):
        b, j = live[w_i % len(live)], w_i // len(live)
        J = len(range(w_i % len(live), active, len(live)))
        chunks.append((b, j, J, len(range(j, per_bank_rows[b], J))))
    assert all(c[3] > 0 for c in chunks)
    chunk_max = max([c[3] for c in chunks], default=1)
    # reduction tree over the partials of the active chunk workers and the tail core: groups of FANIN; a
    # group head sums its members' partials, the leader sums the group heads'
    nodes = list(range(active)) + [WORKERS - 1]
    groups_ = [nodes[g : g + FANIN] for g in range(0, len(nodes), FANIN)]
    slots_per_node = max(FANIN - 1, len(groups_))
    rows_t = sharded(torch.zeros(2, d.row_cols), prep_grid, [1, d.row_cols])
    q_t = sharded(torch.zeros(WORKERS * TILE, Dt * TILE), work_grid, [TILE, Dt * TILE], tile=full)
    M_t = sharded(torch.zeros((WORKERS + 1) * TILE, TILE), attn_grid, [TILE, TILE], tile=full)
    m_slots_t = ttnn.from_torch(
        torch.zeros(1, WORKERS * 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=rep,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(lead_grid, [1, WORKERS * 128], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    one_t = sharded(torch.ones(WORKERS * TILE, TILE), work_grid, [TILE, TILE], tile=full)
    mean_t = sharded(torch.full((2 * TILE, TILE), 1.0 / HD), prep_grid, [TILE, TILE], tile=full)
    rows32 = lambda v: v[None].expand(TILE, -1)
    qw_t = sharded(
        torch.cat([rows32(w["wq"] / math.sqrt(HD)) for w in weights], dim=1), lead_grid, [TILE, sets * HD], tile=full
    )
    kw_t = sharded(torch.cat([rows32(w["wk"]) for w in weights], dim=1), tail_grid, [TILE, sets * HD], tile=full)
    cos, sin = rope_tables(pos)
    rope_t = sharded(
        torch.cat([rows32(cos), rows32(sin), rows32(-sin)], dim=1).repeat(2, 1), prep_grid, [TILE, 3 * TILE], tile=full
    )

    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    # KV cache: position tile t in bank t % banks, row t // banks of that bank's shard
    rows_per_bank = tp // banks + 1

    def cache(key, s):
        t = torch.zeros(n, banks * rows_per_bank * TILE, HD)
        for c in range(n):
            src = weights[s]["chips"][c][key]
            for pt in range(src.shape[0] // TILE):
                at = (pt % banks) * rows_per_bank + pt // banks
                t[c, at * TILE : (at + 1) * TILE] = src[pt * TILE : (pt + 1) * TILE]
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [rows_per_bank * TILE, HD], ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc, mesh_mapper=per_chip
        )

    kv_t = [(cache("K", s), cache("V", s)) for s in range(sets)]

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

    # ---- DRAM weights (per chip), a bank's cores' blocks interleaved
    entries = [(d.Ht, QKVG_DT), (d.Ot, OUT_DT), (d.Ht, GU_DT), (d.It, DOWN_DT)]
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
    full_page = TILE * TILE * 2

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

    def cb(idx, grid_, pages, dt=ttnn.bfloat16, page=full_page):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=grid_,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
        )

    def from_tensor(idx, t):
        return ttnn.cb_descriptor_from_sharded_tensor(idx, t)

    gamma_cb = from_tensor(11, gamma_t)
    gamma_cb.format_descriptors = list(gamma_cb.format_descriptors) + [
        ttnn.CBFormatDescriptor(buffer_index=14, data_format=ttnn.bfloat16, page_size=full_page)
    ]
    f32 = (ttnn.float32, 4096)
    kv_page = TILE_BYTES[ttnn.bfloat8_b]
    cbs = [
        from_tensor(0, slots_t),
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
        cb(7, grid, 1, ttnn.uint32, 16),
        from_tensor(8, act_t),
        tiny_cb(9, nd_max),
        tiny_cb(10, nd_max),
        gamma_cb,
        aliased(15, 27, TILE),
        aliased(26, 28, TILE),
        aliased(16, 29, TILE),
        aliased(17, 22, TILE),
        aliased(18, 23, TILE),
        with_full_view(from_tensor(19, conv_w_t), 24),
        with_full_view(from_tensor(20, hist_t), 25),
        from_tensor(21, o_in_t),
        # attention workers (chunk workers and the tail core)
        from_tensor(0, q_t),
        from_tensor(3, M_t),
        from_tensor(4, one_t),
        cb(1, work_grid, 2 * chunk_max * Dt, ttnn.bfloat8_b, kv_page),
        cb(2, work_grid, 2 * chunk_max * Dt, ttnn.bfloat8_b, kv_page),
        cb(5, work_grid, chunk_max, *f32),
        cb(6, work_grid, chunk_max),
        cb(7, work_grid, 1),
        cb(8, work_grid, Dt + 1),
        cb(23, attn_grid, slots_per_node * (Dt + 1)),
        cb(25, attn_grid, Dt + 1),
        # leader and tail core: q / k preparation
        cb(13, prep_grid, Dt),
        cb(14, prep_grid, 2),
        from_tensor(17, rope_t),
        from_tensor(29, mean_t),
        # leader
        cb(2, lead_grid, 1, page=heads * Dt * 64),
        cb(9, lead_grid, Dt),
        cb(10, lead_grid, Dt),
        from_tensor(15, qw_t),
        cb(24, lead_grid, Dt),
        cb(26, lead_grid, Dt),
        cb(27, lead_grid, Dt),
        cb(30, lead_grid, 1),
        # tail core
        cb(11, tail_grid, Dt),
        cb(12, tail_grid, 2),
        from_tensor(16, kw_t),
        cb(18, tail_grid, Dt),
        cb(19, tail_grid, Dt),
        cb(20, tail_grid, Dt),
        cb(21, tail_grid, Dt),
        cb(22, tail_grid, 1),
        cb(28, tail_grid, 2 * Dt, ttnn.bfloat8_b, kv_page),
        cb(31, tail_grid, 2 * Dt, ttnn.bfloat8_b, kv_page),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_grid, initial_value=0) for i in range(11)]

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

    NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
    NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default
    attn_ct = [layers, sets, WORKERS, S, chunk_max, heads, Dt, ROT // TILE]
    attn_ct += [SEM_ROWS, SEM_Q, SEM_m, SEM_M, SEM_PART, SEM_HEADS, f32_bits(EPS), SEM_ADDR, slots_per_node]
    attn_compute = ttnn.ComputeConfigDescriptor()
    attn_compute.math_fidelity = ttnn.MathFidelity.HiFi4
    attn_compute.fp32_dest_acc_en = True
    attn_compute.dst_full_sync_en = True  # rope holds 8 DST tiles
    # the workers only run matmuls with a bf8 operand in SrcA (k, v), for which HiFi2 matches HiFi4
    work_compute = ttnn.ComputeConfigDescriptor()
    work_compute.math_fidelity = ttnn.MathFidelity.HiFi2
    work_compute.fp32_dest_acc_en = True

    mesh_pd = ttnn.MeshProgramDescriptor()
    for chip in range(n):
        coord = ttnn.MeshCoordinate(0, chip)
        ct = [layers, sets, n, S, d.Ht, d.It, chip, ring_bytes, f32_bits(EPS), f32_bits(1 / math.sqrt(HIDDEN))]
        ct += [SEM_SLOTS, SEM_ACT, SEM_GATHER]
        for (Kt, _), (sb, pages, page, block) in zip(entries, geo):
            ct += [Kt, sb, pages, page, block]
        ct += [ng_max, nd_max, nq_max, 0, d.Ot, 2, SEM_HEADS, d.row_tiles, SEM_ROWS, 1]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        peers = [v for p in phys for v in (p.x, p.y)]
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
                    lead_phys.x,
                    lead_phys.y,
                    tail_phys.x,
                    tail_phys.y,
                ]
                + peers
                + [ts_addr]
            )
            compute_rt[c.x][c.y] = [ng, nd, bank * ht_bank + dfirst, nq, 0]

        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = STREAMER_FIDELITY
        compute.fp32_dest_acc_en = FP32_ACC
        compute.dst_full_sync_en = FP32_ACC
        hub_rt = ttnn.RuntimeArgs()
        hub_rt[hub.x][hub.y] = [
            hub_slots.buffer_address(),
            ttnn.get_global_semaphore_address(ccl_sem),
            slots_t.buffer_address(),
        ]
        hub_ct = [n, chip, HIDDEN * 2, packet_bytes, 2 * layers, S, SEM_GATHER, SEM_SLOTS]
        hub_ct += [p0.x, p0.y, p1.x, p1.y, mc_dests, SEM_FLAG]

        wr, ww, wc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        kv_addrs = [t.buffer_address() for pair in kv_t for t in pair]
        wphys = [mesh.worker_core_from_logical_core(c) for c in workers]
        parent = {}  # worker -> (parent phys, slot)
        for g_i, grp in enumerate(groups_):
            parent[grp[0]] = (lead_phys, g_i)
            for k, w_i in enumerate(grp[1:]):
                parent[w_i] = (wphys[grp[0]], k)
        kids = {grp[0]: grp[1:] for grp in groups_}
        for w_i, c in enumerate(workers):
            is_tail = w_i == WORKERS - 1
            role = 2 if is_tail else int(w_i < active)
            b_, j_, J_, cnt = (
                (tp % banks, tp // banks, 1, 1) if is_tail else (chunks[w_i] if w_i < active else (0, 0, 1, 0))
            )
            ch = kids.get(w_i, [])
            pp, slot = parent.get(w_i, (lead_phys, 0))
            m_slot = active if is_tail else w_i
            wr[c.x][c.y] = (
                [role, cnt, b_, j_, J_]
                + kv_addrs
                + [len(ch), wts_addr]
                + ([rows_t.buffer_address(), r] if is_tail else [])
            )
            ww[c.x][c.y] = (
                [role, m_slot, lead_phys.x, lead_phys.y, m_slots_t.buffer_address(), pp.x, pp.y, slot, len(ch)]
                + [v for k in ch for v in (wphys[k].x, wphys[k].y)]
                + [wts_addr]
                + ([tp % banks, tp // banks] + kv_addrs if is_tail else [])
            )
            wc[c.x][c.y] = [role, cnt, len(ch)]
        lr, lw, lc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        lr[leader.x][leader.y] = [rows_t.buffer_address(), len(groups_), lts_addr]
        lw[leader.x][leader.y] = (
            [q_t.buffer_address(), M_t.buffer_address(), m_slots_t.buffer_address(), active + 1]
            + [w0p.x, w0p.y, w1p.x, w1p.y, o_in_t.buffer_address()]
            + peers
            + [len(groups_)]
            + [v for grp in groups_ for v in (wphys[grp[0]].x, wphys[grp[0]].y)]
            + [lts_addr]
        )
        lc[leader.x][leader.y] = [len(groups_)]

        kernels = [
            kernel("gdn_streamer_reader.cpp", grid, ct, reader_rt, dm(NCRISC, NOC0)),
            kernel("gdn_streamer_writer.cpp", grid, ct, writer_rt, dm(BRISC, NOC1)),
            kernel("gdn_streamer_compute.cpp", grid, ct, compute_rt, compute),
            kernel("mlp_hub.cpp", hub_grid, hub_ct, hub_rt, dm(BRISC, NOC0)),
            kernel("attn_worker_reader.cpp", work_grid, attn_ct, wr, dm(NCRISC, NOC1)),
            kernel("attn_worker_writer.cpp", work_grid, attn_ct, ww, dm(BRISC, NOC0)),
            kernel("attn_worker_compute.cpp", chunk_grid, attn_ct, wc, work_compute),
            kernel("attn_worker_compute.cpp", tail_grid, attn_ct, wc, attn_compute),
            kernel("attn_leader_reader.cpp", lead_grid, attn_ct, lr, dm(NCRISC, NOC1)),
            kernel("attn_leader_writer.cpp", lead_grid, attn_ct, lw, dm(BRISC, NOC0)),
            kernel("attn_leader_compute.cpp", lead_grid, attn_ct, lc, attn_compute),
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

    io = ([ts_t, lts_t, wts_t] if timeline else []) + [
        slots_t,
        act_t,
        o_in_t,
        gamma_t,
        conv_w_t,
        hist_t,
        rows_t,
        q_t,
        M_t,
        m_slots_t,
        one_t,
        mean_t,
    ]
    io += [qw_t, kw_t, rope_t, hub_slots] + [t for pair in kv_t for t in pair]
    io += [t for ws in w_dram for t in ws]

    def run(reset=True, keep=(ccl_sem,)):
        if reset:
            ttnn.copy_host_to_device_tensor(slots_host, slots_t)
        ttnn.generic_op(io, mesh_pd)

    def attn_output():
        """the last layer's gated attention output as received by streamer 0 of each chip"""
        o = ttnn.to_torch(o_in_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
        return o.reshape(n, S, -1)[:, 0]

    def result():
        got = ttnn.to_torch(hub_slots, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
        p = (2 * layers - 1) & 1
        return got[: 2 * n][p * n : (p + 1) * n].sum(0)

    weight_bytes = sum(
        k * c_ * TILE_BYTES[dt] / (TILE * TILE)
        for (k, c_, dt) in (
            (HIDDEN, d.row_cols, QKVG_DT),
            (d.qd, HIDDEN, OUT_DT),
            (HIDDEN, 2 * d.ic, GU_DT),
            (d.ic, HIDDEN, DOWN_DT),
        )
    )
    kv_bytes = 2 * tp * TILE * HD * TILE_BYTES[ttnn.bfloat8_b] / (TILE * TILE)
    info = dict(
        chips=n,
        streamers=S,
        workers=WORKERS,
        active_workers=active,
        pos=pos,
        leader=(leader.x, leader.y),
        hub=(hub.x, hub.y),
        bytes_per_layer_per_chip=weight_bytes + kv_bytes,
        kv_bytes_per_layer_per_chip=kv_bytes,
    )
    info["attn_output"] = attn_output
    if timeline:
        info["timeline"] = lambda: ttnn.to_torch(ts_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        info["worker_timeline"] = lambda: ttnn.to_torch(wts_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        info["leader_timeline"] = lambda: ttnn.to_torch(lts_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return run, result, info


def run_case(mesh, layers_short, layers_long, pos, sets=2):
    n = mesh.get_num_devices()
    d = Dims(n, mesh.dram_grid_size().x)
    weights = make_weights(d, sets, pos)
    x0 = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(1)).bfloat16().float()
    times, checks = {}, {}
    for L in (layers_short, layers_long):
        run, res, info = build(mesh, d, weights, x0, L, pos)
        times[L] = timed(mesh, lambda: run(reset=False), reps=5)
        run()
        ttnn.synchronize_device(mesh)
        got = res()
        o_got = info.pop("attn_output")()
        run()
        ttnn.synchronize_device(mesh)
        repeat_exact = torch.equal(got, res())
        o_ref = []
        ref = torch_reference(x0, d, weights, L, pos, attn_out=o_ref)
        o_ref = torch.stack(o_ref)
        ref_bf = torch_reference(x0, d, weights, L, pos, bf16_dataflow=True)
        checks[L] = dict(
            pcc=pcc(got, ref),
            rel_err=((got - ref).norm() / ref.norm()).item(),
            rel_err_vs_bf16_dataflow=((got - ref_bf).norm() / ref_bf.norm()).item(),
            repeat_exact=repeat_exact,
            attn_pcc=pcc(o_got, o_ref),
            attn_rel_err=((o_got - o_ref).norm() / o_ref.norm()).item(),
        )
        del run, res
        gc.collect()
    per_layer = (times[layers_long] - times[layers_short]) / (layers_long - layers_short)
    rec = dict(
        info,
        us_per_layer=per_layer * 1e6,
        GBps_per_chip=info["bytes_per_layer_per_chip"] / per_layer / 1e9,
        checks=checks,
    )
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


@pytest.mark.parametrize("pos", [int(p) for p in os.environ.get("RESIDENT_ATTN_POS", "8100").split(",")])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_attn_4chip(mesh_device, pos):
    rec = run_case(mesh_device, 2, 10, pos)
    assert all(c["pcc"] > 0.99 and c["attn_pcc"] > 0.99 and c["repeat_exact"] for c in rec["checks"].values())
