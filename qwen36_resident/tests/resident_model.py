# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident Qwen3.6-27B decode on QB2: builds the one-program-per-chip decode step and its torch reference.

A decode step runs `layers` decoder layers (layer l is a full-attention layer when attn_interval > 0 and
(l + 1) % attn_interval == 0, otherwise a Gated DeltaNet layer) and optionally the lm_head, TP-sharded over
a 1 x N mesh. Per chip:
  - 32 streamer cores (4 per DRAM bank) hold a replica of x, run the rmsnorms, stream their column slices
    of every weight through the weight ring and run the projections, conv1d + silu and the MLP;
  - 12 GDN head cores (one per local value head) run the delta-rule recurrence;
  - the attention leader, the tail core and 30 chunk workers run the gated attention over the KV cache;
  - one hub core all-reduces the partials over fabric and multicasts the slots back.
Recurrent state, conv history and KV cache live in DRAM and are updated in place every step; the host
writes the token state (position, x0 = embedding, rope tables) before a step and reads x / logits after.

Weights are given per kind with `copies` distinct layers each ("gdn", "attn": mixer weights of the k-th
layer of that kind use copy k % copies; "mlp", "out": per layer l, copy l % copies). Norm weights are
folded into the following projection. The torch reference applies the same (dequantized) weights and
bf16 rounding points.
"""
import math

import numpy as np
import torch

import ttnn
from models.experimental.qwen36_resident.tests.bench_resident_mlp import (
    EPS,
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
    quantize,
    rect_cover,
    split,
    streamer_cores,
)

KDIR = "models/experimental/qwen36_resident/kernels/"
# GDN
NK, NV, DK, DV, CONV_K = 16, 48, 128, 128, 4
# attention
NQ, NKV, HD, ROT = 24, 4, 256, 64
ROPE_THETA = 10_000_000.0
PROJ_DT, OUT_DT, GU_DT, DOWN_DT, HEAD_DT = (
    ttnn.bfloat8_b,
    ttnn.bfloat8_b,
    ttnn.bfloat4_b,
    ttnn.bfloat8_b,
    ttnn.bfloat8_b,
)
WORKERS, FANIN = 32, 8
(
    SEM_SLOTS,
    SEM_ACT,
    SEM_GATHER,
    SEM_FLAG,
    SEM_HEADS,
    SEM_ROWS,
    SEM_Q,
    SEM_m,
    SEM_M,
    SEM_PART,
    SEM_ADDR,
    SEM_LOCAL,
) = range(12)
HEAD_FP32_CBS = (5, 6, 7, 15, 30, 31)
TOK_X0 = 64
KV_ROW = (HD // TILE) * TILE_BYTES[ttnn.bfloat8_b]


def bf16(t):
    return t.bfloat16().float()


def rmsnorm(x, w=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y if w is None else y * w


def is_attn(l, interval):
    return interval > 0 and (l + 1) % interval == 0


class Dims:
    def __init__(self, n, banks, vocab=0):
        self.n, self.banks = n, banks
        step = TILE * banks
        pad = lambda c: (c + step - 1) // step * step
        # GDN projection row: q | k | v (conv) | z | a | b
        self.nk, self.nv = NK // n, NV // n
        self.gq, self.gv = self.nk * DK, self.nv * DV
        self.conv_ch = 2 * self.gq + self.gv
        self.z0 = self.conv_ch
        self.a0 = self.z0 + self.gv
        self.b0 = self.a0 + self.nv
        self.g_cols = pad(self.b0 + self.nv)
        self.conv_tiles = self.conv_ch // TILE
        # attention projection row: q heads | gates | k | v
        self.h = NQ // n
        assert NKV // n == 1, "one KV head per chip"
        self.Dt = HD // TILE
        self.aq = self.h * HD
        self.gate0, self.k0 = self.aq, 2 * self.aq
        self.v0 = self.k0 + HD
        self.a_cols = pad(self.v0 + HD)
        assert self.aq == self.gv, "the out-projections of both mixers take the same K"
        self.Ot = self.gv // TILE
        self.ic = INTER // n
        self.It = self.ic // TILE
        self.Ht = HIDDEN // TILE
        self.vocab = pad(vocab // n) if vocab else 0  # lm_head columns per chip (padded)


def rope_tables(pos):
    inv = 1.0 / ROPE_THETA ** (torch.arange(0, ROT, 2, dtype=torch.float64) / ROT)
    ang = pos * inv
    return torch.cos(ang).float(), torch.sin(ang).float()


def rope(t, cos, sin):
    h = ROT // 2
    t = t.clone()
    a, b = t[..., :h].clone(), t[..., h:ROT].clone()
    t[..., :h] = a * cos - b * sin
    t[..., h:ROT] = b * cos + a * sin
    return t


def quantize_rows(t, dtype=ttnn.bfloat8_b):
    """bf8 of each row on its own (bf8 shares exponents within 16 elements of a row)."""
    rows = t.reshape(-1, t.shape[-1])
    padded = torch.zeros((rows.shape[0] + TILE - 1) // TILE * TILE, rows.shape[1])
    padded[: rows.shape[0]] = rows
    return quantize(padded, dtype)[: rows.shape[0]].reshape(t.shape)


class State:
    """Decode state (torch): GDN recurrent states / conv histories per GDN copy, KV caches per attention
    copy (positions < pos, bf8), shared by the device model (as its initial DRAM contents) and the
    reference (updated in place by torch_step)."""

    def __init__(self, d, gdn_copies, attn_copies, pos, max_pos, seed=1, zero=False):
        g = torch.Generator().manual_seed(seed)
        rn = (lambda *s: torch.zeros(*s)) if zero else (lambda *s: torch.randn(*s, generator=g))
        self.pos, self.max_pos = pos, max_pos
        # [copy][chip] fp32 [nv, DK, DV]
        self.gdn = [[0.05 * rn(d.nv, DK, DV) for _ in range(d.n)] for _ in range(gdn_copies)]
        # [copy][chip] [3, conv_ch] inputs at pos - 3, pos - 2, pos - 1 (bf16)
        self.hist = [[bf16(rn(CONV_K - 1, d.conv_ch)) for _ in range(d.n)] for _ in range(gdn_copies)]
        # [copy][chip] bf8 [max_pos, HD], rows >= pos zero
        self.K, self.V = [], []
        for _ in range(attn_copies):
            ks, vs = [], []
            for _ in range(d.n):
                k, v = quantize_rows(rn(max_pos, HD)), quantize_rows(rn(max_pos, HD))
                k[pos:], v[pos:] = 0, 0
                ks.append(k)
                vs.append(v)
            self.K.append(ks)
            self.V.append(vs)


def random_weights(d, gdn_copies, attn_copies, mlp_copies, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    w = dict(gdn=[], attn=[], mlp=[], out=[])
    for _ in range(gdn_copies):
        chips = []
        for _ in range(d.n):
            Wq = torch.zeros(HIDDEN, d.g_cols)
            Wq[:, : d.b0 + d.nv] = rn(HIDDEN, d.b0 + d.nv) / math.sqrt(HIDDEN)
            chips.append(
                dict(
                    Wq=quantize(Wq, PROJ_DT),
                    taps=bf16(0.5 * rn(d.conv_ch, CONV_K)),
                    dt=0.5 * rn(d.nv),
                    neg_a=-torch.exp(torch.rand(d.nv, generator=g) * 2 - 1),
                )
            )
        w["gdn"].append(dict(chips=chips, norm_w=bf16(1 + 0.1 * rn(DV))))
    for _ in range(attn_copies):
        chips = []
        for _ in range(d.n):
            Wq = torch.zeros(HIDDEN, d.a_cols)
            Wq[:, : d.v0 + HD] = rn(HIDDEN, d.v0 + HD) / math.sqrt(HIDDEN)
            chips.append(dict(Wq=quantize(Wq, PROJ_DT)))
        w["attn"].append(dict(chips=chips, wq=bf16(1 + 0.1 * rn(HD)), wk=bf16(1 + 0.1 * rn(HD))))
    for _ in range(mlp_copies):
        w["out"].append([quantize(rn(d.gv, HIDDEN) * (0.5 / math.sqrt(d.gv)), OUT_DT) for _ in range(d.n)])
        w["mlp"].append(
            [
                dict(
                    G=quantize(rn(HIDDEN, d.ic) / math.sqrt(HIDDEN), GU_DT),
                    U=quantize(rn(HIDDEN, d.ic) / math.sqrt(HIDDEN), GU_DT),
                    D=quantize(rn(d.ic, HIDDEN) * (0.5 / math.sqrt(d.ic)), DOWN_DT),
                )
                for _ in range(d.n)
            ]
        )
    if d.vocab:
        w["head"] = [quantize(rn(HIDDEN, d.vocab) / math.sqrt(HIDDEN), HEAD_DT) for _ in range(d.n)]
    return w


def torch_step(x0, d, w, st, layers, interval, lm_head=False, attn_out=None):
    """One decode step at st.pos (fp32 reference, bf16 at the device's rounding points); updates st in place
    (states, histories, KV rows at pos) and returns x (and the per-chip logits with lm_head)."""
    pos = st.pos
    cos, sin = rope_tables(pos)
    x = x0.clone()
    g_i = a_i = 0
    for l in range(layers):
        h = bf16(rmsnorm(x))
        parts = []
        if is_attn(l, interval):
            c = a_i % len(w["attn"])
            wa = w["attn"][c]
            for chip, cw in enumerate(wa["chips"]):
                y = bf16(h @ cw["Wq"])
                q = y[: d.aq].view(d.h, HD)
                gate = y[d.gate0 : d.k0].view(d.h, HD)
                k, v = y[d.k0 : d.v0], y[d.v0 : d.v0 + HD]
                qn = rope(rmsnorm(q, wa["wq"] / math.sqrt(HD)), cos, sin)
                kn = bf16(rope(rmsnorm(k, wa["wk"]), cos, sin))
                K = torch.cat([st.K[c][chip][:pos], kn[None]])
                V = torch.cat([st.V[c][chip][:pos], v[None]])
                p = torch.softmax(qn @ K.T, dim=-1)
                o = bf16((p @ V) * torch.sigmoid(gate)).flatten()
                if attn_out is not None:
                    attn_out.append((l, chip, o))
                st.K[c][chip][pos] = quantize_rows(kn[None])[0]
                st.V[c][chip][pos] = quantize_rows(v[None])[0]
                parts.append(o @ w["out"][l % len(w["out"])][chip] + (x if chip == 0 else 0))
            a_i += 1
        else:
            c = g_i % len(w["gdn"])
            wg = w["gdn"][c]
            scale = DK**-0.5
            for chip, cw in enumerate(wg["chips"]):
                y = bf16(h @ cw["Wq"])
                qkv = y[: d.conv_ch]
                window = torch.cat([st.hist[c][chip], qkv[None]], 0)  # oldest first
                st.hist[c][chip] = window[1:]
                conv = bf16(torch.nn.functional.silu((window * cw["taps"].T).sum(0)))
                q, k, v = conv[: d.gq], conv[d.gq : 2 * d.gq], conv[2 * d.gq : d.conv_ch]
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
                    S = st.gdn[c][chip][hh]
                    delta = beta * (v[hh * DV : (hh + 1) * DV] - (decay * kn) @ S)
                    S = decay * S + torch.outer(kn, delta)
                    st.gdn[c][chip][hh] = S
                    on = rmsnorm(qn @ S, wg["norm_w"])
                    outs.append(bf16(on * torch.nn.functional.silu(z[hh * DV : (hh + 1) * DV])))
                o = torch.cat(outs)
                if attn_out is not None:
                    attn_out.append((l, chip, o))
                parts.append(o @ w["out"][l % len(w["out"])][chip] + (x if chip == 0 else 0))
            g_i += 1
        x = sum(parts)
        h = bf16(rmsnorm(x))
        m = w["mlp"][l % len(w["mlp"])]
        x = sum(
            bf16(torch.nn.functional.silu(h @ cm["G"]) * (h @ cm["U"])) @ cm["D"] + (x if chip == 0 else 0)
            for chip, cm in enumerate(m)
        )
    st.pos += 1
    if lm_head:
        h = bf16(rmsnorm(x))
        return x, [h @ wh for wh in w["head"]]
    return x


class BankLayout:
    """interleaved_bank_layout (bench_resident_mlp) as one gather: the tile order is computed once per
    (Kt, core_cols, sb) and applied to every weight of that shape."""

    def __init__(self, Kt, core_cols, sb):
        self.Kt, self.sb = Kt, sb
        nkb = Kt // sb
        self.width = PER_BANK * max(len(c) for cols in core_cols for c in cols)
        self.banks = len(core_cols)
        rows, cols_ = [], []
        for cols in core_cols:
            src_k = torch.full((Kt * self.width,), -1, dtype=torch.long)
            src_c = torch.full((Kt * self.width,), -1, dtype=torch.long)
            for j, cj in enumerate(cols):
                for t, col in enumerate(cj):
                    for kb in range(nkb):
                        slot = (t * nkb + kb) * PER_BANK + j
                        lin = slot * sb + torch.arange(sb)
                        src_k[lin] = kb * sb + torch.arange(sb)
                        src_c[lin] = col
            rows.append(src_k)
            cols_.append(src_c)
        self.src_k, self.src_c = torch.stack(rows), torch.stack(cols_)  # [banks, Kt * width]

    def __call__(self, w):
        """w [K, N] -> [K, banks * width] (see interleaved_bank_layout)"""
        K, N = w.shape
        Kt, Nt = K // TILE, N // TILE
        tiles = w.reshape(Kt, TILE, Nt, TILE).permute(0, 2, 1, 3)
        tiles = torch.cat([tiles, torch.zeros(Kt, 1, TILE, TILE, dtype=w.dtype)], dim=1)  # column Nt = zeros
        k = self.src_k.clamp(min=0)
        c = torch.where(self.src_c < 0, torch.full_like(self.src_c, Nt), self.src_c)
        g = tiles[k, c]  # [banks, Kt * width, 32, 32]
        g = g.reshape(self.banks, Kt, self.width, TILE, TILE).permute(1, 3, 0, 2, 4)
        return g.reshape(K, self.banks * self.width * TILE)


def worker_rect(mesh, used, count):
    """The first rectangle of `count` free cores (row-major search, widest first)."""
    g = mesh.compute_with_storage_grid_size()
    for wd in range(min(count, g.x), 0, -1):
        if count % wd:
            continue
        hgt = count // wd
        for y0 in range(g.y - hgt + 1):
            for x0 in range(g.x - wd + 1):
                cells = [(x, y) for y in range(y0, y0 + hgt) for x in range(x0, x0 + wd)]
                if not any(c in used for c in cells):
                    return [ttnn.CoreCoord(x, y) for x, y in cells]
    raise RuntimeError(f"no free rectangle of {count} cores")


def chunk_of(w, tp, banks):
    """Mirror of attn_common.hpp chunk_of: (bank, j, J, rows) of chunk worker w."""
    live = min(tp, banks)
    if live == 0:
        return 0, 0, 1, 0
    b, j = w % live, w // live
    J = (WORKERS - 1 - b + live - 1) // live
    rows = (tp - b + banks - 1) // banks
    return b, j, J, ((rows - j + J - 1) // J if rows > j else 0)


def tile_bytes_rows_replicated(v):
    """bf16 32x32 tile (TILE layout bytes) whose 32 rows all equal v [32]."""
    t = torch.tensor(v, dtype=torch.float32).bfloat16().view(torch.int16).numpy().astype(np.uint16)
    face_rows = [np.tile(t[:16], (16, 1)), np.tile(t[16:], (16, 1))]
    return np.concatenate([face_rows[0], face_rows[1], face_rows[0], face_rows[1]]).reshape(-1)


def bf16_bits(t):
    return t.flatten().bfloat16().view(torch.int16).numpy().astype(np.uint16)


class ResidentModel:
    def __init__(self, mesh, d, w, st, layers, interval, lm_head=False, timeline=False, packet_bytes=4096):
        self.mesh, self.d, self.layers, self.interval, self.lm_head = mesh, d, layers, interval, lm_head
        n, banks = d.n, d.banks
        self.n_attn = sum(is_attn(l, interval) for l in range(layers))
        self.n_gdn = layers - self.n_attn
        gdn_copies, attn_copies = max(len(w["gdn"]), 1), max(len(w["attn"]), 1)
        mlp_copies = len(w["mlp"])
        assert len(w["out"]) == mlp_copies

        # ---- cores
        groups, used = streamer_cores(mesh)
        cores = [c for grp in groups for c in grp]
        S = len(cores)
        self.S = S
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        xs, ys = [c.x for c in cores], [c.y for c in cores]
        x0r, x1r, y0r, y1r = min(xs), max(xs), min(ys), max(ys)
        hub = next(ttnn.CoreCoord(x, y) for x in range(x0r, x1r + 1) for y in range(y0r, y1r + 1) if (x, y) not in used)
        used.add((hub.x, hub.y))
        workers = worker_rect(mesh, used, WORKERS)
        used |= {(c.x, c.y) for c in workers}
        cgrid = mesh.compute_with_storage_grid_size()
        free = [ttnn.CoreCoord(x, y) for y in range(cgrid.y) for x in range(cgrid.x) if (x, y) not in used]
        leader, heads = free[0], free[1 : 1 + d.nv]
        tail_core = workers[-1]
        C = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])
        hub_grid, lead_grid, tail_grid, head_grid = C([hub]), C([leader]), C([tail_core]), C(heads)
        work_grid = ttnn.CoreRangeSet([ttnn.CoreRange(workers[0], workers[-1])])
        chunk_grid = work_grid.subtract(tail_grid)
        prep_grid = lead_grid.merge(tail_grid)
        attn_grid = work_grid.merge(lead_grid)
        mixer_grid = head_grid.merge(prep_grid)
        all_grid = grid.merge(hub_grid).merge(attn_grid).merge(head_grid)
        phys = lambda c: mesh.worker_core_from_logical_core(c)
        sphys = [phys(c) for c in cores]
        hub_p, lead_p, tail_p = phys(hub), phys(leader), phys(tail_core)
        head_p = [phys(c) for c in heads]
        wphys = [phys(c) for c in workers]

        tiny, full = ttnn.Tile([1, TILE]), ttnn.Tile([TILE, TILE])
        rep, per_chip = ttnn.ReplicateTensorToMesh(mesh), ttnn.ShardTensorToMesh(mesh, dim=0)
        dram = ttnn.DRAM_MEMORY_CONFIG
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})

        # Every setup tensor is synchronized before the next is created: tilize ops queued behind each other
        # while L1 buffers are still being allocated were seen to hang the device.
        def settled(t):
            ttnn.synchronize_device(mesh)
            return t

        def l1(t, grid_, shard, dtype=ttnn.bfloat16, tile=tiny, mapper=rep, layout=ttnn.TILE_LAYOUT):
            return settled(_l1(t, grid_, shard, dtype, tile, mapper, layout))

        def _l1(t, grid_, shard, dtype=ttnn.bfloat16, tile=tiny, mapper=rep, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=layout,
                device=mesh,
                tile=tile if layout == ttnn.TILE_LAYOUT else None,
                mesh_mapper=mapper,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid_, shard, ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )

        def dram_t(t, dtype, layout=ttnn.TILE_LAYOUT, mapper=per_chip):
            return settled(
                ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=dram, mesh_mapper=mapper)
            )

        # ---- column assignment (the remainders of the projections and gate|up on different cores)
        qg_bank, qa_bank = d.g_cols // TILE // banks, d.a_cols // TILE // banks
        ht_bank, it_bank = d.Ht // banks, d.It // banks
        v_bank = d.vocab // TILE // banks if d.vocab else 0
        nqg_split, nqa_split = split(qg_bank, PER_BANK), split(qa_bank, PER_BANK)
        nd_split = split(ht_bank, PER_BANK)
        g_extra = qg_bank % PER_BANK
        ng_split = split(it_bank, PER_BANK, g_extra)
        nv_split = split(v_bank, PER_BANK) if v_bank else [(0, 0)] * PER_BANK
        mx = lambda sp: max(c for c, _ in sp)
        nq_max, nd_max, ng_max = max(mx(nqg_split), mx(nqa_split)), mx(nd_split), mx(ng_split)
        nv_max = max(mx(nv_split), 1)
        assert nq_max <= TILE and ng_max <= TILE

        # ---- DRAM weights: per entry, copies stacked along K (a fixed per-bank stride)
        entries = [
            (d.Ht, PROJ_DT, d.g_cols),
            (d.Ht, PROJ_DT, d.a_cols),
            (d.Ot, OUT_DT, HIDDEN),
            (d.Ht, GU_DT, 2 * d.ic),
            (d.It, DOWN_DT, HIDDEN),
            (d.Ht, HEAD_DT, d.vocab),
        ]
        geo = [block_geometry(Kt, dt) for Kt, dt, _ in entries]
        cols_of = [
            bank_core_cols(d.g_cols // TILE, banks),
            bank_core_cols(d.a_cols // TILE, banks),
            bank_core_cols(d.Ht, banks),
            gate_up_core_cols(d.It, banks, g_extra),
            bank_core_cols(d.Ht, banks),
            bank_core_cols(d.vocab // TILE, banks) if d.vocab else None,
        ]
        per_entry_mats = [
            [[c["Wq"] for c in wg["chips"]] for wg in w["gdn"]],
            [[c["Wq"] for c in wa["chips"]] for wa in w["attn"]],
            [list(o) for o in w["out"]],
            [[torch.cat([cm["G"], cm["U"]], dim=1) for cm in m] for m in w["mlp"]],
            [[cm["D"] for cm in m] for m in w["mlp"]],
            [list(w["head"])] if lm_head else [],
        ]
        # one DRAM tensor per copy, allocated back to back so the copies sit at a fixed per-bank stride
        self._keep = []
        self.w_entries = []  # (first copy's tensor or None, stride, copies)
        for (Kt, dt, _), (sb, _, _, _), cc, mats in zip(entries, geo, cols_of, per_entry_mats):
            if not mats:
                self.w_entries.append((None, 0, 1))
                continue
            layout = BankLayout(Kt, cc, sb)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [Kt * TILE, layout.width * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )
            copies = []
            for m in mats:
                t = ttnn.from_torch(
                    torch.stack([layout(mm) for mm in m]).unsqueeze(1),
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=mc,
                    mesh_mapper=per_chip,
                )
                copies.append(settled(t))
            addrs = [t.buffer_address() for t in copies]
            stride = (addrs[1] - addrs[0]) & 0xFFFFFFFF if len(addrs) > 1 else 0
            assert all(((b - a) & 0xFFFFFFFF) == stride for a, b in zip(addrs, addrs[1:])), (
                "weight copies are not at a fixed DRAM stride",
                addrs[:4],
            )
            self._keep += copies
            self.w_entries.append((copies[0], stride, len(copies)))

        lcm = math.lcm(*TILE_BYTES.values())
        ring_bytes = (L1_WEIGHT_BUDGET // lcm) * lcm

        # ---- streamer tensors
        self.slots_t = l1(torch.zeros(S, n * HIDDEN), grid, [1, n * HIDDEN])
        act_t = l1(torch.zeros(S, d.ic), grid, [1, d.ic])
        self.o_in_t = l1(torch.zeros(S, d.gv), grid, [1, d.gv])
        self.logits_t = l1(torch.zeros(S, nv_max * TILE), grid, [1, nv_max * TILE]) if lm_head else None
        blk = TILE * TILE
        # conv taps per (streamer, GDN copy): [tap][32 x 1x32 tiles], only the core's q|k|v columns
        taps = torch.zeros(n, S * gdn_copies, CONV_K * blk)
        hist = torch.zeros(n, S * gdn_copies, (CONV_K - 1) * blk)
        tok_pos = st.pos
        for chip in range(n):
            for i in range(S):
                bank, j = divmod(i, PER_BANK)
                cnt, first = nqg_split[j]
                for c, wg in enumerate(w["gdn"]):
                    row = i * gdn_copies + c
                    tp_ = wg["chips"][chip]["taps"]
                    for t in range(cnt):
                        g = bank * qg_bank + first + t
                        if g >= d.conv_tiles:
                            continue
                        cols = slice(g * TILE, (g + 1) * TILE)
                        for tap in range(CONV_K):
                            o = tap * blk + t * TILE
                            taps[chip, row, o : o + TILE] = tp_[cols, tap]
                        uses = (self.n_gdn - c + gdn_copies - 1) // gdn_copies
                        for age in range(CONV_K - 1):  # age 0 = oldest; see conv_step in streamer_common.hpp
                            o = ((tok_pos * uses + age) % 3) * blk + t * TILE
                            hist[chip, row, o : o + TILE] = st.hist[c][chip][age, cols]
        taps_t = dram_t(taps.reshape(n * S * gdn_copies, -1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.hist_t = dram_t(hist.reshape(n * S * gdn_copies, -1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        # host copies of the decode state's initial contents (reset() restores them)
        self._initial = [
            (
                ttnn.from_torch(hist.reshape(n * S * gdn_copies, -1), dtype=ttnn.bfloat16, mesh_mapper=per_chip),
                self.hist_t,
            )
        ]

        # ---- mixer rows (GDN heads, attention leader and tail core)
        row_w = max(d.g_cols, d.a_cols)
        rows_t = l1(torch.zeros(len(heads) + 2, row_w), mixer_grid, [1, row_w])

        # ---- GDN heads: state [copy][head][16] fp32 tiles (interleaved: a state's tiles spread over the
        # banks), norm weight [copy][4] row-0 tiles
        state = torch.zeros(n, gdn_copies * d.nv * 16 * TILE, TILE)
        norm = torch.zeros(n, gdn_copies * 4 * TILE, TILE)
        for chip in range(n):
            for c in range(gdn_copies):
                for hh in range(d.nv):
                    Sm = st.gdn[c][chip][hh] if w["gdn"] else torch.zeros(DK, DV)
                    for kt in range(4):
                        for vt in range(4):
                            i = ((c * d.nv + hh) * 16 + kt * 4 + vt) * TILE
                            state[chip, i : i + TILE] = Sm[kt * TILE : (kt + 1) * TILE, vt * TILE : (vt + 1) * TILE]
                for vt in range(4):
                    if w["gdn"]:
                        norm[chip, (c * 4 + vt) * TILE] = w["gdn"][c]["norm_w"][vt * TILE : (vt + 1) * TILE]
        self.state_t = dram_t(state.reshape(n * gdn_copies * d.nv * 16 * TILE, TILE), ttnn.float32)
        self._initial.append(
            (
                ttnn.from_torch(
                    state.reshape(n * gdn_copies * d.nv * 16 * TILE, TILE),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=per_chip,
                ),
                self.state_t,
            )
        )
        norm_t = dram_t(norm.reshape(-1, TILE), ttnn.bfloat16)

        # ---- attention: q / M buffers, norm weights, KV caches (position tile t -> bank t % banks)
        Dt, heads_a = d.Dt, d.h
        q_t = l1(torch.zeros(WORKERS * TILE, Dt * TILE), work_grid, [TILE, Dt * TILE], tile=full)
        M_t = l1(torch.zeros((WORKERS + 1) * TILE, TILE), attn_grid, [TILE, TILE], tile=full)
        m_slots_t = l1(torch.zeros(1, WORKERS * 128), lead_grid, [1, WORKERS * 128], layout=ttnn.ROW_MAJOR_LAYOUT)
        one_t = l1(torch.ones(WORKERS * TILE, TILE), work_grid, [TILE, TILE], tile=full)
        mean_t = l1(torch.full((2 * TILE, TILE), 1.0 / HD), prep_grid, [TILE, TILE], tile=full)
        rows32 = lambda v: v[None].expand(TILE, -1)
        qw = torch.cat([rows32(wa["wq"] / math.sqrt(HD)) for wa in w["attn"]] or [torch.zeros(TILE, HD)], dim=1)
        kw = torch.cat([rows32(wa["wk"]) for wa in w["attn"]] or [torch.zeros(TILE, HD)], dim=1)
        # as [copies * Dt] tiles in order: a (TILE, copies*HD) row-major tile order is copy-major
        qw_t = dram_t(qw.reshape(TILE, -1), ttnn.bfloat16, mapper=rep)
        kw_t = dram_t(kw.reshape(TILE, -1), ttnn.bfloat16, mapper=rep)
        max_tp = st.max_pos // TILE
        rpb = (max_tp + banks - 1) // banks
        self.rpb = rpb
        kv_stride = rpb * KV_ROW

        def cache(which):
            t = torch.zeros(n, banks, attn_copies, rpb * TILE, HD)
            for c in range(len(w["attn"])):
                for chip in range(n):
                    src = (st.K if which == "K" else st.V)[c][chip]
                    for pt in range(src.shape[0] // TILE):
                        t[chip, pt % banks, c, (pt // banks) * TILE : (pt // banks + 1) * TILE] = src[
                            pt * TILE : (pt + 1) * TILE
                        ]
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [attn_copies * rpb * TILE, HD], ttnn.ShardOrientation.ROW_MAJOR),
            )
            flat = t.reshape(n * banks * attn_copies * rpb * TILE, HD)
            dev = ttnn.from_torch(
                flat, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc, mesh_mapper=per_chip
            )
            host = ttnn.from_torch(flat, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=per_chip)
            self._initial.append((host, dev))
            return dev

        self.K_t, self.V_t = settled(cache("K")), settled(cache("V"))
        chunk_max = max([chunk_of(wi, tp, banks)[3] for tp in range(max_tp + 1) for wi in range(WORKERS - 1)] + [1])
        nodes = list(range(WORKERS))
        groups_ = [nodes[i : i + FANIN] for i in range(0, WORKERS, FANIN)]
        slots_per_node = max(FANIN - 1, len(groups_))

        # ---- token state: position, x0, rope tables of the position
        self.rope_off = TOK_X0 + d.Ht * 64
        self.tok_words = (self.rope_off + 3 * 2048) // 4
        self.tok_t = dram_t(torch.zeros(1, self.tok_words, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, rep)

        # ---- hub
        self.hub_slots = l1(torch.zeros(2 * n, HIDDEN), hub_grid, [2 * n, HIDDEN], layout=ttnn.ROW_MAJOR_LAYOUT)
        self.ccl_sems = [ttnn.create_global_semaphore(mesh, hub_grid, 0) for _ in range(2)]  # per slot parity

        # ---- optional timelines
        def ts_buf(grid_, count, words):
            return ttnn.from_torch(
                torch.zeros(count, words, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid_, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )

        self.ts_t = ts_buf(grid, S, layers * 16) if timeline else None
        ts_addr = self.ts_t.buffer_address() if timeline else 0

        # ---- CBs
        full_page = TILE * TILE * 2
        f32 = (ttnn.float32, 4096)
        kv_page = TILE_BYTES[ttnn.bfloat8_b]

        def cb(idx, grid_, pages, dt=ttnn.bfloat16, page=full_page, tiny_tile=False):
            fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)
            if tiny_tile:
                fd = ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=dt, page_size=page, tile=ttnn.TileDescriptor(1, TILE)
                )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid_, format_descriptors=[fd])

        def tiny_cb(idx, pages, grid_=grid):
            return cb(idx, grid_, pages, page=64, tiny_tile=True)

        def aliased(tiny_idx, full_idx, tiles):
            return ttnn.CBDescriptor(
                total_size=tiles * 64,
                core_ranges=grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=tiny_idx,
                        data_format=ttnn.bfloat16,
                        page_size=64,
                        tile=ttnn.TileDescriptor(1, TILE),
                    ),
                    ttnn.CBFormatDescriptor(buffer_index=full_idx, data_format=ttnn.bfloat16, page_size=full_page),
                ],
            )

        T = ttnn.cb_descriptor_from_sharded_tensor
        cbs = [
            T(0, self.slots_t),
            ttnn.CBDescriptor(
                total_size=ring_bytes,
                core_ranges=grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=1 + e, data_format=dt, page_size=TILE_BYTES[dt])
                    for e, (_, dt, _) in enumerate(entries)
                ],
            ),
            cb(7, grid, 1, ttnn.uint32, 16),
            T(8, act_t),
            tiny_cb(9, nd_max),
            tiny_cb(10, nd_max),
            aliased(11, 12, d.Ht),
            aliased(13, 14, d.Ht),
            aliased(15, 27, TILE),
            aliased(26, 28, TILE),
            aliased(16, 29, TILE),
            aliased(17, 22, TILE),
            aliased(18, 23, TILE),
            aliased(19, 24, CONV_K * TILE),
            aliased(20, 25, (CONV_K - 1) * TILE),
            T(21, self.o_in_t),
            cb(30, grid, 1),
        ]
        if lm_head:
            cbs.append(T(31, self.logits_t))
        # GDN heads (same CB set as the per-layer benchmark)
        bf, f4 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
        head_cb_spec = {
            0: (bf, 4),
            1: (bf, 4),
            2: (bf, 4),
            3: (bf, 4),
            4: (bf, 4),
            5: (f4, 16),
            6: (f4, 1),
            7: (f4, 1),
            8: (f4, 1),
            9: (f4, 1),
            10: (f4, 4),
            11: (f4, 4),
            12: (f4, 1),
            13: (f4, 4),
            14: (f4, 4),
            15: (f4, 1),
            16: (f4, 1),
            17: (f4, 16),
            18: (f4, 4),
            19: (f4, 1),
            20: (f4, 4),
            21: (f4, 4),
            23: (f4, 16),
            24: (f4, 16),
            25: (f4, 4),
            26: (f4, 4),
            27: (f4, 4),
            28: (bf, 4),
            30: (f4, 1),
            31: (f4, 1),
        }
        for idx, ((dt, page), cnt) in head_cb_spec.items():
            cbs.append(cb(idx, head_grid, cnt, dt, page))
        cbs.append(cb(22, head_grid, 1, page=4 * 64))
        # attention workers (chunk workers and the tail core)
        cbs += [
            T(0, q_t),
            T(3, M_t),
            T(4, one_t),
            cb(1, work_grid, 2 * chunk_max * Dt, ttnn.bfloat8_b, kv_page),
            cb(2, work_grid, 2 * chunk_max * Dt, ttnn.bfloat8_b, kv_page),
            cb(5, work_grid, chunk_max, *f32),
            cb(6, work_grid, chunk_max),
            cb(7, work_grid, 1),
            cb(8, work_grid, Dt + 1),
            cb(10, work_grid, 1, ttnn.uint32, 16),
            cb(23, attn_grid, slots_per_node * (Dt + 1)),
            cb(25, attn_grid, Dt + 1),
            # leader and tail core
            cb(13, prep_grid, Dt),
            cb(14, prep_grid, 2),
            cb(17, prep_grid, 3),
            T(29, mean_t),
            # leader
            cb(2, lead_grid, 1, page=heads_a * Dt * 64),
            cb(9, lead_grid, Dt),
            cb(10, lead_grid, Dt),
            cb(15, lead_grid, Dt),
            cb(24, lead_grid, Dt),
            cb(26, lead_grid, Dt),
            cb(27, lead_grid, Dt),
            cb(30, lead_grid, 1),
            # tail core
            cb(11, tail_grid, Dt),
            cb(12, tail_grid, 2),
            cb(16, tail_grid, Dt),
            cb(18, tail_grid, Dt),
            cb(19, tail_grid, Dt),
            cb(20, tail_grid, Dt),
            cb(21, tail_grid, Dt),
            cb(22, tail_grid, 1),
            cb(28, tail_grid, 2 * Dt, ttnn.bfloat8_b, kv_page),
            cb(31, tail_grid, 2 * Dt, ttnn.bfloat8_b, kv_page),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_grid, initial_value=0) for i in range(12)]

        def kernel(src, core_ranges, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=KDIR + src,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )

        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
        NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default

        def compute_cfg(fidelity, fp32, full_sync, unpack=None):
            c = ttnn.ComputeConfigDescriptor()
            c.math_fidelity = fidelity
            c.fp32_dest_acc_en = fp32
            c.dst_full_sync_en = full_sync
            if unpack is not None:
                c.unpack_to_dest_mode = unpack
            return c

        streamer_compute = compute_cfg(ttnn.MathFidelity.HiFi4, True, True)
        head_unpack = type(ttnn.ComputeConfigDescriptor().unpack_to_dest_mode)([ttnn.UnpackToDestMode.Default] * 64)
        for i in HEAD_FP32_CBS:
            head_unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        head_compute = compute_cfg(ttnn.MathFidelity.HiFi4, True, False, head_unpack)
        attn_compute = compute_cfg(ttnn.MathFidelity.HiFi4, True, True)
        work_compute = compute_cfg(ttnn.MathFidelity.HiFi2, True, False)

        entry_rt = []
        attn_ct = [self.n_attn, attn_copies, WORKERS, S, chunk_max, heads_a, Dt, ROT // TILE]
        attn_ct += [
            SEM_ROWS,
            SEM_Q,
            SEM_m,
            SEM_M,
            SEM_PART,
            SEM_HEADS,
            f32_bits(EPS),
            SEM_ADDR,
            slots_per_node,
            banks,
            SEM_LOCAL,
        ]
        group = d.nv // d.nk
        kv_addrs = [self.K_t.buffer_address(), self.V_t.buffer_address()]
        tok_addr = self.tok_t.buffer_address()
        mesh_pd = ttnn.MeshProgramDescriptor()
        for chip in range(n):
            coord = ttnn.MeshCoordinate(0, chip)
            ct = [layers, interval, n, S, d.Ht, d.It, chip, ring_bytes, f32_bits(EPS), f32_bits(1 / math.sqrt(HIDDEN))]
            ct += [SEM_SLOTS, SEM_ACT, SEM_GATHER, SEM_HEADS, SEM_ROWS, ng_max, nd_max, nq_max, nv_max]
            ct += [d.conv_tiles, d.Ot, d.nv, int(lm_head)]
            for (Kt, _, _), (sb, pages, page, block) in zip(entries, geo):
                ct += [Kt, sb, pages, page, block]
            ct += [SEM_LOCAL]
            rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            peers = [v for p in sphys for v in (p.x, p.y)]
            head_xy = [v for p in head_p for v in (p.x, p.y)]
            for i, c in enumerate(cores):
                bank, j = divmod(i, PER_BANK)
                nqg, qgf = nqg_split[j]
                nqa, qaf = nqa_split[j]
                nd, dfirst = nd_split[j]
                ng, gfirst = ng_split[j]
                nvv, _ = nv_split[j]
                counts = [nqg, nqa, nd, 2 * ng, nd, nvv]
                ent = []
                for cnt, (t, stride, copies) in zip(counts, self.w_entries):
                    ent += [cnt if t is not None else 0, t.buffer_address() if t is not None else 0, stride, copies]
                rr[c.x][c.y] = (
                    [bank, i & 0x3, j, PER_BANK]
                    + ent
                    + [
                        taps_t.buffer_address(),
                        self.hist_t.buffer_address(),
                        i * gdn_copies,
                        gdn_copies,
                        tok_addr,
                        ts_addr,
                    ]
                )
                n_conv = max(0, min(nqg, d.conv_tiles - (bank * qg_bank + qgf)))
                wr[c.x][c.y] = (
                    [ng, bank * it_bank + gfirst, nd, bank * ht_bank + dfirst, hub_p.x, hub_p.y]
                    + [self.hub_slots.buffer_address(), act_t.buffer_address()]
                    + [nqg, bank * qg_bank + qgf, nqa, bank * qa_bank + qaf, rows_t.buffer_address(), n_conv]
                    + [self.hist_t.buffer_address(), i * gdn_copies, gdn_copies, tok_addr]
                    + head_xy
                    + [lead_p.x, lead_p.y, tail_p.x, tail_p.y]
                    + peers
                    + [ts_addr]
                )
                cr[c.x][c.y] = [ng, nd, bank * ht_bank + dfirst, nqg, nqa, n_conv, nvv]

            hub_rt = ttnn.RuntimeArgs()
            hub_rt[hub.x][hub.y] = [
                self.hub_slots.buffer_address(),
                *[ttnn.get_global_semaphore_address(s) for s in self.ccl_sems],
                self.slots_t.buffer_address(),
            ] + rect_cover(mesh, cores)
            hub_ct = [n, chip, HIDDEN * 2, packet_bytes, 2 * layers, S, SEM_GATHER, SEM_SLOTS]
            hub_ct += [SEM_FLAG, int(lm_head)]

            # GDN heads
            hr, hw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for hh, c in enumerate(heads):
                kh = hh // group
                a_col, b_col = d.a0 + hh, d.b0 + hh
                gates = []
                for wg in w["gdn"] or [None]:
                    cw = wg["chips"][chip] if wg else None
                    gates += [f32_bits(float(cw["dt"][hh])) if cw else 0, f32_bits(float(cw["neg_a"][hh])) if cw else 0]
                hr[c.x][c.y] = (
                    [
                        rows_t.buffer_address(),
                        self.state_t.buffer_address(),
                        norm_t.buffer_address(),
                        kh * 4,
                        d.gq // TILE + kh * 4,
                        2 * d.gq // TILE + hh * 4,
                        d.z0 // TILE + hh * 4,
                        a_col // TILE,
                        a_col % TILE,
                        b_col // TILE,
                        b_col % TILE,
                        hh,
                    ]
                    + gates
                    + [0]
                )
                hw[c.x][c.y] = [self.state_t.buffer_address(), self.o_in_t.buffer_address(), hh] + peers + [0]
            head_ct = [4, 4, self.n_gdn, gdn_copies, S]
            head_compute_ct = [d.nk, d.nv, 4, 4, group, f32_bits(DK**-0.5), f32_bits(1e-6), f32_bits(1e-6)]
            head_compute_ct += [f32_bits(1.0 / DV), self.n_gdn]

            # attention cores
            parent = {}
            for g_i, grp in enumerate(groups_):
                parent[grp[0]] = (lead_p, g_i)
                for k, w_i in enumerate(grp[1:]):
                    parent[w_i] = (wphys[grp[0]], k)
            kids = {grp[0]: grp[1:] for grp in groups_}
            ar, aw, ac = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for w_i, c in enumerate(workers):
                is_tail = w_i == WORKERS - 1
                role = 2 if is_tail else 1
                ch = kids.get(w_i, [])
                pp, slot = parent[w_i]
                ar[c.x][c.y] = (
                    [role, w_i]
                    + kv_addrs
                    + [kv_stride, len(ch), 0, tok_addr, self.rope_off]
                    + ([rows_t.buffer_address(), kw_t.buffer_address()] if is_tail else [])
                )
                aw[c.x][c.y] = (
                    [role, w_i, lead_p.x, lead_p.y, m_slots_t.buffer_address(), pp.x, pp.y, slot, len(ch)]
                    + [v for k in ch for v in (wphys[k].x, wphys[k].y)]
                    + [0]
                    + (kv_addrs + [kv_stride, tok_addr] if is_tail else [])
                )
                ac[c.x][c.y] = [role, len(ch)]
            lr, lw, lc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            lr[leader.x][leader.y] = [rows_t.buffer_address(), len(groups_), 0, tok_addr, self.rope_off]
            lr[leader.x][leader.y].append(qw_t.buffer_address())
            w0p, w1p = wphys[0], wphys[-1]
            lw[leader.x][leader.y] = (
                [q_t.buffer_address(), M_t.buffer_address(), m_slots_t.buffer_address(), WORKERS]
                + [w0p.x, w0p.y, w1p.x, w1p.y, self.o_in_t.buffer_address()]
                + peers
                + [len(groups_)]
                + [v for grp in groups_ for v in (wphys[grp[0]].x, wphys[grp[0]].y)]
                + [0]
            )
            lc[leader.x][leader.y] = [len(groups_)]

            kernels = [
                kernel("streamer_reader.cpp", grid, ct, rr, dm(NCRISC, NOC0)),
                kernel("streamer_writer.cpp", grid, ct, wr, dm(BRISC, NOC1)),
                kernel("streamer_compute.cpp", grid, ct, cr, streamer_compute),
                kernel("mlp_hub.cpp", hub_grid, hub_ct, hub_rt, dm(BRISC, NOC0)),
            ]
            if self.n_gdn:
                kernels += [
                    kernel(
                        "gdn_head_reader.cpp", head_grid, head_ct + [SEM_ROWS, d.nv, SEM_LOCAL], hr, dm(NCRISC, NOC1)
                    ),
                    kernel(
                        "gdn_head_writer.cpp", head_grid, head_ct + [SEM_HEADS, d.nv, SEM_LOCAL], hw, dm(BRISC, NOC0)
                    ),
                    kernel("gdn_head_compute.cpp", head_grid, head_compute_ct, ttnn.RuntimeArgs(), head_compute),
                ]
            if self.n_attn:
                kernels += [
                    kernel("attn_worker_reader.cpp", work_grid, attn_ct, ar, dm(NCRISC, NOC1)),
                    kernel("attn_worker_writer.cpp", work_grid, attn_ct, aw, dm(BRISC, NOC0)),
                    kernel("attn_worker_compute.cpp", chunk_grid, attn_ct, ac, work_compute),
                    kernel("attn_worker_compute.cpp", tail_grid, attn_ct, ac, attn_compute),
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
        self.mesh_pd = mesh_pd
        self.io = [self.slots_t, act_t, self.o_in_t, rows_t, q_t, M_t, m_slots_t, one_t, mean_t, self.hub_slots]
        self.io += [taps_t, self.hist_t, self.state_t, norm_t, qw_t, kw_t, self.K_t, self.V_t, self.tok_t]
        self.io += self._keep
        if lm_head:
            self.io.append(self.logits_t)
        if timeline:
            self.io.append(self.ts_t)
        self.nv_split, self.v_bank = nv_split, v_bank

    def token_state(self, x0, pos):
        """host tensor of the token state for x0 at pos"""
        words = np.zeros(self.tok_words * 2, dtype=np.uint16)
        words[0], words[1] = pos & 0xFFFF, pos >> 16
        words[TOK_X0 // 2 : TOK_X0 // 2 + HIDDEN] = bf16_bits(x0)
        cos, sin = rope_tables(pos)
        r0 = self.rope_off // 2
        for i, v in enumerate((cos, sin, -sin)):
            words[r0 + i * 1024 : r0 + (i + 1) * 1024] = tile_bytes_rows_replicated(v)
        t = torch.from_numpy(words.view(np.int32).copy()).reshape(1, -1)
        return ttnn.from_torch(
            t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )

    def reset(self):
        """restore the decode state (GDN states, conv histories, KV caches) to its initial contents"""
        for host, dev in self._initial:
            ttnn.copy_host_to_device_tensor(host, dev)
        ttnn.synchronize_device(self.mesh)

    def step(self, tok_host):
        ttnn.copy_host_to_device_tensor(tok_host, self.tok_t)
        ttnn.generic_op(self.io, self.mesh_pd)

    def x(self):
        got = ttnn.to_torch(self.hub_slots, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        n = self.d.n
        p = (2 * self.layers - 1) & 1
        return got[: 2 * n][p * n : (p + 1) * n].sum(0)

    def mixer_output(self):
        """the last layer's mixer output as received by streamer 0 of each chip"""
        o = ttnn.to_torch(self.o_in_t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        return o.reshape(self.d.n, self.S, -1)[:, 0]

    def logits(self):
        """per-chip logits [n, vocab_chip] in vocab order"""
        raw = ttnn.to_torch(self.logits_t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        raw = raw.reshape(self.d.n, self.S, -1)
        out = torch.zeros(self.d.n, self.d.vocab)
        for i in range(self.S):
            bank, j = divmod(i, PER_BANK)
            cnt, first = self.nv_split[j]
            c0 = (bank * self.v_bank + first) * TILE
            out[:, c0 : c0 + cnt * TILE] = raw[:, i, : cnt * TILE]
        return out
