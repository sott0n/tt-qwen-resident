# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefill engine's MLP stack (prefill/kernels/mlp_*.cpp) on one chip: x += mlp(rmsnorm(x)) for a few
layers in one program, the residual stream resident in L1 on an R x Q core grid. Checked against torch and
timed against the ttnn op sequence of pp_prefill on the same data."""
import math
import os
import struct
import time

import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident.prefill.pp_prefill import MM_BEST
from qwen36_resident.tests.bench_resident_mlp import EPS, HIDDEN, INTER
from qwen36_resident import PKG_DIR

KDIR = f"{PKG_DIR}/prefill/kernels/"
TILE = 32
TB = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def weights(L, real):
    g = torch.Generator().manual_seed(1)
    if not real:
        mk = lambda k, n: torch.randn(k, n, generator=g) / math.sqrt(k)
        return [dict(G=mk(HIDDEN, INTER), U=mk(HIDDEN, INTER), D=mk(INTER, HIDDEN)) for _ in range(L)]
    from qwen36_resident.prefill import pp_weights as PW
    from qwen36_resident.tests import qwen36_weights as QW

    ck = QW.Checkpoint()
    return [{k: v.float() for k, v in PW.mlp(ck, i).items()} for i in range(L)]


def inputs(C, real):
    g = torch.Generator().manual_seed(0)
    if not real:
        return torch.randn(C, HIDDEN, generator=g).bfloat16().float()
    from qwen36_resident.tests import qwen36_weights as QW

    e = QW.embedding(QW.Checkpoint())
    return e[torch.randint(0, e.shape[0], (C,), generator=g)].bfloat16().float()


class Engine:
    def __init__(self, device, C, ws, R=8, Q=13, gu_slot=5, d_slot=6):
        self.R, self.Q, self.C, self.L = R, Q, C, len(ws)
        Mt, Ht, It = C // TILE, HIDDEN // TILE, INTER // TILE
        assert Mt % R == 0
        mt, HC, IC = Mt // R, math.ceil(Ht / Q), math.ceil(It / Q)
        IC += IC % 2
        W = HC + HC % 2
        self.mt, self.HC, self.IC, self.W = mt, HC, IC, W
        gu_blocks, d_blocks = math.ceil(HC / gu_slot), math.ceil(IC / d_slot)
        S0 = max(gu_slot, d_slot)
        Hp, Ip = Q * HC * TILE, Q * IC * TILE

        # weights per grid column, K-major: G, U [Q, Hp, IC], D [Q, Ip, W] (hidden padded at the end, the
        # column's HC output tiles then a zero tile)
        def cols(w, Kp, n_real, n_col, n_slot):
            K = w.shape[0]
            wp = torch.zeros(Kp, Q, n_slot * TILE)
            full = torch.zeros(Kp, Q * n_col * TILE)
            full[:K, :n_real] = w
            wp[:, :, : n_col * TILE] = full.reshape(Kp, Q, n_col * TILE)
            return wp.permute(1, 0, 2).reshape(Q * Kp, n_slot * TILE)

        banks = device.dram_grid_size().x
        assert banks == R, "one DRAM bank per grid row"
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})

        def bank_major(wc, n_col, bpo, slot, per, dt):
            """column-major weights [Q * Kp, n_col] -> block b of column c (its kb x n_col tiles, K-major) in
            DRAM bank (b * Q + c) % banks at ((b * Q + c) // banks) * slot * N tiles"""
            N = n_col // TILE
            Kt = wc.shape[0] // Q // TILE
            wt = wc.reshape(Q, Kt, TILE, N, TILE).permute(0, 1, 3, 2, 4)
            n, blk = Q * bpo, slot * N
            per_bank = math.ceil(n * Q / banks) * blk
            out = torch.zeros(banks, per_bank, TILE, TILE)
            for b in range(n):
                o, i = divmod(b, bpo)
                kb = slot if i + 1 < bpo else per - slot * (bpo - 1)
                k0 = o * per + i * slot
                for c in range(Q):
                    g = b * Q + c
                    at = (g // banks) * blk
                    out[g % banks, at : at + kb * N] = wt[c, k0 : k0 + kb].reshape(kb * N, TILE, TILE)
            host = out.permute(2, 0, 1, 3).reshape(TILE, banks * per_bank * TILE)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [TILE, per_bank * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )
            return ttnn.from_torch(host, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)

        # the reference uses the same tiles quantized on the host (bf4 / bf8 blocks are tile-local)
        quant = lambda t, dt: ttnn.to_torch(ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT)).float()
        back = lambda wc, Kp, n_col, n_slot, K, N: (
            wc.reshape(Q, Kp, n_slot)[:, :, :n_col].permute(1, 0, 2).reshape(Kp, Q * n_col)[:K, :N]
        )
        self.w_dev, self.w_ref = [], []
        for w in ws:
            g = quant(cols(w["G"], Hp, INTER, IC, IC), ttnn.bfloat4_b)
            u = quant(cols(w["U"], Hp, INTER, IC, IC), ttnn.bfloat4_b)
            d = quant(cols(w["D"], Ip, HIDDEN, HC, W), ttnn.bfloat8_b)
            self.w_dev.append(
                (
                    bank_major(g, IC * TILE, gu_blocks, gu_slot, HC, ttnn.bfloat4_b),
                    bank_major(u, IC * TILE, gu_blocks, gu_slot, HC, ttnn.bfloat4_b),
                    bank_major(d, W * TILE, d_blocks, d_slot, IC, ttnn.bfloat8_b),
                )
            )
            self.w_ref.append(
                dict(
                    G=back(g, Hp, IC * TILE, IC * TILE, HIDDEN, INTER),
                    U=back(u, Hp, IC * TILE, IC * TILE, HIDDEN, INTER),
                    D=back(d, Ip, HC * TILE, W * TILE, INTER, HIDDEN),
                )
            )

        # one L1 arena per core; every CB is placed in it, u's memory shared with the down weights and the
        # norm's squares, h's with d
        layout, off = {}, 0

        def place(name, size, at=None):
            nonlocal off
            if at is None:
                layout[name] = (off, size)
                off += size
            else:
                layout[name] = (layout[at][0], size)
                assert size <= layout[at][1], name

        bf = TB[ttnn.bfloat16]
        place("x", mt * W * bf)
        place("h", mt * W * bf)
        place("d", mt * W * bf, at="h")
        place("a", mt * IC * bf)
        place("u", mt * IC * bf)
        dw_bytes = d_slot * W * TB[ttnn.bfloat8_b]
        place("dw", (mt * IC * bf // dw_bytes) * dw_bytes, at="u")
        place("sq", HC * bf, at="u")
        place("in0", 2 * mt * S0 * bf)
        place("gu", 2 * gu_slot * IC * TB[ttnn.bfloat4_b])
        place("recv", Q * mt * bf)
        place("part", mt * bf)
        place("rs", mt * bf)
        place("scaler", bf)
        for t in ("h_rdy", "a_rdy", "u_free"):
            place(t, 32)
        A = math.ceil(off / bf)
        logger.info(f"arena {off / 1024:.0f} KB per core; d ring {layout['dw'][1] // dw_bytes} blocks")
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(Q - 1, R - 1))])
        self.arena = ttnn.from_torch(
            torch.zeros(R * Q * TILE, A * TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [TILE, A * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        self.A = A
        cbs = []
        for idx, name, dt, page in [
            (0, "in0", ttnn.bfloat16, bf),
            (1, "gu", ttnn.bfloat4_b, TB[ttnn.bfloat4_b]),
            (2, "dw", ttnn.bfloat8_b, TB[ttnn.bfloat8_b]),
            (10, "x", ttnn.bfloat16, bf),
            (11, "h", ttnn.bfloat16, bf),
            (12, "a", ttnn.bfloat16, bf),
            (13, "u", ttnn.bfloat16, bf),
            (14, "d", ttnn.bfloat16, bf),
            (16, "part", ttnn.bfloat16, bf),
            (17, "recv", ttnn.bfloat16, bf),
            (18, "rs", ttnn.bfloat16, bf),
            (19, "scaler", ttnn.bfloat16, bf),
            (20, "sq", ttnn.bfloat16, bf),
            (24, "h_rdy", ttnn.bfloat16, 32),
            (25, "a_rdy", ttnn.bfloat16, 32),
            (26, "u_free", ttnn.bfloat16, 32),
        ]:
            o, size = layout[name]
            cb = ttnn.cb_descriptor_from_sharded_tensor(idx, self.arena, address_offset=o, total_size=size)
            cb.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)]
            cbs.append(cb)

        cores = [ttnn.CoreCoord(c, r) for r in range(R) for c in range(Q)]
        phys = {(c.x, c.y): device.worker_core_from_logical_core(c) for c in cores}
        r0, r1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        waddr = [a.buffer_address() for t in self.w_dev for a in t]
        for core in cores:
            c, r = core.x, core.y
            a, b = phys[(0, r)], phys[(Q - 1, r)]
            peers = [v for q in range(Q) for v in (phys[(q, r)].x, phys[(q, r)].y)]
            r0[c][r] = [c, a.x, a.y, b.x, b.y] + peers
            lo, hi = phys[(c, 0)], phys[(c, R - 1)]
            col = [v for q in range(R) for v in (phys[(c, q)].x, phys[(c, q)].y)]
            r1[c][r] = [r, c, hi.x, hi.y, lo.x, lo.y] + col + waddr  # NOC1: rectangle mirrored
        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = ttnn.MathFidelity.LoFi
        compute.fp32_dest_acc_en = False
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        L = self.L
        kernels = [
            kd(
                "mlp_in0.cpp",
                [mt, W, IC, Q, S0, gu_blocks, gu_slot, HC, d_blocks, d_slot, L],
                r0,
                ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            ),
            kd(
                "mlp_in1.cpp",
                [R, Q, IC, W, HC, gu_blocks, gu_slot, d_blocks, d_slot, L, 2, layout["dw"][1] // dw_bytes],
                r1,
                ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                ),
            ),
            kd(
                "mlp_compute.cpp",
                [mt, HC, W, IC, Q, S0, gu_blocks, gu_slot, d_blocks, d_slot, L, f32_bits(EPS), f32_bits(1 / HIDDEN)],
                ttnn.RuntimeArgs(),
                compute,
            ),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=int(i == 7)) for i in range(8)]
        self.program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        self.io = [self.arena] + [a for t in self.w_dev for a in t]

    def _tiles(self, x):
        """[C, HIDDEN] -> arena rows: core (r, c) tile i * W + j = x tile (r * mt + i, c * HC + j)"""
        R, Q, mt, HC, W = self.R, self.Q, self.mt, self.HC, self.W
        xp = torch.zeros(self.C, Q * HC * TILE)
        xp[:, :HIDDEN] = x
        t = xp.reshape(R, mt, TILE, Q, HC, TILE).permute(0, 3, 1, 4, 2, 5)  # r c i j y x
        full = torch.zeros(R, Q, mt, W, TILE, TILE)
        full[:, :, :, :HC] = t
        return full.reshape(R, Q, mt * W, TILE, TILE)

    def load(self, x):
        arena = torch.zeros(self.R, self.Q, self.A, TILE, TILE)
        arena[:, :, : self.mt * self.W] = self._tiles(x)
        host = arena.permute(0, 1, 3, 2, 4).reshape(self.R * self.Q * TILE, self.A * TILE)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.arena)

    def read(self):
        R, Q, mt, HC, W = self.R, self.Q, self.mt, self.HC, self.W
        a = ttnn.to_torch(self.arena).float().reshape(R, Q, TILE, self.A, TILE).permute(0, 1, 3, 2, 4)
        t = a[:, :, : mt * W].reshape(R, Q, mt, W, TILE, TILE)[:, :, :, :HC]
        return t.permute(0, 2, 4, 1, 3, 5).reshape(self.C, Q * HC * TILE)[:, :HIDDEN]

    def run(self):
        ttnn.generic_op(self.io, self.program)


def ref_mlp(x, ws):
    for w in ws:
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
        a = torch.nn.functional.silu(h @ w["G"]) * (h @ w["U"])
        x = x + a @ w["D"]
    return x


def ttnn_mlp(device, C, ws):
    """pp_prefill's MLP op sequence on interleaved DRAM tensors"""
    mm = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    def pc(key, N, act=None):
        gx, gy, bw, sh, sw = MM_BEST[(key, C)]
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=bw,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=math.ceil(C // TILE / gy),
            per_core_N=math.ceil(N // TILE / gx),
            transpose_mcast=False,
            fused_activation=act,
            fuse_batch=False,
        )

    pg, pu, pd = pc("gu", INTER, ttnn.UnaryOpType.SILU), pc("gu", INTER), pc("down", HIDDEN)
    dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    wd = [(dev(w["G"], ttnn.bfloat4_b), dev(w["U"], ttnn.bfloat4_b), dev(w["D"], ttnn.bfloat8_b)) for w in ws]
    lin = lambda x, w, p: ttnn.linear(
        x, w, compute_kernel_config=mm, program_config=p, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )

    def run(x):
        for G, U, D in wd:
            h = ttnn.rms_norm(x, epsilon=EPS)
            a = ttnn.multiply(lin(h, G, pg), lin(h, U, pu))
            x = ttnn.add(x, lin(a, D, pd))
        return x

    return run


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
def test_eng_mlp(device):
    C = int(os.environ.get("ENG_C", "1024"))
    L = int(os.environ.get("ENG_LAYERS", "1"))
    real = bool(os.environ.get("ENG_REAL"))
    ws = weights(L, real)
    x = inputs(C, real)
    eng = Engine(device, C, ws)
    ref = ref_mlp(x, eng.w_ref)
    eng.load(x)
    eng.run()
    ttnn.synchronize_device(device)
    got = eng.read()
    p = pcc(got - x, ref - x)
    logger.info(f"engine pcc {pcc(got, ref):.5f}, of the MLP output {p:.5f}")

    def timed(fn, reps=5):
        fn()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(reps):
            fn()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        dt = (time.perf_counter() - t0) / reps
        ttnn.release_trace(device, tid)
        return dt

    dt = timed(eng.run)
    ttnn.deallocate(eng.arena)
    run = ttnn_mlp(device, C, ws)
    xt = ttnn.from_torch(x[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = run(xt)
    p_ttnn = pcc(ttnn.to_torch(out).float()[0, 0] - x, ref - x)
    dt_ttnn = timed(lambda: run(xt))
    flops = 2 * C * HIDDEN * INTER * 3 * L
    logger.info(
        f"{L} layers: engine {dt * 1e3:.3f} ms ({flops / dt / 1e12:.0f} TFLOPS); "
        f"ttnn {dt_ttnn * 1e3:.3f} ms ({flops / dt_ttnn / 1e12:.0f} TFLOPS), pcc {p_ttnn:.5f}"
    )
    assert p > 0.99
