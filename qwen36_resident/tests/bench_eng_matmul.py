# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefill engine's distributed matmul (prefill/kernels/eng_*.cpp) on one chip: [C, K] activation resident
in L1 as (row group, K block) on an R x Q core grid, weights streamed per grid column from DRAM. Checked
against torch and timed against ttnn.linear on the same shape."""
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident import PKG_DIR

KDIR = f"{PKG_DIR}/prefill/kernels/"
TILE = 32


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def build(device, C, K, N, wdtype, R=8, Q=13, kb=4, sh=4, sw=2):
    Mt, Kt = C // TILE, K // TILE
    mt, blocks = Mt // R, Kt // kb
    nt = math.ceil(N // TILE / Q / sw) * sw
    owned = [blocks // Q + (c < blocks % Q) for c in range(Q)]
    first = [sum(owned[:c]) for c in range(Q)]
    owner = [c for c in range(Q) for _ in range(owned[c])]
    act_w = max(owned) * kb
    cores = [ttnn.CoreCoord(c, r) for r in range(R) for c in range(Q)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(Q - 1, R - 1))])
    phys = {(c.x, c.y): device.worker_core_from_logical_core(c) for c in cores}

    g = torch.Generator().manual_seed(0)
    x = torch.randn(C, K, generator=g).bfloat16().float()
    w = (torch.randn(K, N, generator=g) / math.sqrt(K)).bfloat16().float()
    if os.environ.get("ENG_REAL"):
        # the checkpoint's layer-0 MLP gate (norm folded), rows of real embeddings normalized
        from qwen36_resident.prefill import pp_weights as PW
        from qwen36_resident import weights as QW

        ck = QW.Checkpoint()
        w = PW.mlp(ck, 0)["G"].float()[:K, :N]
        e = QW.embedding(ck)[torch.randint(0, 200000, (C,), generator=g)].float()
        x = (e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16().float()[:, :K]
    # activation shards: core (r, c) holds rows of group r, K tiles of its blocks
    shards = torch.zeros(R, Q, mt * TILE, act_w * TILE)
    for r in range(R):
        for c in range(Q):
            k0, k1 = first[c] * kb * TILE, (first[c] + owned[c]) * kb * TILE
            shards[r, c, :, : k1 - k0] = x[r * mt * TILE : (r + 1) * mt * TILE, k0:k1]
    act_t = ttnn.from_torch(
        shards.reshape(R * Q * mt * TILE, act_w * TILE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [mt * TILE, act_w * TILE], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    # weights: column c's [K, nt] block, K-major
    wp = torch.zeros(K, Q * nt * TILE)
    wp[:, :N] = w
    wcols = wp.reshape(K, Q, nt * TILE).permute(1, 0, 2).reshape(Q * K, nt * TILE)
    w_t = ttnn.from_torch(wcols, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device)
    w_ref = ttnn.to_torch(w_t).float().reshape(Q, K, nt * TILE).permute(1, 0, 2).reshape(K, Q * nt * TILE)[:, :N]
    out_t = ttnn.from_torch(torch.zeros(C, Q * nt * TILE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tile_bytes = lambda dt: {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[dt]
    cb = lambda idx, dt, n: ttnn.CBDescriptor(
        total_size=n * tile_bytes(dt),
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=tile_bytes(dt))],
    )
    cbs = [
        cb(0, ttnn.bfloat16, 2 * mt * kb),
        cb(1, wdtype, 2 * kb * nt),
        cb(16, ttnn.bfloat16, mt * nt),
        ttnn.cb_descriptor_from_sharded_tensor(10, act_t),
    ]
    r0, r1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core in cores:
        c, r = core.x, core.y
        a, b = phys[(0, r)], phys[(Q - 1, r)]
        peers = [v for q in range(Q) for v in (phys[(q, r)].x, phys[(q, r)].y)]
        r0[c][r] = [c, first[c], owned[c], a.x, a.y, b.x, b.y, out_t.buffer_address(), Q * nt, r, nt] + peers + owner
        top, lo, hi = phys[(c, 0)], phys[(c, 1 if R > 1 else 0)], phys[(c, R - 1)]
        # NOC1 multicast: start / end mirrored
        r1[c][r] = [r, w_t.buffer_address(), c, hi.x, hi.y, lo.x, lo.y, top.x, top.y]
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
    kernels = [
        kd(
            "eng_in0.cpp",
            [mt, kb, blocks, Q, act_w],
            r0,
            ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        ),
        kd(
            "eng_in1.cpp",
            [kb, nt, blocks, R],
            r1,
            ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
            ),
        ),
        kd("eng_compute.cpp", [mt, kb, nt, blocks, sh, sw], ttnn.RuntimeArgs(), compute),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=0) for i in range(4)]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = [act_t, w_t, out_t]
    run = lambda: ttnn.generic_op(io, program)
    ref = x @ w_ref
    read = lambda: ttnn.to_torch(out_t).float()[:, :N]
    return run, read, ref, (x, w)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
def test_eng_matmul(device):
    C, K, N = (int(v) for v in os.environ.get("ENG_SHAPE", "1024,5120,17408").split(","))
    wdtype = dict(bf4=ttnn.bfloat4_b, bf8=ttnn.bfloat8_b)[os.environ.get("ENG_W", "bf4")]
    kb = int(os.environ.get("ENG_KB", "4"))
    run, read, ref, (x, w) = build(device, C, K, N, wdtype, kb=kb)
    run()
    ttnn.synchronize_device(device)
    got = read()
    p = pcc(got, ref)

    def timed(fn, reps=10):
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

    dt = timed(run)
    xt = ttnn.from_torch(x[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    wt = ttnn.from_torch(w, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    outs = []
    dt_ttnn = timed(lambda: outs.append(ttnn.linear(xt, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16)))
    flops = 2 * C * K * N
    logger.info(
        f"engine {dt * 1e6:.0f} us ({flops / dt / 1e12:.0f} TFLOPS), pcc {p:.5f}; "
        f"ttnn.linear {dt_ttnn * 1e6:.0f} us ({flops / dt_ttnn / 1e12:.0f} TFLOPS)"
    )
    assert p > 0.99
