# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The small-M streaming matmul (prefill/smm.py) against torch and against ttnn's 1D multicast matmul on the
prefill projection shapes, traced."""
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident.prefill.smm import SmallMatmul

SHAPES = dict(
    gdn=(5120, 16512, ttnn.bfloat8_b),
    attn=(5120, 14336, ttnn.bfloat8_b),
    out=(6144, 5120, ttnn.bfloat8_b),
    gu=(5120, 17408, ttnn.bfloat4_b),
    down=(17408, 5120, ttnn.bfloat8_b),
)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_smm(mesh_device):
    n = mesh_device.get_num_devices()
    M = int(os.environ.get("SMM_M", "128"))
    kb = int(os.environ.get("SMM_KB", "8"))
    names = os.environ.get("SMM_SHAPES", "gu,down,gdn,attn,out").split(",")
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    def timed(fn, reps=8):
        fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(reps):
            fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        dt = (time.perf_counter() - t0) / reps
        ttnn.release_trace(mesh_device, tid)
        return dt

    g = torch.Generator().manual_seed(0)
    for name in names:
        K, N, dt = SHAPES[name]
        x = torch.randn(n, 1, M, K, generator=g).bfloat16().float()
        w = torch.randn(n, K, N, generator=g) * 0.02
        mm = SmallMatmul(mesh_device, M, K, N, dt, kb=kb)
        dev = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=shard)
        x_t = dev(x, ttnn.bfloat16)
        w_b = mm.weights(w)
        out = dev(torch.zeros(n, 1, M, N), ttnn.bfloat16)
        mm(x_t, w_b, out)
        ttnn.synchronize_device(mesh_device)
        wq = ttnn.to_torch(ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT)).float()
        p = pcc(ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float(), x @ wq[:, None])
        t_smm = timed(lambda: mm(x_t, w_b, out))
        w_i = dev(w[:, None], dt)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(13, 10),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=max(d for d in range(1, 9) if math.ceil(N // 32 / 130) % d == 0),
            per_core_M=M // 32,
            per_core_N=math.ceil(N // 32 / 130),
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        t_1d = timed(
            lambda: ttnn.linear(
                x_t,
                w_i,
                program_config=pc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=cfg,
                dtype=ttnn.bfloat16,
            )
        )
        wbytes = K * N * {ttnn.bfloat4_b: 0.5625, ttnn.bfloat8_b: 1.0625}[dt]
        logger.info(
            f"M={M} {name}: smm {t_smm*1e6:.0f} us ({wbytes/t_smm/1e9:.0f} GB/s), pcc {p:.5f}; "
            f"ttnn 1d {t_1d*1e6:.0f} us ({wbytes/t_1d/1e9:.0f} GB/s)"
        )
        assert p > 0.99
