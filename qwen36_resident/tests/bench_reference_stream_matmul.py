# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reference point for the resident decode design: how fast does a DRAM-streaming M=1 matmul run
when it loops inside one resident kernel (no op boundaries), at the Qwen3.6-27B TP=4 shapes?

Uses the deepseek_v3_b1 DRAMStreamingMatmul micro-op (one DRAM-adjacent core per bank, K-contiguous
weight sticks, triple-buffered reads, 1x32 activation tiles). The per-iteration time is the slope
between an N-iteration and a 1-iteration launch, so launch/dispatch cost cancels out.
"""
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import (
    pad_to_dram_banks,
    shuffle_tensor_tiles,
)

OUT = os.environ.get("BENCH_OUT", "/tmp/ref_stream_matmul.jsonl")
BYTES_PER_ELEM = {ttnn.bfloat4_b: 0.5625, ttnn.bfloat8_b: 1.0625}


def build(device, k, n, wdtype, loop_iters):
    tile_w = 32
    in0_tile = ttnn.Tile([1, tile_w])
    compute_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)
    num_banks = device.dram_grid_size().x
    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )

    in0 = torch.randn(1, 1, 1, k).bfloat16().float()
    in1 = torch.randn(1, 1, k, n_padded).bfloat16().float()
    in0_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [1, k], ttnn.ShardOrientation.ROW_MAJOR),
    )
    in0_t = ttnn.from_torch(
        in0.repeat(1, 1, num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mc,
        tile=in0_tile,
    )
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
            )
        }
    )
    in1_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [k, n_padded // num_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1_t = ttnn.from_torch(
        shuffle_tensor_tiles(in1, tile_w, num_banks),
        dtype=wdtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_mc,
    )
    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (1, n_padded // num_banks), ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_t = ttnn.from_torch(
        torch.zeros(1, 1, 1, n_padded),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mc,
        tile=in0_tile,
    )
    Kt = k // tile_w
    subblock_k = next(s for s in (Kt // 4, Kt // 2, Kt) if s > 0 and Kt % s == 0)
    wb_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (tile_w, subblock_k * 3 * tile_w), ttnn.ShardOrientation.ROW_MAJOR),
    )
    wb_t = ttnn.from_torch(
        torch.zeros(1, 1, tile_w, subblock_k * 3 * tile_w * num_cores),
        dtype=wdtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=wb_mc,
        tile=ttnn.Tile([tile_w, tile_w]),
    )

    def run(iters):
        return DRAMStreamingMatmul.op(
            in0_t,
            in1_t,
            out_t,
            fp32_dest_acc_en=False,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            subblock_k=subblock_k,
            num_loop_iters=iters,
            working_buf_tensor=wb_t,
        )

    return run, k * n_padded * BYTES_PER_ELEM[wdtype], num_cores


def timed(device, fn, reps=5):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / reps


@pytest.mark.parametrize(
    "k, n, wdtype",
    [
        (7168, 2048, ttnn.bfloat4_b),  # deepseek reference shape (sanity)
        (5120, 8704, ttnn.bfloat4_b),  # MLP gate|up per device
        (5120, 4128, ttnn.bfloat8_b),  # GDN qkvzab
        (4352, 5120, ttnn.bfloat8_b),  # MLP down
        (1536, 5120, ttnn.bfloat8_b),  # GDN / attention out-proj
        (5120, 8704, ttnn.bfloat8_b),  # gate|up if kept bfp8
    ],
    ids=["ds_ref", "gate_up_bfp4", "qkvzab_bfp8", "down_bfp8", "out_bfp8", "gate_up_bfp8"],
)
@pytest.mark.parametrize("iters", [100])
def test_reference_stream_matmul(device, k, n, wdtype, iters):
    torch.manual_seed(0)
    run, weight_bytes, num_cores = build(device, k, n, wdtype, iters)
    if os.environ.get("BENCH_MODE", "both") == "single":
        # One launch only: per-iteration time includes 1/iters of the launch cost.
        t1 = 0.0
        tn = timed(device, lambda: run(iters), reps=1)
        per_iter = tn / iters
    else:
        t1 = timed(device, lambda: run(1))
        tn = timed(device, lambda: run(iters))
        per_iter = (tn - t1) / (iters - 1)
    rec = dict(
        k=k,
        n=n,
        dtype=str(wdtype),
        cores=num_cores,
        us_per_iter=per_iter * 1e6,
        GBps=weight_bytes / per_iter / 1e9,
        launch_1iter_us=t1 * 1e6,
    )
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
