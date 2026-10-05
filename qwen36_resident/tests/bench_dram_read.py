# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure DRAM read bandwidth of one Blackhole chip, as a ceiling for the weight-streaming engine.

Readers (NCRISC and optionally BRISC on 1..4 cores per bank, bank-adjacent first) stream disjoint
contiguous regions of their DRAM bank into an L1 ring with `depth` trid-tracked packets in flight;
nothing consumes the data. Sweeps cores per bank x packet size x depth x readers per core (optionally one RISC issuing on both
NOCs), with all 8 banks active or bank 0 only, and the readers of a bank on disjoint regions or
interleaved packet by packet. Bandwidth is the slope between two iteration counts (launch cost cancels).
"""
import itertools
import json
import os
import time

import torch
from loguru import logger

import ttnn
import qwen36_resident.tests.bench_resident_mlp as M
from qwen36_resident import PKG_DIR

OUT = os.environ.get("BENCH_OUT", "/tmp/dram_read.jsonl")
KERNEL = f"{PKG_DIR}/kernels/dram_read_bench.cpp"
BANK_BYTES = 8 << 20
MAX_DEPTH = 15


def build(
    device,
    dram,
    scratch,
    cores_per_bank,
    packet,
    depth,
    iters,
    two_riscs,
    banks_active,
    both_nocs=False,
    interleave=False,
):
    M.PER_BANK = cores_per_bank
    groups, _ = M.streamer_cores(device)
    groups = groups[:banks_active]
    readers_per_bank = cores_per_bank * (2 if two_riscs else 1)
    region = BANK_BYTES // readers_per_bank // packet * packet
    packets = region // packet
    cores = [c for g in groups for c in g]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    kernels = []
    for risc, proc, noc in (
        (0, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_0_default),
        (1, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_1_default),
    ):
        if risc == 1 and not two_riscs:
            continue
        rt = ttnn.RuntimeArgs()
        for b, g in enumerate(groups):
            for j, c in enumerate(g):
                r = j * (2 if two_riscs else 1) + risc
                offset, stride = (r * packet, readers_per_bank * packet) if interleave else (r * region, packet)
                rt[c.x][c.y] = [
                    b,
                    dram.buffer_address(),
                    offset,
                    scratch.buffer_address() + risc * MAX_DEPTH * 16384,
                    r & 0x3,
                    stride,
                ]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=grid,
                compile_time_args=[packet, depth, iters, packets, int(both_nocs)],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
            )
        )
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
    total = banks_active * readers_per_bank * region * iters
    return (lambda: ttnn.generic_op([dram, scratch], program)), total


def timed(device, fn, reps=3):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / reps


def test_dram_read_sweep(device):
    banks = device.dram_grid_size().x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    cols = BANK_BYTES // 2 // 2048
    dram = ttnn.from_torch(
        torch.zeros(2048, cols * banks).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [2048, cols], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    scratch_elems = 2 * MAX_DEPTH * 16384 // 2
    scratch = ttnn.from_torch(
        torch.zeros(grid.x * grid.y, scratch_elems).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(all_cores, [1, scratch_elems], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    sweep = os.environ.get("DRAM_SWEEP", "full")
    if sweep == "full":
        configs = list(itertools.product([1, 2, 4], [2048, 4096, 8192, 16384], [2, 4, 8, 15], [False], [banks]))
        configs += list(itertools.product([1, 2, 4], [8192, 16384], [8, 15], [True], [banks]))
        configs += list(itertools.product([1, 2, 4], [16384], [15], [False, True], [1]))
    elif sweep == "list":
        configs = [
            tuple(int(v) if v.isdigit() else v == "True" for v in c.split(","))
            for c in os.environ["DRAM_CONFIGS"].split(";")
        ]
    else:
        configs = [(4, 16384, 15, False, banks, False, False)]
    configs = [c + (False,) * (7 - len(c)) for c in configs]
    for cpb, packet, depth, two, active, both, inter in configs:
        it_lo, it_hi = int(os.environ.get("DRAM_ITERS_LO", "2")), int(os.environ.get("DRAM_ITERS_HI", "10"))
        reps = int(os.environ.get("DRAM_REPS", "3"))
        lo, total_lo = build(device, dram, scratch, cpb, packet, depth, it_lo, two, active, both, inter)
        hi, total_hi = build(device, dram, scratch, cpb, packet, depth, it_hi, two, active, both, inter)
        t_lo, t_hi = timed(device, lo, reps), timed(device, hi, reps)
        gbps = (total_hi - total_lo) / (t_hi - t_lo) / 1e9
        rec = dict(
            cores_per_bank=cpb,
            packet=packet,
            depth=depth,
            two_riscs=two,
            banks=active,
            both_nocs=both,
            interleave=inter,
            GBps=gbps,
            GBps_per_bank=gbps / active,
        )
        logger.info(rec)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
