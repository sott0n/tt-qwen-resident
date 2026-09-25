# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident decode design: the weight-streaming engine on one Blackhole chip.

One resident program streams the per-layer projection weights of Qwen3.6-27B at TP=4
(qkvzab bfp8, out bfp8, gate|up bfp4, down bfp8) for `layers` layers back to back, computing each
M=1 matmul on the DRAM-bank cores. Without delay it measures sustained weight bandwidth across matmul
boundaries; with delay_us > 0 it inserts a modeled activation-chain delay between matmuls (the reader is not gated),
checking that weight streaming hides it: time per layer should stay ~max(weights, chain), not the sum.
"""
import json
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import (
    pad_to_dram_banks,
    shuffle_tensor_tiles,
)

OUT = os.environ.get("BENCH_OUT", "/tmp/stream_engine.jsonl")
KDIR = "models/experimental/qwen36_resident/kernels/"
TILE = 32
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}

# (name, K, N, dtype): per-device projections of one GDN layer at TP=4.
ENTRIES = [
    ("qkvzab", 5120, 4128, ttnn.bfloat8_b),
    ("out", 1536, 5120, ttnn.bfloat8_b),
    ("gate_up", 5120, 8704, ttnn.bfloat4_b),
    ("down", 4352, 5120, ttnn.bfloat8_b),
]
WEIGHT_SETS = 4
L1_WEIGHT_BUDGET = 1_000_000  # bytes of weight CBs per core (prefetch window)
AICLK_HZ = 1.35e9


def block_geometry(Kt, dtype, max_block=20_000, max_page=16_384):
    tb = TILE_BYTES[dtype]
    sb = max(d for d in range(1, Kt + 1) if Kt % d == 0 and d % 2 == 0 and d * tb <= max_block)  # custom_mm: even kt
    block = sb * tb
    page = (max_page // tb) * tb
    while block % page:
        page -= tb
    return sb, block // page, page, block


def streamer_cores(device, per_bank):
    """per_bank cores per DRAM bank: the bank-adjacent core plus its nearest free neighbours in the same column."""
    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    grid = device.compute_with_storage_grid_size()
    used = {(c.x, c.y) for c in primary}
    groups = []
    for c in primary:
        g = [c]
        for dy in (1, -1, 2, -2, 3, -3):
            if len(g) == per_bank:
                break
            y = c.y + dy
            if 0 <= y < grid.y and (c.x, y) not in used:
                used.add((c.x, y))
                g.append(ttnn.CoreCoord(c.x, y))
        assert len(g) == per_bank
        groups.append(g)
    return groups


def build(device, layers, iters, delays_us, per_bank=1):
    torch.manual_seed(0)
    groups = streamer_cores(device, per_bank)
    banks = device.dram_grid_size().x
    assert len(groups) == banks
    cores = [c for g in groups for c in g]  # bank-major: core i serves bank i // per_bank
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})

    k_max = max(k for _, k, _, _ in ENTRIES)
    x = torch.randn(1, 1, 1, k_max).bfloat16().float()
    tiny = ttnn.Tile([1, TILE])
    in0 = ttnn.from_torch(
        x.repeat(1, 1, len(cores), 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=tiny,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [1, k_max], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )

    geo, weights, per_bank_tiles, torch_w = [], [], [], []
    for name, k, n, dt in ENTRIES:
        n_pad = pad_to_dram_banks(n, TILE, TILE * banks)
        per_bank_tiles.append(n_pad // banks // TILE)
        geo.append(block_geometry(k // TILE, dt))
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [k, n_pad // banks], ttnn.ShardOrientation.ROW_MAJOR),
        )
        sets = []
        for s in range(WEIGHT_SETS):
            w = torch.randn(1, 1, k, n_pad).bfloat16().float()
            if s == (layers - 1) % WEIGHT_SETS:
                torch_w.append(w)
            sets.append(
                ttnn.from_torch(
                    shuffle_tensor_tiles(w, TILE, banks),
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=mc,
                )
            )
        weights.append(sets)

    # column split of each bank shard over its per_bank cores: (n_tiles, first tile) per slot
    split = [
        [(pb // per_bank + (j < pb % per_bank), j * (pb // per_bank) + min(j, pb % per_bank)) for j in range(per_bank)]
        for pb in per_bank_tiles
    ]
    out_cols = sum(sp[0][0] for sp in split) * TILE
    out = ttnn.from_torch(
        torch.zeros(1, 1, len(cores), out_cols),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=tiny,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [1, out_cols], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )

    # CBs: activation (tensor-backed); one weight ring aliased by the per-entry CBs 1..4 (blocks are
    # placed in schedule order, so the whole ring is the prefetch window); out; gate; consumed word.
    lcm = math.lcm(*TILE_BYTES.values())
    ring_bytes = (L1_WEIGHT_BUDGET // lcm) * lcm
    assert all(ring_bytes >= 4 * g[3] for g in geo)
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, in0)]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=ring_bytes,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1 + e, data_format=dt, page_size=TILE_BYTES[dt])
                for e, (_, _, _, dt) in enumerate(ENTRIES)
            ],
        )
    )
    for idx, pages in ((5, 4), (6, 2)):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=pages * 64,
                core_ranges=grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=idx, data_format=ttnn.bfloat16, page_size=64, tile=ttnn.TileDescriptor(1, TILE)
                    )
                ],
            )
        )
    cbs.append(
        ttnn.CBDescriptor(
            total_size=16,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=7, data_format=ttnn.uint32, page_size=16)],
        )
    )

    ct = [layers, iters, WEIGHT_SETS]
    for (_, k, _, _), (sb, pages, page, block) in zip(ENTRIES, geo):
        ct += [k // TILE, sb, pages, page, block]
    ct.append(ring_bytes)

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    delay_cycles = [int(d * 1e-6 * AICLK_HZ) for d in delays_us]
    for i, c in enumerate(cores):
        bank, j = divmod(i, per_bank)
        r = [bank, i & 0x3]
        for (_, k, _, dt), sp in zip(ENTRIES, split):
            n_t, first = sp[j]
            r += [n_t, first * (k // TILE) * TILE_BYTES[dt]]
        for s in range(WEIGHT_SETS):
            r += [weights[e][s].buffer_address() for e in range(len(ENTRIES))]
        reader_rt[c.x][c.y] = r
        w = [out.buffer_address()]
        off = 0
        for e, sp in enumerate(split):
            w += [sp[j][0], off, delay_cycles[e]]
            off += sp[0][0]
        writer_rt[c.x][c.y] = w
        compute_rt[c.x][c.y] = [sp[j][0] for sp in split]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "stream_engine_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=ct,
            runtime_args=reader_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "stream_engine_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=ct,
            runtime_args=writer_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default
            ),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "stream_engine_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=ct,
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(),
        ),
    ]
    kernels[2].config.math_fidelity = ttnn.MathFidelity.LoFi
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    io = [in0] + [w for sets in weights for w in sets] + [out]
    bytes_per_layer = sum(
        k * pb * banks * TILE * TILE_BYTES[dt] / (TILE * TILE) for (_, k, _, dt), pb in zip(ENTRIES, per_bank_tiles)
    )
    return (lambda: ttnn.generic_op(io, program)), out, x, torch_w, split, len(cores), bytes_per_layer


def timed(device, fn, reps=3):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / reps


@pytest.mark.parametrize("per_bank", [1, 2], ids=["1pb", "2pb"])
@pytest.mark.parametrize("delay_us", [0.0, 10.0, 30.0], ids=["no_chain", "chain10us", "chain30us"])
def test_stream_engine(device, delay_us, per_bank):
    layers, iters = 16, 8
    delays = [delay_us] * len(ENTRIES)
    run1, _, _, _, _, _, _ = build(device, layers, 1, delays, per_bank)
    runN, out, x, torch_w, split, ncores, bytes_per_layer = build(device, layers, iters, delays, per_bank)
    t1 = timed(device, run1)
    tN = timed(device, runN)
    per_layer = (tN - t1) / ((iters - 1) * layers)

    # correctness: last layer's outputs
    got = ttnn.to_torch(out)[0, 0]  # [cores, out_cols]
    xs = x[0, 0, 0]
    worst = 1.0
    banks = ncores // per_bank
    for j in range(per_bank):  # the cores of bank 0
        col0 = 0
        for e, ((name, k, n, dt), w) in enumerate(zip(ENTRIES, torch_w)):
            ref = xs[:k] @ w[0, 0]  # [n_pad]; bank 0 shard = first per_bank_tiles columns
            n_t, first = split[e][j]
            ref_c = ref[first * TILE : (first + n_t) * TILE]
            g = got[j, col0 : col0 + n_t * TILE]
            p = torch.corrcoef(torch.stack([g.double(), ref_c.double()]))[0, 1].item()
            worst = min(worst, p)
            col0 += split[e][0][0] * TILE
    weight_us = bytes_per_layer / 360e9 * 1e6
    chain_us = delay_us * len(ENTRIES)
    rec = dict(
        delay_us=delay_us,
        layers=layers,
        cores=ncores,
        per_bank=per_bank,
        us_per_layer=per_layer * 1e6,
        GBps=bytes_per_layer / per_layer / 1e9,
        weight_us_at_360=weight_us,
        modeled_chain_us=chain_us,
        sum_us=weight_us + chain_us,
        pcc_worst=worst,
    )
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert worst > 0.98
