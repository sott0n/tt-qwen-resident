# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident decode design: all-reduce latency when it runs inside a resident kernel.

The residual stream of Qwen3.6-27B at TP=4 is replicated, so each decoder layer needs two all-reduces
of a [1, 5120] bf16 vector (10 KB) across the 4 chips of QB2. The ttnn op-by-op path pays ~14 us per
CCL op. Here one kernel per chip runs `rounds` serial one-shot all-reduce transports (line multicast
of the local vector into every peer's receive slot + fused semaphore increment), and the per-round
latency is the slope between a long and a 1-round launch.
"""
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

OUT = os.environ.get("BENCH_OUT", "/tmp/ccl_allreduce.jsonl")
KERNEL = "models/experimental/qwen36_resident/kernels/ccl_allreduce_bench.cpp"
CORE = ttnn.CoreCoord(0, 0)


def build(mesh_device, width, rounds, packet_bytes):
    n = mesh_device.get_num_devices()
    rows, cols = tuple(mesh_device.shape)
    assert rows == 1, "1D line along the mesh columns"
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(CORE, CORE)})
    vector_bytes = width * 2

    def l1_rows(h):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [h, width], ttnn.ShardOrientation.ROW_MAJOR),
        )

    src_torch = torch.randn(n, width).bfloat16()
    src = ttnn.from_torch(
        src_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=l1_rows(1),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    recv = ttnn.from_torch(
        torch.zeros(2 * n, width).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=l1_rows(2 * n),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sem = ttnn.create_global_semaphore(mesh_device, grid, 0)
    sem_addr = ttnn.get_global_semaphore_address(sem)

    mesh_pd = ttnn.MeshProgramDescriptor()
    for j in range(cols):
        coord = ttnn.MeshCoordinate(0, j)
        rt = ttnn.RuntimeArgs()
        rt[CORE.x][CORE.y] = [src.buffer_address(), recv.buffer_address(), sem_addr]
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[n, j, vector_bytes, packet_bytes, rounds],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
        me = mesh_device.get_fabric_node_id(coord)
        args = program.kernels[0].runtime_args[CORE.x][CORE.y]
        for nb in (j + 1, j - 1):  # forward (towards higher index), then backward
            if 0 <= nb < cols:
                args.append(1)
                args.extend(
                    ttnn.setup_fabric_connection(
                        me, mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)), 0, program, CORE
                    )
                )
            else:
                args.append(0)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
    # The launcher holds the semaphore: once freed, its L1 word can be reallocated under another tensor
    # while a launch of this program still increments it.
    return (lambda keep=sem: ttnn.generic_op([src, recv], mesh_pd)), src_torch, recv


def timed(mesh_device, fn, reps=5):
    fn()
    ttnn.synchronize_device(mesh_device)
    t = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh_device)  # semaphores are rearmed at kernel end; no launch overlap
        t += time.perf_counter() - t0
    return t / reps


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("width", [5120])
@pytest.mark.parametrize("packet_bytes", [2048, 4096])
def test_ccl_allreduce(mesh_device, width, packet_bytes):
    torch.manual_seed(0)
    rounds = 500
    max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    if packet_bytes > max_payload:
        pytest.skip(f"packet {packet_bytes} > fabric max payload {max_payload}")
    run1, _, _ = build(mesh_device, width, 1, packet_bytes)
    runN, src_torch, recv = build(mesh_device, width, rounds, packet_bytes)
    t1 = timed(mesh_device, run1)
    tN = timed(mesh_device, runN)
    per_round = (tN - t1) / (rounds - 1)

    n = mesh_device.get_num_devices()
    got = ttnn.to_torch(recv, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape(n, 2 * n, width)
    last = (rounds - 1) & 1
    match = [[bool(torch.equal(got[d, last * n + s], src_torch[s])) for s in range(n)] for d in range(n)]
    ok = all(all(r) for r in match)
    if not ok:
        logger.warning(f"slot match [dst][src]: {match}")
        for d in range(n):
            for s in range(n):
                if not match[d][s]:
                    bad = (got[d, last * n + s] != src_torch[s]).nonzero().flatten()
                    logger.warning(
                        f"dst {d} src {s}: {bad.numel()} bad elems, first {bad[:4].tolist()} last {bad[-1:].tolist()}"
                    )
    rec = dict(
        width=width,
        packet_bytes=packet_bytes,
        max_payload=max_payload,
        rounds=rounds,
        us_per_allreduce=per_round * 1e6,
        launch_1round_us=t1 * 1e6,
        exact=ok,
    )
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert ok
