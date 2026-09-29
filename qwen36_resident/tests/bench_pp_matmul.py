# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sweep of 2D multicast matmul configs for the prefill projections (the source of MM_BEST in
prefill/pp_prefill.py): grid, in0 block width and output subblock per projection and chunk (RESIDENT_CHUNKS),
LoFi with the model's weight formats, timed as traced replays."""
import time, os, math, itertools, torch, pytest, ttnn
from loguru import logger

SHAPES = dict(
    gdn=(5120, 16512, ttnn.bfloat8_b),
    attn=(5120, 14336, ttnn.bfloat8_b),
    out=(6144, 5120, ttnn.bfloat8_b),
    gu=(5120, 17408, ttnn.bfloat4_b),
    down=(17408, 5120, ttnn.bfloat8_b),
)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_pp_matmul_sweep(mesh_device):
    cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    G = mesh_device.compute_with_storage_grid_size()
    names = os.environ.get("RESIDENT_SHAPES", ",".join(SHAPES)).split(",")
    for C in [int(c) for c in os.environ.get("RESIDENT_CHUNKS", "512,1024").split(",")]:
        x = ttnn.from_torch(
            torch.randn(1, 1, C, 17408),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        Mt = C // 32
        for name in names:
            K, N, dt = SHAPES[name]
            xin = ttnn.slice(x, (0, 0, 0, 0), (1, 1, C, K))
            w = ttnn.from_torch(
                torch.randn(K, N) * 0.02, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
            )
            Kt, Nt = K // 32, N // 32

            def timeit(pc):
                run = lambda: ttnn.linear(
                    xin,
                    w,
                    compute_kernel_config=cfg,
                    program_config=pc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )
                o = run()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(o)
                tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                outs = [run() for _ in range(4)]
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                for _ in range(3):
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                dt_ = (time.perf_counter() - t0) / 12
                ttnn.release_trace(mesh_device, tid)
                for o in outs:
                    ttnn.deallocate(o)
                return dt_

            res = []
            try:
                res.append((timeit(None), "auto"))
            except Exception as e:
                pass
            for gx, gy in itertools.product([8, 10, 11, 12, 13], [8, 10]):
                if gx > G.x or gy > G.y:
                    continue
                pm, pn = math.ceil(Mt / gy), math.ceil(Nt / gx)
                for bw in [b for b in (2, 4, 8, 16) if Kt % b == 0]:
                    for sh, sw in [(1, 4), (2, 4), (4, 2), (1, 8), (2, 2), (1, 2), (1, 1)]:
                        if pm % sh or pn % sw:
                            continue
                        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(gx, gy),
                            in0_block_w=bw,
                            out_subblock_h=sh,
                            out_subblock_w=sw,
                            per_core_M=pm,
                            per_core_N=pn,
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=False,
                        )
                        try:
                            res.append((timeit(pc), f"g{gx}x{gy} bw{bw} sb{sh}x{sw} pm{pm} pn{pn}"))
                        except Exception as e:
                            pass
                        break  # largest valid subblock only
            res.sort()
            fl = 2 * C * K * N
            logger.info(
                f"C={C} {name}: "
                + " | ".join(f"{d*1e6:.0f}us {fl/d/1e12:.0f}TF {s}" for d, s in res[:4])
                + f" | auto={[f'{d*1e6:.0f}' for d, s in res if s=='auto']}"
            )
            ttnn.deallocate(w)
            ttnn.deallocate(xin)
