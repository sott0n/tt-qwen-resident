# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel prefill of Qwen3.6-27B on QB2 (1 x 4 mesh), feeding the resident decode.

Prefill is compute-bound, and tensor parallelism over the 4 chips makes it communication-bound (every
half-layer all-reduces a [chunk, 5120] activation over ethernet). Here chip p instead holds the full
weights of layers [p * Ls, (p + 1) * Ls) (Ls = layers / chips) and the prompt flows through the chips in
chunks of C tokens: in tick t chip p runs its Ls layers on chunk t - p, then hands its output to chip
p + 1. With Ls a multiple of the attention interval every stage has the same layer schedule, so one op
sequence runs on all chips at once (each on its own weights, state and chunk); a tick is the same
program every time, and everything that differs per chip and per tick (positions, which KV blocks to
write, whether the chunk is real) is data in small control tensors written before the tick.

Chunks outside a chip's range (pipeline fill and drain) run on the same program but leave the state
alone: their GDN gates are masked to g = 0, beta = 0 (the delta rule is then the identity on the state),
the conv carry update is skipped, and their KV writes go to scratch blocks. The same masks cover the
padding rows of the last chunk.

State per chip: GDN recurrent states fp32 [48, 128, 128] and conv carries (last 3 conv inputs), and paged
bf8 KV caches of its attention layers. export_state() converts them into the resident decode's
TP-sharded State.
"""
import math

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.experimental.qwen36_resident.prefill import pp_weights as PW
from models.experimental.qwen36_resident.tests.bench_resident_mlp import EPS, HIDDEN, INTER
from models.experimental.qwen36_resident.tests.resident_model import (
    CONV_K,
    DK,
    DV,
    HD,
    NK,
    NKV,
    NQ,
    NV,
    ROPE_THETA,
    ROT,
    State,
    is_attn,
    quantize_rows,
)

KDIR = "models/experimental/qwen36_resident/prefill/kernels/"
QD, VD = NK * DK, NV * DV  # 2048, 6144
CONV_CH = 2 * QD + VD  # 10240
Z0, A0 = CONV_CH, CONV_CH + VD  # z | a | b columns
GDN_COLS = 16512  # q | k | v | z | a | b (16480) padded to 12 x 43 tiles
ATTN_COLS = 2 * NQ * HD + 2 * NKV * HD  # q | gate | k | v = 14336
TILE = 32
MM_SHAPES = dict(
    gdn=(HIDDEN, GDN_COLS), attn=(HIDDEN, ATTN_COLS), out=(VD, HIDDEN), gu=(HIDDEN, INTER), down=(INTER, HIDDEN)
)
# (projection, chunk) -> grid x, grid y, in0_block_w, out_subblock h, w: the fastest of a sweep on QB2
# (LoFi, bf8 / bf4 weights; 250-290 TFLOPS at chunk 1024, 195-240 at 512)
MM_BEST = {
    ("gdn", 1024): (13, 10, 8, 1, 4),
    ("attn", 1024): (12, 8, 8, 4, 2),
    ("out", 1024): (10, 8, 16, 1, 4),
    ("gu", 1024): (13, 10, 16, 4, 2),
    ("down", 1024): (10, 10, 16, 1, 4),
    ("gdn", 512): (13, 8, 4, 1, 4),
    ("attn", 512): (12, 10, 4, 2, 2),
    ("out", 512): (13, 8, 16, 1, 1),
    ("gu", 512): (13, 8, 4, 2, 2),
    ("down", 512): (13, 8, 16, 1, 1),
}


def rope_cos_sin(positions):
    """[T, ROT] cos / sin (both halves) of the rotated dims at the positions"""
    inv = 1.0 / ROPE_THETA ** (torch.arange(0, ROT, 2, dtype=torch.float64) / ROT)
    ang = positions.double()[:, None] * inv[None]
    cos, sin = torch.cos(ang).float(), torch.sin(ang).float()
    return torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)


def rotate_half_matrix():
    """R with (a | b) @ R = (-b | a) on the ROT rotated dims"""
    h = ROT // 2
    r = torch.zeros(ROT, ROT)
    r[torch.arange(h) + h, torch.arange(h)] = -1.0
    r[torch.arange(h), torch.arange(h) + h] = 1.0
    return r


class PPPrefill:
    def __init__(self, mesh, ck, layers, interval, chunk=1024, max_len=None, block=64, mm_cols=10):
        self.mesh, self.n = mesh, mesh.get_num_devices()
        n = self.n
        assert layers % n == 0 and (layers // n) % interval == 0, "every stage needs the same layer schedule"
        self.layers, self.interval, self.Ls, self.C, self.block = layers, interval, layers // n, chunk, block
        self.mm_cols = mm_cols
        self.max_len = max_len or 4 * chunk
        assert chunk % block == 0 and self.max_len % chunk == 0
        self.kinds = [is_attn(j, interval) for j in range(self.Ls)]
        C = chunk
        shard, rep = ttnn.ShardTensorToMesh(mesh, dim=0), ttnn.ReplicateTensorToMesh(mesh)
        DRAM = ttnn.DRAM_MEMORY_CONFIG

        def dev(t, dtype, layout=ttnn.TILE_LAYOUT, mapper=shard):
            out = ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper, memory_config=DRAM)
            ttnn.synchronize_device(mesh)
            return out

        self._dev = dev
        self.mm = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        self.hifi = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )

        self.pc = {key: self._mm_config(key) for key in MM_SHAPES}

        # ---- weights: stage-layer j of chip p is layer p * Ls + j
        self.w = []
        for j in range(self.Ls):
            ws = [PW.layer(ck, p * self.Ls + j, interval) for p in range(n)]
            stack = lambda key, f: torch.stack([f(w[key]) for w in ws])[:, None]
            e = dict(
                G=dev(stack("mlp", lambda m: m["G"]), ttnn.bfloat4_b),
                U=dev(stack("mlp", lambda m: m["U"]), ttnn.bfloat4_b),
                D=dev(stack("mlp", lambda m: m["D"]), ttnn.bfloat8_b),
                out=dev(stack("mixer", lambda m: m["out"]), ttnn.bfloat8_b),
            )
            if self.kinds[j]:
                e["W"] = dev(stack("mixer", lambda m: m["W"]), ttnn.bfloat8_b)
                row = lambda v: v.reshape(1, -1)
                e["wq"] = dev(stack("mixer", lambda m: row(m["wq"] / math.sqrt(HD))), ttnn.bfloat16)
                e["wk"] = dev(stack("mixer", lambda m: row(m["wk"])), ttnn.bfloat16)
            else:

                def padded(m):
                    W = torch.zeros(HIDDEN, GDN_COLS, dtype=torch.bfloat16)
                    W[:, : m["W"].shape[1]] = m["W"]
                    return W

                def taps(m):
                    t = torch.zeros(4 * TILE, CONV_CH)
                    for k in range(CONV_K):
                        t[k * TILE] = m["taps"][:, k]
                    return t

                e["W"] = dev(stack("mixer", padded), ttnn.bfloat8_b)
                e["taps"] = dev(stack("mixer", taps), ttnn.bfloat16)
                e["dt"] = dev(stack("mixer", lambda m: m["dt"].reshape(1, -1)), ttnn.float32)
                e["neg_a"] = dev(stack("mixer", lambda m: m["neg_a"].reshape(1, -1)), ttnn.float32)
            self.w.append(e)
            del ws

        # ---- state
        self.blocks = self.max_len // block
        self.scratch0 = self.blocks  # C // block scratch blocks after the real ones
        kv_blocks = self.blocks + C // block
        self.S, self.carry, self.K, self.V = {}, {}, {}, {}
        for j, attn in enumerate(self.kinds):
            if attn:
                z = torch.zeros(n * kv_blocks, NKV, block, HD)
                self.K[j], self.V[j] = dev(z, ttnn.bfloat8_b), dev(z, ttnn.bfloat8_b)
            else:
                self.S[j] = dev(torch.zeros(n, NV, DK, DV), ttnn.float32)
                self.carry[j] = dev(torch.zeros(n, 1, TILE, CONV_CH), ttnn.bfloat16)

        # ---- constants and buffers
        self.R = dev(torch.stack([rotate_half_matrix()] * n)[:, None], ttnn.bfloat16)
        c = TILE  # the delta-rule op's chunk
        consts = [torch.eye(c), torch.tril(torch.ones(c, c)), torch.ones(c, c)]
        ii, jj = torch.arange(32)[:, None], torch.arange(32)[None]
        lo_i, lo_j = ii < 16, jj < 16
        consts.append(torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1))
        self.gdn_consts = [dev(t.reshape(1, 1, *t.shape), ttnn.float32, mapper=rep) for t in consts]
        self.x_in = dev(torch.zeros(n, 1, C, HIDDEN), ttnn.bfloat16)
        self.x_last = dev(torch.zeros(n, 1, C, HIDDEN), ttnn.bfloat16)  # a stage's output of the last tick
        # the embedding table on every chip (chip 0 looks up the next chunk's rows from its token ids)
        self.embed = dev(
            ck.get("model.language_model.embed_tokens.weight").bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, rep
        )
        self.ids = dev(torch.zeros(n, 1, C, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._shift_program()
        self.conv_y = dev(torch.zeros(n, 1, C, CONV_CH), ttnn.bfloat16)
        self.gate_y = dev(torch.zeros(n, 1, C, VD), ttnn.bfloat16)
        # per tick and chip: [valid rows, update] for the conv, the row mask of the gates, rope, KV blocks
        self.ctrl = dev(torch.zeros(n, 1, 1, 8, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.mask = dev(torch.zeros(n, 1, C, 1), ttnn.float32)
        self.cos = dev(torch.zeros(n, 1, C, ROT), ttnn.bfloat16)
        self.sin = dev(torch.zeros(n, 1, C, ROT), ttnn.bfloat16)
        self.fill_pt = dev(torch.zeros(n, C // block, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.read_pt = dev(torch.zeros(n, self.blocks, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.cstart = dev(torch.zeros(n, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._conv_program()
        self.trace_id = None

    # ------------------------------------------------------------------ pipeline shift
    def _shift_program(self):
        mesh, n = self.mesh, self.n
        core = ttnn.CoreCoord(0, 0)
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        self._shift_sems = [ttnn.create_global_semaphore(mesh, grid, 0) for _ in range(2)]
        pages = self.C * HIDDEN // (TILE * TILE)
        banks = mesh.dram_grid_size().x
        assert pages % banks == 0
        bank_bytes = pages // banks * TILE * TILE * 2
        cbs = [
            ttnn.CBDescriptor(
                total_size=4 * 4096,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=4096)],
            )
        ]
        pd = ttnn.MeshProgramDescriptor()
        for chip in range(n):
            coord = ttnn.MeshCoordinate(0, chip)
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [
                chip,
                n,
                self.x_last.buffer_address(),
                self.x_in.buffer_address(),
                0,
                bank_bytes,
            ] + [ttnn.get_global_semaphore_address(s) for s in self._shift_sems]
            kernel = ttnn.KernelDescriptor(
                kernel_source=KDIR + "shift.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=grid,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
            program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
            args = program.kernels[0].runtime_args[core.x][core.y]
            for nb in (chip + 1, chip - 1):
                if 0 <= nb < n:
                    args.append(1)
                    args.extend(
                        ttnn.setup_fabric_connection(
                            mesh.get_fabric_node_id(coord),
                            mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)),
                            0,
                            program,
                            core,
                        )
                    )
                else:
                    args.append(0)
            pd[ttnn.MeshCoordinateRange(coord, coord)] = program
        self._shift_pd = pd

    # ------------------------------------------------------------------ conv kernel
    def _conv_program(self):
        C, mesh = self.C, self.mesh
        grid = mesh.compute_with_storage_grid_size()
        Ct, Rt, Xw = CONV_CH // TILE, C // TILE, GDN_COLS // TILE
        cores = min(grid.x * grid.y, Ct)
        cols = [(Ct - c + cores - 1) // cores for c in range(cores)]
        f32, bf = (ttnn.float32, 4096), (ttnn.bfloat16, 2048)
        spec = {
            0: (bf, 3),
            1: (bf, 4),
            2: (bf, 1),
            3: (bf, 10),
            4: (f32, 4),
            5: (f32, 4),
            6: (bf, 2),
            7: (bf, 2),
            8: (f32, 1),
            9: (bf, 1),
        }
        self._conv_cores = []
        groups = {}
        for c in range(cores):
            core = ttnn.CoreCoord(c % grid.x, c // grid.x)
            self._conv_cores.append(core)
            groups.setdefault(cols[c], []).append(core)
        self._conv_groups = groups
        self._conv_spec = spec
        self._conv_dims = (Ct, Rt, Xw, cores)

    def _conv(self, p, j):
        """silu(causal conv1d) of the q | k | v columns of p into conv_y; updates carry[j] per ctrl"""
        Ct, Rt, Xw, cores = self._conv_dims
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self._conv_cores])
        cbs = [
            ttnn.CBDescriptor(
                total_size=cnt * page,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
            )
            for i, ((dt, page), cnt) in self._conv_spec.items()
        ]
        rr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(self._conv_cores):
            rr[core.x][core.y] = [
                i,
                cores,
                p.buffer_address(),
                self.carry[j].buffer_address(),
                self.w[j]["taps"].buffer_address(),
            ]
            wr[core.x][core.y] = [
                i,
                cores,
                self.conv_y.buffer_address(),
                self.carry[j].buffer_address(),
                self.ctrl.buffer_address(),
            ]
        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = ttnn.MathFidelity.HiFi4
        compute.fp32_dest_acc_en = True
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "conv_reader.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=all_cores,
                compile_time_args=[Rt, Xw, Ct],
                runtime_args=rr,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "conv_writer.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=all_cores,
                compile_time_args=[Rt, Ct],
                runtime_args=wr,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            ),
        ]
        for cnt, group in self._conv_groups.items():
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "conv_compute.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in group]),
                    compile_time_args=[Rt, cnt],
                    runtime_args=ttnn.RuntimeArgs(),
                    config=compute,
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        mesh_pd = ttnn.MeshProgramDescriptor()
        mesh_pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, self.n - 1))] = program
        ttnn.generic_op([p, self.carry[j], self.w[j]["taps"], self.conv_y, self.ctrl], mesh_pd)
        return self.conv_y

    def _gate(self, o, p):
        """rmsnorm over each head's 128 dims of the head-major delta-rule output o, times silu(z) of the
        projection p, as token-major rows (kernels/gnorm_*.cpp) into gate_y"""
        mesh, C = self.mesh, self.C
        grid = mesh.compute_with_storage_grid_size()
        Rt = C // TILE
        groups = Rt * NV
        cores = min(grid.x * grid.y, groups)
        per = [groups // cores + (i < groups % cores) for i in range(cores)]
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cores)]
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        bf, f32 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
        o_fmt = f32 if o.dtype == ttnn.float32 else bf
        spec = {0: (o_fmt, 8), 1: (bf, 8), 2: (bf, 1), 3: (f32, 4), 4: (f32, 1), 5: (f32, 2), 6: (f32, 2), 7: (bf, 4)}
        cbs = [
            ttnn.CBDescriptor(
                total_size=cnt * page,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
            )
            for i, ((dt, page), cnt) in spec.items()
        ]
        rr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        first, by_count = 0, {}
        for c, cnt in zip(coords, per):
            rr[c.x][c.y] = [first, cnt, o.buffer_address(), p.buffer_address()]
            wr[c.x][c.y] = [first, cnt, self.gate_y.buffer_address()]
            by_count.setdefault(cnt, []).append(c)
            first += cnt
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = ttnn.MathFidelity.HiFi4
        compute.fp32_dest_acc_en = True
        kernel = lambda src, cores_, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores_,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        kernels = [
            kernel(
                "gnorm_reader.cpp",
                all_cores,
                [Rt, NV, GDN_COLS // TILE, Z0 // TILE],
                rr,
                dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
            ),
            kernel(
                "gnorm_writer.cpp",
                all_cores,
                [NV],
                wr,
                dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
            ),
        ]
        eps_bits = int(torch.tensor([EPS], dtype=torch.float32).view(torch.int32)[0])
        for cnt, cs in by_count.items():
            kernels.append(
                kernel(
                    "gnorm_compute.cpp",
                    ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs]),
                    [cnt, eps_bits],
                    ttnn.RuntimeArgs(),
                    compute,
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        pd = ttnn.MeshProgramDescriptor()
        pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, self.n - 1))] = program
        ttnn.generic_op([o, p, self.gate_y], pd)
        return self.gate_y

    # ------------------------------------------------------------------ layers
    def _mm_config(self, key, fused_activation=None):
        """the measured best 2D multicast config of a projection at this chunk (tests/_sweep), else the
        qwen36 prefill heuristic"""
        K, N = MM_SHAPES[key]
        best = MM_BEST.get((key, self.C))
        if best is None:
            try:
                return tpc.create_prefill_mlp_matmul_program_config(
                    self.C, K, N, max_cols=self.mm_cols, fused_activation=fused_activation
                )
            except Exception:
                return None
        gx, gy, bw, sh, sw = best
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=bw,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=math.ceil(self.C // TILE / gy),
            per_core_N=math.ceil(N // TILE / gx),
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=False,
        )

    def _linear(self, x, w, key, activation=None):
        pc = self.pc[key]
        if activation is not None and pc is not None:
            if "silu" not in self.pc:
                self.pc["silu"] = self._mm_config(key, fused_activation=ttnn.UnaryOpType.SILU)
            pc, activation = self.pc["silu"], None
        return ttnn.linear(
            x,
            w,
            compute_kernel_config=self.mm,
            program_config=pc,
            activation=activation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def _gdn(self, x, j):
        C, w = self.C, self.w[j]
        h = ttnn.rms_norm(x, epsilon=EPS)
        p = self._linear(h, w["W"], "gdn")
        ttnn.deallocate(h)
        y = self._conv(p, j)
        # token-major q | k | v rows go to the delta-rule op as they are ([1, C, width] is a view): it
        # L2-normalizes q and k per head itself (folding q's scale) and returns o head-major
        q = ttnn.reshape(ttnn.slice(y, (0, 0, 0, 0), (1, 1, C, QD)), (1, C, QD))
        k = ttnn.reshape(ttnn.slice(y, (0, 0, 0, QD), (1, 1, C, 2 * QD)), (1, C, QD))
        v = ttnn.reshape(ttnn.slice(y, (0, 0, 0, 2 * QD), (1, 1, C, CONV_CH)), (1, C, VD))
        ab = ttnn.typecast(ttnn.slice(p, (0, 0, 0, A0), (1, 1, C, A0 + 2 * NV)), ttnn.float32)
        a = ttnn.slice(ab, (0, 0, 0, 0), (1, 1, C, NV))
        b = ttnn.slice(ab, (0, 0, 0, NV), (1, 1, C, 2 * NV))
        # masked rows / chunks: g = 0, beta = 0 leave the state unchanged
        beta = ttnn.multiply(ttnn.sigmoid(b), self.mask)
        g = ttnn.multiply(ttnn.multiply(ttnn.softplus(ttnn.add(a, w["dt"])), w["neg_a"]), self.mask)
        eye, tril, ones, masks = self.gdn_consts
        o, S = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            ttnn.reshape(g, (1, C, NV)),
            ttnn.reshape(beta, (1, C, NV)),
            scale=DK**-0.5,
            initial_state=self.S[j],
            output_final_state=True,
            chunk_size=TILE,
            output_head_major=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )
        ttnn.copy(S, self.S[j])
        ttnn.deallocate(S)
        # per-head rmsnorm (its weight is folded into out) * silu(z), token-major
        o = self._gate(o, p)
        out = self._linear(o, w["out"], "out")
        return ttnn.add(x, out)

    def _rope(self, t):
        r, rest = ttnn.slice(t, (0, 0, 0, 0), (1, t.shape[1], self.C, ROT)), None
        rest = ttnn.slice(t, (0, 0, 0, ROT), (1, t.shape[1], self.C, HD))
        rot = ttnn.matmul(r, self.R, compute_kernel_config=self.hifi)
        r = ttnn.add(ttnn.multiply(r, self.cos), ttnn.multiply(rot, self.sin))
        return ttnn.concat([r, rest], dim=-1)

    def _attn(self, x, j):
        C, w = self.C, self.w[j]
        h = ttnn.rms_norm(x, epsilon=EPS)
        p = self._linear(h, w["W"], "attn")
        qd = NQ * HD
        gate = ttnn.slice(p, (0, 0, 0, qd), (1, 1, C, 2 * qd))
        qkv = ttnn.concat(
            [ttnn.slice(p, (0, 0, 0, 0), (1, 1, C, qd)), ttnn.slice(p, (0, 0, 0, 2 * qd), (1, 1, C, ATTN_COLS))],
            dim=-1,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NQ, num_kv_heads=NKV, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = self._rope(ttnn.multiply(ttnn.rms_norm(q, epsilon=EPS), w["wq"]))
        k = self._rope(ttnn.multiply(ttnn.rms_norm(k, epsilon=EPS), w["wk"]))
        ttnn.experimental.paged_fill_cache(self.K[j], ttnn.typecast(k, ttnn.bfloat8_b), self.fill_pt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(self.V[j], ttnn.typecast(v, ttnn.bfloat8_b), self.fill_pt, batch_idx=0)
        o = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.K[j],
            self.V[j],
            self.read_pt,
            chunk_start_idx_tensor=self.cstart,
            scale=1.0,
            program_config=self.sdpa_pc,
            compute_kernel_config=self.hifi,
        )
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        out = self._linear(o, w["out"], "out")
        return ttnn.add(x, out)

    def _mlp(self, x, j):
        C, w = self.C, self.w[j]
        h = ttnn.rms_norm(x, epsilon=EPS)
        g = self._linear(h, w["G"], "gu", activation="silu")
        u = self._linear(h, w["U"], "gu")
        a = ttnn.multiply(g, u)
        out = self._linear(a, w["D"], "down")
        return ttnn.add(x, out)

    def _body(self):
        """one tick: stage inputs (chip 0 the chunk's embeddings, chip p > 0 chip p - 1's last output),
        the stage's layers, the stage output kept for the next tick"""
        emb = ttnn.embedding(self.ids, self.embed, layout=ttnn.TILE_LAYOUT)
        ttnn.copy(ttnn.reshape(emb, (1, 1, self.C, HIDDEN)), self.x_in)
        ttnn.generic_op([self.x_last, self.x_in], self._shift_pd)
        x = self.x_in
        for j, attn in enumerate(self.kinds):
            x = self._attn(x, j) if attn else self._gdn(x, j)
            x = self._mlp(x, j)
        ttnn.copy(x, self.x_last)
        return x

    # ------------------------------------------------------------------ ticks
    def _control(self, t, n_tokens):
        """host control tensors of tick t: chip p runs chunk t - p"""
        n, C, blk = self.n, self.C, self.block
        chunks = (n_tokens + C - 1) // C
        ctrl = torch.zeros(n, 1, 1, 8, dtype=torch.int32)
        mask = torch.zeros(n, 1, C, 1)
        cos = torch.zeros(n, 1, C, ROT)
        sin = torch.zeros(n, 1, C, ROT)
        fill = torch.zeros(n, C // blk, dtype=torch.int32)
        read = torch.arange(self.blocks, dtype=torch.int32)[None].repeat(n, 1)
        cstart = torch.zeros(n, dtype=torch.int32)
        for p in range(n):
            c = t - p
            if 0 <= c < chunks:
                valid = min(C, n_tokens - c * C)
                ctrl[p, 0, 0, :2] = torch.tensor([valid, 1])
                mask[p, 0, :valid] = 1.0
                cos[p, 0], sin[p, 0] = rope_cos_sin(torch.arange(c * C, (c + 1) * C))
                fill[p] = torch.arange(c * C // blk, (c + 1) * C // blk)
                cstart[p] = c * C
            else:
                ctrl[p, 0, 0, :2] = torch.tensor([C, 0])
                fill[p] = torch.arange(self.scratch0, self.scratch0 + C // blk)
                cos[p, 0] = 1.0
        shard = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        host = lambda v, dt, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(v, dtype=dt, layout=layout, mesh_mapper=shard)
        return [
            (host(ctrl, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), self.ctrl),
            (host(mask, ttnn.float32), self.mask),
            (host(cos, ttnn.bfloat16), self.cos),
            (host(sin, ttnn.bfloat16), self.sin),
            (host(fill, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.fill_pt),
            (host(read, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.read_pt),
            (host(cstart, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.cstart),
        ]

    def capture(self):
        """compile the tick body, then record it into a trace (replayed by run())"""
        for host, dev in self._control(-1, 1):  # every chip idle: KV writes go to the scratch blocks
            ttnn.copy_host_to_device_tensor(host, dev)
        self.x_out = self._body()
        ttnn.synchronize_device(self.mesh)
        self.reset()
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.x_out = self._body()
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh)

    def reset(self):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.zeros(self.n, 1, self.C, HIDDEN),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            ),
            self.x_last,
        )
        for j in self.S:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.zeros(self.n, NV, DK, DV),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
                ),
                self.S[j],
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.zeros(self.n, 1, TILE, CONV_CH),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
                ),
                self.carry[j],
            )
        ttnn.synchronize_device(self.mesh)

    def run(self, token_ids, timings=None):
        """prefill the prompt token_ids [T]: one tick per chunk plus the pipeline drain"""
        import time

        n, C = self.n, self.C
        T = token_ids.shape[0]
        assert T <= self.max_len
        chunks = (T + C - 1) // C
        ids = torch.zeros(chunks * C, dtype=torch.int32)
        ids[:T] = token_ids.to(torch.int32)
        shard = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        for t in range(chunks + n - 1):
            t0 = time.perf_counter()
            chunk_ids = torch.zeros(n, 1, C, dtype=torch.int32)
            if t < chunks:
                chunk_ids[0, 0] = ids[t * C : (t + 1) * C]
            writes = self._control(t, T) + [
                (
                    ttnn.from_torch(chunk_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard),
                    self.ids,
                )
            ]
            for host, dev in writes:
                ttnn.copy_host_to_device_tensor(host, dev)
            t1 = time.perf_counter()
            if self.trace_id is not None:
                ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            else:
                self._body()
            if timings is not None:
                ttnn.synchronize_device(self.mesh)
                timings.append((t1 - t0, time.perf_counter() - t1))
        ttnn.synchronize_device(self.mesh)
        return T

    # ------------------------------------------------------------------ handoff
    def handoff(self, model, n_tokens):
        """move the state after n_tokens prompt tokens into the resident decode `model` (TP-sharded):
        GDN states and KV caches chip to chip on device (kernels/handoff.cpp), conv histories via the host"""
        import numpy as np

        n, Ls, d = self.n, self.Ls, model.d
        mesh = self.mesh
        if not hasattr(self, "_ho"):
            core = ttnn.CoreCoord(0, 0)
            self._ho = dict(
                core=core, sem=ttnn.create_global_semaphore(mesh, ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), 0)
            )
        core, sem = self._ho["core"], self._ho["sem"]
        banks = d.banks
        kv_row = (HD // TILE) * 1088
        jobs = [[] for _ in range(n)]
        packets_in = [0] * n
        g_i = a_i = 0
        for i in range(self.layers):
            p, j = divmod(i, Ls)
            if self.kinds[j]:
                tiles = (n_tokens + TILE - 1) // TILE
                per_block = self.block // TILE
                for which, src, dst in (("K", self.K[j], model.K_t), ("V", self.V[j], model.V_t)):
                    for c in range(n):
                        for t in range(tiles):
                            bk, rt = divmod(t, per_block)
                            first = ((bk * NKV + c) * per_block + rt) * (HD // TILE)
                            off = (a_i * model.rpb + t // banks) * kv_row
                            jobs[p].append(
                                [
                                    c,
                                    src.buffer_address(),
                                    first,
                                    1088,
                                    1,
                                    dst.buffer_address() + off,
                                    t % banks,
                                    HD // TILE,
                                ]
                            )
                            if c != p:
                                packets_in[c] += -(-kv_row // 4096)
                a_i += 1
            else:
                pages = d.nv * 16
                for c in range(n):
                    jobs[p].append(
                        [
                            c,
                            self.S[j].buffer_address(),
                            c * pages,
                            4096,
                            0,
                            model.state_t.buffer_address(),
                            g_i * pages,
                            pages,
                        ]
                    )
                    if c != p:
                        packets_in[c] += pages
                g_i += 1
        most = max(1, max(len(jb) for jb in jobs))
        table = torch.zeros(n, most, 16, dtype=torch.int64)
        for p in range(n):
            if jobs[p]:
                table[p, : len(jobs[p]), :8] = torch.tensor(jobs[p], dtype=torch.int64)
        table = (table & 0xFFFFFFFF).to(torch.int64).numpy().astype(np.uint32).view(np.int32)
        jobs_t = ttnn.from_torch(
            torch.from_numpy(table.reshape(n * most, 16).copy()),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        cbs = [
            ttnn.CBDescriptor(
                total_size=8 * 4096,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=4096)],
            ),
            ttnn.CBDescriptor(
                total_size=64 * 64,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.uint32, page_size=64)],
            ),
        ]
        mesh_pd = ttnn.MeshProgramDescriptor()
        for chip in range(n):
            coord = ttnn.MeshCoordinate(0, chip)
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [
                jobs_t.buffer_address(),
                len(jobs[chip]),
                chip,
                ttnn.get_global_semaphore_address(sem),
                packets_in[chip],
            ]
            kernel = ttnn.KernelDescriptor(
                kernel_source=KDIR + "handoff.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=grid,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
            program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
            args = program.kernels[0].runtime_args[core.x][core.y]
            for nb in (chip + 1, chip - 1):
                if 0 <= nb < n:
                    args.append(1)
                    args.extend(
                        ttnn.setup_fabric_connection(
                            mesh.get_fabric_node_id(coord),
                            mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)),
                            0,
                            program,
                            core,
                        )
                    )
                else:
                    args.append(0)
            mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
        io = (
            [jobs_t, model.state_t, model.K_t, model.V_t]
            + list(self.S.values())
            + list(self.K.values())
            + list(self.V.values())
        )
        ttnn.generic_op(io, mesh_pd)

        # conv histories (a few MB): carry rows through the host into the decode's history layout
        cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        n_gdn = sum(not k for k in self.kinds) * n
        st = State(d, n_gdn, 1, n_tokens, TILE, zero=True)
        gq, gv = d.gq, d.gv
        # chip-local q | k | v conv columns of every chip
        local = torch.stack(
            [
                torch.cat(
                    [
                        torch.arange(chip * gq, (chip + 1) * gq),
                        QD + torch.arange(chip * gq, (chip + 1) * gq),
                        2 * QD + torch.arange(chip * gv, (chip + 1) * gv),
                    ]
                )
                for chip in range(d.n)
            ]
        )
        carry = {j: cat(self.carry[j]).float().reshape(n, TILE, CONV_CH)[:, : CONV_K - 1] for j in self.carry}
        g_i = 0
        for i in range(self.layers):
            p, j = divmod(i, Ls)
            if self.kinds[j]:
                continue
            cols = carry[j][p][:, local]  # [3, chips, local]
            st.hist[g_i] = [cols[:, chip] for chip in range(d.n)]
            g_i += 1
        model.load_hist(st)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(jobs_t)

    def export_state(self, d, n_tokens, max_pos):
        """the resident decode State (TP-sharded over d.n chips) after n_tokens prompt tokens"""
        n, Ls = self.n, self.Ls
        cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        n_gdn = sum(not a for a in self.kinds) * n
        n_attn = sum(self.kinds) * n
        st = State(d, n_gdn, n_attn, n_tokens, max_pos, zero=True)
        gq, gv = d.gq, d.gv
        S_all = {j: cat(self.S[j]) for j in self.S}
        carry_all = {j: cat(self.carry[j]) for j in self.carry}
        K_all = {j: cat(self.K[j]) for j in self.K}
        V_all = {j: cat(self.V[j]) for j in self.V}
        kv_blocks = K_all[next(iter(K_all))].shape[0] // n if K_all else 0
        g_i = a_i = 0
        for i in range(self.layers):
            p, j = divmod(i, Ls)
            if self.kinds[j]:
                for which, src, dst in (("K", K_all, st.K), ("V", V_all, st.V)):
                    c = src[j].reshape(n, kv_blocks, NKV, self.block, HD)[p, : self.blocks]
                    seq = c.permute(1, 0, 2, 3).reshape(NKV, self.blocks * self.block, HD)[:, :n_tokens]
                    for chip in range(d.n):
                        rows = torch.zeros(max_pos, HD)
                        rows[:n_tokens] = seq[chip]
                        dst[a_i][chip] = quantize_rows(rows)
                a_i += 1
            else:
                S = S_all[j].reshape(n, NV, DK, DV)[p]
                carry = carry_all[j].reshape(n, TILE, CONV_CH)[p, : CONV_K - 1]
                for chip in range(d.n):
                    st.gdn[g_i][chip] = S[chip * d.nv : (chip + 1) * d.nv].clone()
                    st.hist[g_i][chip] = torch.cat(
                        [
                            carry[:, chip * gq : (chip + 1) * gq],
                            carry[:, QD + chip * gq : QD + (chip + 1) * gq],
                            carry[:, 2 * QD + chip * gv : 2 * QD + (chip + 1) * gv],
                        ],
                        dim=1,
                    )
                g_i += 1
        return st
