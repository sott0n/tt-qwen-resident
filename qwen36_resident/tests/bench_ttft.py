# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time to first token of Qwen3.6-27B on QB2 as a user sees it: the resident decode model is resident, the
prompt goes through the pipeline-parallel prefill (chunk size picked per prompt), the state is handed to the
decode model, and the prompt's last token runs one decode step to the first generated token."""
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State

OUT = os.environ.get("BENCH_OUT", "/tmp/resident_ttft.jsonl")
DECODE_STEPS = int(os.environ.get("RESIDENT_DECODE_STEPS", "32"))


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 128 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_ttft(mesh_device):
    layers = int(os.environ.get("RESIDENT_LAYERS", "64"))
    chunks = [int(c) for c in os.environ.get("RESIDENT_CHUNK", "128,256,512,1024").split(",")]
    prompts = [int(p) for p in os.environ.get("RESIDENT_PROMPTS", "127,255,511,1024,2048,8192").split(",")]
    ck = Q.Checkpoint()
    n = mesh_device.get_num_devices()
    d = Dims(n, mesh_device.dram_grid_size().x, vocab=ck.config["vocab_size"])
    interval = ck.config["full_attention_interval"]
    emb = Q.embedding(ck)
    n_attn = sum((l + 1) % interval == 0 for l in range(layers))
    C = max(chunks)
    max_len = (max(prompts) + C - 1) // C * C
    w = Q.load(ck, d, layers, interval)
    st = State(d, layers - n_attn, n_attn, 0, max_pos=max_len + 64 + DECODE_STEPS, zero=True)
    model = ResidentModel(mesh_device, d, w, st, layers, interval, lm_head=True)
    del w
    model.step(model.token_state(emb[0].float(), 0))  # compile outside the trace
    model.reset()
    model.capture_trace()
    pp = PPPrefill(mesh_device, ck, layers, interval, chunk=chunks, max_len=max_len)
    pp.capture()
    pp.handoff(model, 1)  # compiles the handoff kernel
    g = torch.Generator().manual_seed(0)
    for T in prompts:
        ids = torch.randint(0, ck.config["vocab_size"], (T + 1,), generator=g)
        for rep in range(2):
            model.reset()
            pp.reset()
            parts = {}
            t0 = time.perf_counter()
            pp.run(ids[:T])
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            pp.handoff(model, T, timings=parts if rep else None)
            t2 = time.perf_counter()
            model.step(model.token_state(emb[ids[T]].float(), T))
            model.argmax()
            t3 = time.perf_counter()
        # greedy decode after the prompt: one step and the argmax read-back per token
        toks = [model.token_state(emb[ids[k % (T + 1)]].float(), T + 1 + k) for k in range(DECODE_STEPS)]
        step_s = []
        for tok in toks:
            s0 = time.perf_counter()
            model.step(tok)
            model.argmax()
            step_s.append(time.perf_counter() - s0)
        rec = dict(
            layers=layers,
            prompt=T,
            chunk=pp.C,
            ttft_s=t3 - t0,
            prefill_s=t1 - t0,
            handoff_s=t2 - t1,
            first_step_s=t3 - t2,
            handoff_parts={k: round(v, 4) for k, v in parts.items()},
            decode_ms_median=round(sorted(step_s)[len(step_s) // 2] * 1e3, 2),
        )
        logger.info(rec)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
    model.release_trace()
