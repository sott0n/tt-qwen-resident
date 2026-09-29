# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B on QB2 with the resident decode step: token accuracy against the HF reference.

Teacher forcing over the committed reference (tt_transformers TokenAccuracy semantics): the first half of
the reference tokens is the prompt, and the model's top-1 after every token from position 511 on is
compared with the reference top-1 / top-5 at that position. The resident model has no prefill yet, so the
prompt runs through the decode step one token at a time as well.

test_resident_repeatable reruns the first reference steps of a few real layers from the same state and
requires bit-identical results: with real weights and the lm_head, and with host gaps between the steps,
the chips drift apart in time enough to expose races in the cross-chip all-reduce.
"""
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.qwen36_resident.prefill.pp_prefill import PPPrefill
from models.experimental.qwen36_resident.tests import qwen36_weights as Q
from models.experimental.qwen36_resident.tests.resident_model import Dims, ResidentModel, State

REFPT = os.environ.get("RESIDENT_REFPT", "models/tt_transformers/tests/reference_outputs/Qwen3.6-27B.refpt")
OUT = os.environ.get("BENCH_OUT", "/tmp/resident_accuracy.jsonl")
LAYERS = int(os.environ.get("RESIDENT_LAYERS", "64"))
TRACE = os.environ.get("RESIDENT_TRACE", "1") == "1"
PREFILL = os.environ.get("RESIDENT_PREFILL", "0") == "1"
CHUNK = int(os.environ.get("RESIDENT_CHUNK", "512"))
TRACE_REGION = 64 << 20


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_accuracy(mesh_device):
    ref = torch.load(REFPT, weights_only=False)
    tokens = ref["reference_tokens"][0]
    top5 = ref["top5_tokens"]
    split = tokens.shape[-1] // 2
    ck = Q.Checkpoint()
    n = mesh_device.get_num_devices()
    vocab = ck.config["vocab_size"]
    d = Dims(n, mesh_device.dram_grid_size().x, vocab=vocab)
    interval = ck.config["full_attention_interval"]
    t0 = time.time()
    w = Q.load(ck, d, LAYERS, interval)
    emb = Q.embedding(ck)
    n_attn = sum((l + 1) % interval == 0 for l in range(LAYERS))
    st = State(d, LAYERS - n_attn, n_attn, 0, max_pos=tokens.shape[-1] + 64, zero=True)
    model = ResidentModel(mesh_device, d, w, st, LAYERS, interval, lm_head=True)
    del w
    logger.info(f"weights loaded and placed in {time.time() - t0:.0f} s")
    if TRACE:
        model.step(model.token_state(emb[tokens[0]].float(), 0))  # compile outside the trace
        model.reset()
        model.capture_trace()
    first = 0
    prefill = {}
    if PREFILL:
        # the prompt up to the first scored position through the pipeline-parallel prefill
        first = split - 1
        pp = PPPrefill(mesh_device, ck, LAYERS, interval, chunk=CHUNK, max_len=(first + CHUNK - 1) // CHUNK * CHUNK)
        if TRACE:
            pp.capture()
        pp.handoff(model, 1)  # compiles the handoff kernel
        model.reset()
        pp.reset()
        t1 = time.perf_counter()
        pp.run(tokens[:first])
        t2 = time.perf_counter()
        pp.handoff(model, first)
        t3 = time.perf_counter()
        prefill = dict(prefill_tokens=first, prefill_s=t2 - t1, handoff_s=t3 - t2)
    top1 = top5_hits = count = 0
    step_times, token_times = [], []
    for pos in range(first, tokens.shape[-1] - 1):
        # one token: host token state (embedding row, position, rope), the step, the device argmax
        t0 = time.perf_counter()
        tok = model.token_state(emb[tokens[pos]].float(), pos)
        t1 = time.perf_counter()
        model.step(tok)
        ttnn.synchronize_device(mesh_device)
        step_times.append(time.perf_counter() - t1)
        pred = model.argmax()
        token_times.append(time.perf_counter() - t0)
        if pos >= split - 1:
            top1 += pred == int(top5[pos, 0])
            top5_hits += pred in top5[pos].tolist()
            count += 1
            if count % 64 == 0:
                logger.info(f"pos {pos}: top-1 {100 * top1 / count:.2f}% top-5 {100 * top5_hits / count:.2f}%")
    if PREFILL:
        # the time to first token: prefill, handoff and the prompt's last token through the decode step
        prefill["ttft_s"] = prefill["prefill_s"] + prefill["handoff_s"] + token_times[0]
    model.release_trace()
    steady = sorted(step_times[16:])
    tokens_steady = sorted(token_times[16:])
    rec = dict(
        layers=LAYERS,
        trace=TRACE,
        **prefill,
        top1=100 * top1 / count,
        top5=100 * top5_hits / count,
        scored=count,
        step_ms_median=1e3 * steady[len(steady) // 2],
        step_ms_p10=1e3 * steady[len(steady) // 10],
        token_ms_median=1e3 * tokens_steady[len(tokens_steady) // 2],
    )
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_repeatable(mesh_device):
    layers = int(os.environ.get("RESIDENT_REPEAT_LAYERS", "8"))
    steps = int(os.environ.get("RESIDENT_STEPS", "16"))
    repeats = int(os.environ.get("RESIDENT_REPEATS", "8"))
    gap_ms = float(os.environ.get("RESIDENT_GAP_MS", "30"))
    tokens = torch.load(REFPT, weights_only=False)["reference_tokens"][0]
    ck = Q.Checkpoint()
    n = mesh_device.get_num_devices()
    d = Dims(n, mesh_device.dram_grid_size().x, vocab=ck.config["vocab_size"])
    interval = ck.config["full_attention_interval"]
    w = Q.load(ck, d, layers, interval)
    emb = Q.embedding(ck)
    n_attn = sum((l + 1) % interval == 0 for l in range(layers))
    st = State(d, max(layers - n_attn, 1), max(n_attn, 1), 0, max_pos=steps + 64, zero=True)
    model = ResidentModel(mesh_device, d, w, st, layers, interval, lm_head=True)
    toks = [model.token_state(emb[tokens[pos]].float(), pos) for pos in range(steps)]
    runs = []
    for _ in range(repeats):
        model.reset()
        outs = []
        for tok in toks:
            model.step(tok)
            ttnn.synchronize_device(mesh_device)
            outs.append((model.x(), model.mixer_output(), model.logits()))
            time.sleep(gap_ms / 1e3)
        runs.append(outs)
    diffs = [
        (r, s)
        for r in range(1, repeats)
        for s in range(steps)
        if not all(torch.equal(a, b) for a, b in zip(runs[r][s], runs[0][s]))
    ]
    logger.info(f"{repeats} repeats x {steps} steps, differing (repeat, step): {diffs[:12]}")
    assert not diffs
