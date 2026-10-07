# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decode with the MTP draft (qwen36_resident.mtp) against plain greedy decode of the
verify-step model, from the same prompt (the reference text's first MTP_PROMPT tokens): the generated tokens
must match, host driven and with the hubs feeding each other. Batch-1 decode rounds differently (its rmsnorm and matmul paths differ from two rows'), so its
greedy tokens are compared for information, with the top-2 logit gap where they first differ. Reports the
acceptance rate, tokens per step and the host-driven step time."""
import gc
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident import TT_METAL_HOME
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State
from qwen36_resident.mtp import SpecDecoder

REFPT = os.environ.get(
    "RESIDENT_REFPT", os.path.join(TT_METAL_HOME, "models/tt_transformers/tests/reference_outputs/Qwen3.6-27B.refpt")
)
OUT = os.environ.get("BENCH_OUT", "/tmp/mtp_spec.jsonl")
LAYERS = int(os.environ.get("RESIDENT_LAYERS", "64"))
PROMPT = int(os.environ.get("MTP_PROMPT", "128"))
NEW = int(os.environ.get("MTP_NEW", "256"))
BLOCK = int(os.environ.get("MTP_BLOCK", "8"))  # tokens per vLLM decode step (streaming check)
DEPTH = int(os.environ.get("MTP_DEPTH", "4"))  # main + draft steps queued ahead of the host
HOST_GAP_S = float(os.environ.get("MTP_HOST_GAP_MS", "40")) / 1e3  # host time between blocks


def plain_greedy(mesh, ck, prompt, n_new, max_pos):
    n = mesh.get_num_devices()
    d = Dims(n, mesh.dram_grid_size().x, vocab=ck.config["vocab_size"])
    interval = ck.config["full_attention_interval"]
    n_attn = sum((l + 1) % interval == 0 for l in range(LAYERS))
    w = Q.load(ck, d, LAYERS, interval)
    model = ResidentModel(mesh, d, w, State(d, LAYERS - n_attn, n_attn, 0, max_pos, zero=True), LAYERS, interval, True)
    del w
    emb = Q.embedding(ck).float()
    model.step(model.token_state(emb[0], 0))
    model.reset()
    model.capture_trace()
    seq = [int(t) for t in prompt]
    times, gaps = [], []
    for p in range(len(prompt) + n_new - 1):
        t0 = time.perf_counter()
        model.step(model.token_state(emb[seq[p]], p))
        a = model.argmax()
        times.append(time.perf_counter() - t0)
        if p + 1 >= len(seq):
            seq.append(a)
            top2 = model.logits()[:, : d.vocab_chip].reshape(-1).topk(2).values
            gaps.append(float(top2[0] - top2[1]))
    model.release_trace()
    del model
    gc.collect()
    steady = sorted(times[len(prompt) :])
    return seq, gaps, 1e3 * steady[len(steady) // 2]


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_mtp_spec(mesh_device):
    tokens = torch.load(REFPT, weights_only=False)["reference_tokens"][0]
    prompt = tokens[:PROMPT].tolist()
    ck = Q.Checkpoint()
    max_pos = PROMPT + NEW + 64
    b1, gaps, plain_ms = plain_greedy(mesh_device, ck, prompt, NEW, max_pos)
    spec = SpecDecoder(mesh_device, ck, LAYERS, max_pos)
    want = spec.greedy(prompt, NEW)
    spec.main.reset()
    spec.draft.reset()
    got, stats = spec.generate(prompt, NEW)
    fed, fed_stats = spec.generate_fed(prompt, NEW)
    first_diff = lambda x, y: next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), NEW)
    same = first_diff(want[PROMPT:], got[PROMPT:])
    fed_same = first_diff(want[PROMPT:], fed[PROMPT:])
    b1_same = first_diff(b1[PROMPT:], want[PROMPT:])
    rec = dict(
        test="mtp_spec",
        layers=LAYERS,
        prompt=PROMPT,
        new=NEW,
        matching=same,
        fed_matching=fed_same,
        batch1_matching=b1_same,
        batch1_gap_there=gaps[b1_same] if b1_same < NEW else None,
        batch1_gap_median=sorted(gaps)[len(gaps) // 2],
        plain_ms=plain_ms,
        **stats,
        **fed_stats,
    )
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert same == NEW and fed_same == NEW


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_mtp_spec_prefilled(mesh_device):
    """after the prefill (handed off to the verify model, no prompt entries in the draft model): hub-fed
    speculative decode against greedy on the verify model from the same prefilled state"""
    tokens = torch.load(REFPT, weights_only=False)["reference_tokens"][0]
    prompt = tokens[:PROMPT].tolist()
    spec = SpecDecoder(mesh_device, Q.Checkpoint(), LAYERS, PROMPT + NEW + 64, prefill=True)
    want = spec.greedy(prompt, NEW, prefill=True)
    got, stats = spec.generate_prefilled(prompt, NEW)
    same = next((i for i, (a, b) in enumerate(zip(want[PROMPT:], got[PROMPT:])) if a != b), NEW)
    # the streaming API as the vLLM block adapter drives it, with a host gap per block like vLLM's step;
    # first a longer request on another prompt, whose state the next request must not read
    spec.start(tokens[PROMPT : 3 * PROMPT].tolist())
    for _ in range(8):
        spec.next_block(BLOCK, DEPTH, PROMPT + NEW + 64)
    t0 = time.perf_counter()
    streamed = [int(spec.start(prompt).argmax())]
    while len(streamed) < NEW:
        streamed += spec.next_block(BLOCK, DEPTH, PROMPT + NEW + 64)
        time.sleep(HOST_GAP_S)
    stream_s = time.perf_counter() - t0
    spec.stop()
    stream_same = next((i for i, (a, b) in enumerate(zip(want[PROMPT:], streamed)) if a != b), NEW)
    rec = dict(test="mtp_spec_prefilled", layers=LAYERS, prompt=PROMPT, new=NEW, matching=same, **stats)
    rec.update(stream_matching=stream_same, stream_tok_s=len(streamed) / stream_s, block=BLOCK, depth=DEPTH)
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert same == NEW and stream_same == NEW
