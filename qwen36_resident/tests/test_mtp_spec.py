# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decode with the MTP draft (qwen36_resident.mtp) against plain greedy decode of the
verify-step model, from the same prompt (the reference text's first MTP_PROMPT tokens): the generated tokens
must match. Batch-1 decode rounds differently (its rmsnorm and matmul paths differ from two rows'), so its
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
    first_diff = lambda x, y: next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), NEW)
    same = first_diff(want[PROMPT:], got[PROMPT:])
    b1_same = first_diff(b1[PROMPT:], want[PROMPT:])
    rec = dict(
        test="mtp_spec",
        layers=LAYERS,
        prompt=PROMPT,
        new=NEW,
        matching=same,
        batch1_matching=b1_same,
        batch1_gap_there=gaps[b1_same] if b1_same < NEW else None,
        batch1_gap_median=sorted(gaps)[len(gaps) // 2],
        plain_ms=plain_ms,
        **stats,
    )
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert same == NEW
