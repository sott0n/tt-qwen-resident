# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel prefill (prefill/pp_prefill.py) handing its state to the resident decode.

test_pp_prefill_matches_decode: a prompt prefilled over the chip pipeline, its state exported into the
resident decode and one decode step, against the resident decode run over the prompt token by token:
the decode states (GDN states, conv histories, KV caches) and the next token's logits must agree.
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.tests import qwen36_weights as Q
from qwen36_resident.tests.bench_resident_mlp import pcc
from qwen36_resident.tests.resident_model import Dims, ResidentModel, State
from qwen36_resident import TT_METAL_HOME

REFPT = os.environ.get("RESIDENT_REFPT", os.path.join(TT_METAL_HOME, "models/tt_transformers/tests/reference_outputs/Qwen3.6-27B.refpt"))


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_pp_prefill_matches_decode(mesh_device):
    layers = int(os.environ.get("RESIDENT_LAYERS", "16"))
    T = int(os.environ.get("RESIDENT_PROMPT", "300"))
    chunks = [int(c) for c in os.environ.get("RESIDENT_CHUNK", "256").split(",")]
    C = max(chunks)
    tokens = torch.load(REFPT, weights_only=False)["reference_tokens"][0]
    ck = Q.Checkpoint()
    n = mesh_device.get_num_devices()
    d = Dims(n, mesh_device.dram_grid_size().x, vocab=ck.config["vocab_size"])
    interval = ck.config["full_attention_interval"]
    emb = Q.embedding(ck)
    n_attn = sum((l + 1) % interval == 0 for l in range(layers))
    max_pos = T + 64

    model = ResidentModel(
        mesh_device,
        d,
        Q.load(ck, d, layers, interval),
        State(d, layers - n_attn, n_attn, 0, max_pos, zero=True),
        layers,
        interval,
        lm_head=True,
    )
    pp = PPPrefill(mesh_device, ck, layers, interval, chunk=chunks, max_len=(T + C - 1) // C * C)
    trace = os.environ.get("RESIDENT_TRACE", "1") == "1"
    if trace:
        pp.capture()
        pp.reset()
    t0 = time.perf_counter()
    pp.run(tokens[:T], chunk=chunks[0])
    logger.info(
        f"prefill of {T} tokens, chunk {pp.C} ({'traced' if trace else 'eager'}): {time.perf_counter() - t0:.3f} s"
    )
    t0 = time.perf_counter()
    pp.handoff(model, T)
    logger.info(f"handoff: {time.perf_counter() - t0:.3f} s")
    cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    got = dict(state=cat(model.state_t), hist=cat(model.hist_t), K=cat(model.K_t), V=cat(model.V_t))

    model.reset()
    for pos in range(T):
        model.step(model.token_state(emb[tokens[pos]].float(), pos))
    ttnn.synchronize_device(mesh_device)
    ref = dict(state=cat(model.state_t), hist=cat(model.hist_t), K=cat(model.K_t), V=cat(model.V_t))
    # compare what a step reads: the conv columns of the histories, KV rows of positions < T
    ones = State(d, layers - n_attn, n_attn, T, max_pos, zero=True)
    for c in range(len(ones.hist)):
        for chip in range(n):
            ones.hist[c][chip][:] = 1.0
    for c in range(len(ones.K)):
        for chip in range(n):
            ones.K[c][chip][:T] = 1.0
    masks = dict(hist=model._hist_host(ones) != 0, K=model._kv_host(ones, "K") != 0)
    masks["V"] = masks["K"]
    for k, m in masks.items():
        got[k], ref[k] = got[k][m], ref[k][m]
    state_pcc = {k: pcc(got[k], ref[k]) for k in ref}
    logger.info(f"decode state, prefill vs token by token: pcc {state_pcc}")

    tok = model.token_state(emb[tokens[T]].float(), T)
    model.step(tok)
    ttnn.synchronize_device(mesh_device)
    logits_ref, top_ref = model.logits(), model.argmax()
    pp.handoff(model, T)
    model.step(tok)
    ttnn.synchronize_device(mesh_device)
    logits_pp, top_pp = model.logits(), model.argmax()
    top5 = lambda lg: lg[:, : d.vocab_chip].reshape(-1).topk(5).indices.tolist()
    rec = dict(logits_pcc=pcc(logits_pp, logits_ref), top1=(top_pp, top_ref), top5=(top5(logits_pp), top5(logits_ref)))
    logger.info(rec)
    assert all(v > 0.99 for v in state_pcc.values()), state_pcc
    # a model cut to a few layers has near-tied top logits: its top-1 is not a stable check (the full
    # model's token accuracy is, see test_resident_accuracy with RESIDENT_PREFILL=1)
    assert rec["logits_pcc"] > 0.998 and top_ref in rec["top5"][0]
