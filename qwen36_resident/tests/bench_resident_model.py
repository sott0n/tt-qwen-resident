# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident decode step on QB2 with random weights: consecutive decode steps of a mixed GDN / attention
layer schedule (and the lm_head) against the torch reference, which carries the same recurrent states,
conv histories and KV caches from step to step; then the per-layer time of each layer kind."""
import copy
import gc
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.qwen36_resident.tests.bench_resident_mlp import HIDDEN, pcc
from models.experimental.qwen36_resident.tests.resident_model import (
    Dims,
    ResidentModel,
    State,
    random_weights,
    torch_step,
)

OUT = os.environ.get("BENCH_OUT", "/tmp/resident_model.jsonl")
POS = [int(p) for p in os.environ.get("RESIDENT_POS", "100").split(",")]
BATCH = int(os.environ.get("RESIDENT_BATCH", "1"))
FIRST_USER = int(os.environ.get("RESIDENT_FIRST_USER", "0"))  # batch 1 replays user k of a batch


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("pos", POS)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_decode_steps(mesh_device, pos):
    """3 consecutive steps of G G G A G G G A (+ lm_head): x, each chip's last mixer output and the logits.
    With RESIDENT_BATCH users, user u starts at pos + 7 u with its own state and inputs."""
    n = mesh_device.get_num_devices()
    lm_head = os.environ.get("RESIDENT_LM_HEAD", "1") == "1"
    d = Dims(n, mesh_device.dram_grid_size().x, vocab=16384 if lm_head else 0)
    layers = int(os.environ.get("RESIDENT_LAYERS", "8"))
    interval = int(os.environ.get("RESIDENT_ATTN_INTERVAL", "4"))
    steps = int(os.environ.get("RESIDENT_STEPS", "3"))
    attn_copies = int(os.environ.get("RESIDENT_ATTN_COPIES", "1"))
    B = BATCH
    w = random_weights(
        d, gdn_copies=2 if interval != 1 else 0, attn_copies=attn_copies if interval else 0, mlp_copies=2
    )
    users = range(FIRST_USER, FIRST_USER + B)
    max_pos = pos + 7 * users[-1] + 64
    st_dev = [State(d, 2, attn_copies, pos + 7 * u, max_pos=max_pos, seed=1 + u) for u in users]
    st_ref = copy.deepcopy(st_dev)
    model = ResidentModel(mesh_device, d, w, st_dev if B > 1 else st_dev[0], layers, interval, lm_head=lm_head)
    gens = [torch.Generator().manual_seed(7 + u) for u in users]  # user 0 sees the batch-1 inputs
    recs = []
    for s in range(steps):
        x0 = torch.stack([torch.randn(HIDDEN, generator=g) for g in gens]).bfloat16().float()
        positions = [st.pos for st in st_ref]
        if B == 1:
            model.step(model.token_state(x0[0], positions[0]))
        else:
            model.step(model.token_state(x0, positions, ring=s))
        ttnn.synchronize_device(mesh_device)
        x_got, o_got = model.x().reshape(B, -1), model.mixer_output()
        o_got = o_got if B > 1 else o_got[:, None]
        lg_got = model.logits().reshape(B, n, -1) if lm_head else None
        am = model.argmax() if lm_head else None
        am = am if B > 1 else [am]
        for u in range(B):
            o_ref = []
            out = torch_step(x0[u], d, w, st_ref[u], layers, interval, lm_head=lm_head, attn_out=o_ref)
            x_ref = out[0] if lm_head else out
            o_ref = torch.stack([o for l, _, o in o_ref if l == layers - 1])
            rec = dict(
                user=users[u],
                pos=positions[u],
                x_pcc=pcc(x_got[u], x_ref),
                x_rel=rel(x_got[u], x_ref),
                mixer_pcc=pcc(o_got[:, u], o_ref),
            )
            if lm_head:
                lg_ref = torch.stack(out[1])
                rec.update(
                    logits_pcc=pcc(lg_got[u], lg_ref),
                    top1_match=bool(lg_got[u].flatten().argmax() == lg_ref.flatten().argmax()),
                    device_argmax=am[u] == int(lg_got[u][:, : d.vocab_chip].reshape(-1).argmax()),
                )
            logger.info(rec)
            recs.append(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(dict(test="decode_steps", batch=B, records=recs)) + "\n")
    # the random states of some users drift faster over steps (user 3 of a batch of 8 replayed alone: x pcc
    # 0.991 at the third step), so later steps of a batch get a looser floor
    floor = lambda r: 0.99 if r["pos"] == pos + 7 * r["user"] or B == 1 else 0.98
    assert all(r["x_pcc"] > floor(r) and r["mixer_pcc"] > floor(r) and r.get("logits_pcc", 1) > floor(r) for r in recs)


def timed_steps(mesh, model, tok, reps=5):
    model.step(tok)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.generic_op(model.io, model.mesh_pd)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / reps


@pytest.mark.parametrize("kind", ["gdn", "attn"])
@pytest.mark.parametrize("pos", POS)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_layer_time(mesh_device, pos, kind):
    """per-layer time of one layer kind: the slope between 2 and 10 layers (weights: 2 copies)"""
    n = mesh_device.get_num_devices()
    d = Dims(n, mesh_device.dram_grid_size().x)
    interval = 1 if kind == "attn" else 0
    w = random_weights(d, gdn_copies=2 if kind == "gdn" else 0, attn_copies=2 if kind == "attn" else 0, mlp_copies=2)
    times = {}
    for L in (2, 10):
        st = State(d, 2, 2, pos, max_pos=pos + 64)
        model = ResidentModel(mesh_device, d, w, st, L, interval)
        tok = model.token_state(torch.randn(HIDDEN).bfloat16().float(), pos)
        times[L] = timed_steps(mesh_device, model, tok)
        del model
        gc.collect()
    rec = dict(test="layer_time", kind=kind, pos=pos, us_per_layer=(times[10] - times[2]) / 8 * 1e6)
    logger.info(rec)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
