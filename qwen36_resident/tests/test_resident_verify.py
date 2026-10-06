# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident verify step (two rows of one user at p and p + 1) on QB2 with random weights, against the torch
reference running the two tokens as consecutive batch-1 steps. Between steps the next one continues from
row 0 (draft rejected: the reference rolls back its second token) or from row 1 (accepted), so the conv
ring and the GDN state slots are exercised both ways."""
import copy
import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident.tests.bench_resident_mlp import HIDDEN, pcc
from qwen36_resident.model import Dims, ResidentModel, State, random_weights, torch_step

OUT = os.environ.get("BENCH_OUT", "/tmp/resident_verify.jsonl")
LAYERS = int(os.environ.get("RESIDENT_LAYERS", "4"))
POS = int(os.environ.get("RESIDENT_POS", "100"))
# per step: 1 = the draft is accepted (continue from row 1), 0 = rejected
ACCEPT = [int(a) for a in os.environ.get("RESIDENT_ACCEPT", "0,1,1,0,1").split(",")]


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_verify_gdn(mesh_device):
    n = mesh_device.get_num_devices()
    d = Dims(n, mesh_device.dram_grid_size().x)
    w = random_weights(d, gdn_copies=LAYERS, attn_copies=0, mlp_copies=2)
    st = State(d, LAYERS, 0, POS, max_pos=POS + 2 * len(ACCEPT) + 64, seed=1)
    ref = copy.deepcopy(st)
    model = ResidentModel(mesh_device, d, w, st, LAYERS, 0, verify=True)
    g = torch.Generator().manual_seed(7)
    ring, slot, p = 0, 0, POS
    recs = []
    for s, accept in enumerate(ACCEPT):
        x0 = torch.randn(2, HIDDEN, generator=g).bfloat16().float()
        model.step(model.token_state(x0, [p, p + 1], ring=ring, slot=slot))
        ttnn.synchronize_device(mesh_device)
        got, o_got = model.x(), model.mixer_output()  # o: [n, 2, gv], the last layer's mixer output
        # control: row 1 run from the state before row 0 (what a missed row-0 dependency would compute)
        o_alone = []
        torch_step(x0[1], d, w, copy.deepcopy(ref), LAYERS, 0, attn_out=o_alone)
        o0, o1 = [], []
        r0 = torch_step(x0[0], d, w, ref, LAYERS, 0, attn_out=o0)
        after0 = copy.deepcopy(ref)
        r1 = torch_step(x0[1], d, w, ref, LAYERS, 0, attn_out=o1)
        last = lambda o: torch.stack([v for l, _, v in o if l == LAYERS - 1])
        rec = dict(
            step=s,
            pos=p,
            accept=accept,
            pcc0=pcc(got[0], r0),
            pcc1=pcc(got[1], r1),
            mixer_pcc0=pcc(o_got[:, 0], last(o0)),
            mixer_pcc1=pcc(o_got[:, 1], last(o1)),
            mixer_pcc1_control=pcc(o_got[:, 1], last(o_alone)),
        )
        logger.info(rec)
        recs.append(rec)
        if accept:
            ring, slot, p = ring + 2, slot ^ 1, p + 2
        else:
            ring, p, ref = ring + 1, p + 1, after0
    with open(OUT, "a") as f:
        f.write(json.dumps(dict(test="verify_gdn", layers=LAYERS, records=recs)) + "\n")
    assert all(min(r["pcc0"], r["pcc1"], r["mixer_pcc0"], r["mixer_pcc1"]) > 0.99 for r in recs)
