# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident verify step (two rows of one user at p and p + 1) on QB2 with random weights, against the torch
reference running the two tokens as consecutive batch-1 steps. Between steps the next one continues from
row 0 (draft rejected: the reference rolls back its second token) or from row 1 (accepted), so the conv
ring, the GDN state slots and the shared KV cache are exercised both ways. The default positions put a
step at the last row of a KV tile (rows p, p + 1 in different tiles), below and above `banks` tiles."""
import copy
import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident.tests.bench_resident_mlp import HIDDEN, pcc
from qwen36_resident.model import Dims, ResidentModel, State, is_attn, random_weights, torch_step

OUT = os.environ.get("BENCH_OUT", "/tmp/resident_verify.jsonl")
LAYERS = int(os.environ.get("RESIDENT_LAYERS", "8"))
POS = [int(p) for p in os.environ.get("RESIDENT_POS", "94,542").split(",")]
# per step: 1 = the draft is accepted (continue from row 1), 0 = rejected
ACCEPT = [int(a) for a in os.environ.get("RESIDENT_ACCEPT", "0,1,1,0,1").split(",")]


@pytest.mark.parametrize("interval", [0, 4], ids=["gdn", "mixed"])
@pytest.mark.parametrize("pos", POS)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_verify(mesh_device, pos, interval):
    n = mesh_device.get_num_devices()
    d = Dims(n, mesh_device.dram_grid_size().x)
    n_attn = sum(is_attn(l, interval) for l in range(LAYERS))
    w = random_weights(d, gdn_copies=LAYERS - n_attn, attn_copies=n_attn, mlp_copies=2)
    st = State(d, LAYERS - n_attn, n_attn, pos, max_pos=pos + 2 * len(ACCEPT) + 64, seed=1)
    ref = copy.deepcopy(st)
    model = ResidentModel(mesh_device, d, w, st, LAYERS, interval, verify=True)
    g = torch.Generator().manual_seed(7)
    ring, slot, p = 0, 0, pos
    recs = []
    for s, accept in enumerate(ACCEPT):
        x0 = torch.randn(2, HIDDEN, generator=g).bfloat16().float()
        model.step(model.token_state(x0, [p, p + 1], ring=ring, slot=slot))
        ttnn.synchronize_device(mesh_device)
        got, o_got = model.x(), model.mixer_output()  # o: [n, 2, gv], the last layer's mixer output
        # control: row 1 run from the state before row 0 (what a missed row-0 dependency would compute)
        o_alone = []
        alone = copy.deepcopy(ref)
        alone.pos += 1
        torch_step(x0[1], d, w, alone, LAYERS, interval, attn_out=o_alone)
        o0, o1 = [], []
        r0 = torch_step(x0[0], d, w, ref, LAYERS, interval, attn_out=o0)
        after0 = copy.deepcopy(ref)
        r1 = torch_step(x0[1], d, w, ref, LAYERS, interval, attn_out=o1)
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
        f.write(json.dumps(dict(test="verify", layers=LAYERS, interval=interval, records=recs)) + "\n")
    # random GDN states drift from the reference over steps (batch-1 decode of 8 GDN layers: mixer PCC 0.955 at
    # the fifth step), so the floor is that of later batch steps; row 1 run without row 0 stays well below
    assert all(min(r["pcc0"], r["pcc1"], r["mixer_pcc0"], r["mixer_pcc1"]) > 0.98 for r in recs)
    assert all(r["mixer_pcc1_control"] < 0.9 for r in recs)
