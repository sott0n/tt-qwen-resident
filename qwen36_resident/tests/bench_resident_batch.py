# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident decode step time on QB2 per batch size: 64 layers (G G G A), random weights with two copies per
kind, the real vocabulary, traced; each step is the trace replay and the argmax read-back of every user."""
import gc
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.qwen36_resident.tests.bench_resident_mlp import HIDDEN
from models.experimental.qwen36_resident.tests.resident_model import Dims, ResidentModel, State, random_weights

OUT = os.environ.get("BENCH_OUT", "/tmp/resident_batch.jsonl")
BATCHES = [int(b) for b in os.environ.get("RESIDENT_BATCHES", "1,2,4,8").split(",")]
LAYERS = int(os.environ.get("RESIDENT_LAYERS", "64"))
POS = int(os.environ.get("RESIDENT_POS", "500"))
VOCAB = 248320


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_batch_time(mesh_device):
    n = mesh_device.get_num_devices()
    d = Dims(n, mesh_device.dram_grid_size().x, vocab=VOCAB)
    w = random_weights(d, gdn_copies=2, attn_copies=2, mlp_copies=2)
    for B in BATCHES:
        sts = [State(d, 2, 2, POS, max_pos=POS + 128, seed=1 + u, zero=True) for u in range(B)]
        model = ResidentModel(mesh_device, d, w, sts if B > 1 else sts[0], LAYERS, 4, lm_head=True)
        x0 = torch.randn(B, HIDDEN)

        def tok(k):
            if B == 1:
                return model.token_state(x0[0], POS + k)
            return model.token_state(x0, [POS + k] * B, ring=k)

        model.step(tok(0))  # compile outside the trace
        ttnn.synchronize_device(mesh_device)
        model.capture_trace()
        toks = [tok(k) for k in range(1, 21)]
        times = []
        for t in toks:
            t0 = time.perf_counter()
            model.step(t)
            model.argmax()
            times.append(time.perf_counter() - t0)
        step_ms = sorted(times)[len(times) // 2] * 1e3
        rec = dict(
            batch=B,
            head_groups=model.head_groups,
            ring_bytes=model.ring_bytes,
            step_ms=round(step_ms, 2),
            tok_s=round(B * 1e3 / step_ms, 1),
            tok_s_user=round(1e3 / step_ms, 1),
        )
        logger.info(rec)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        model.release_trace()
        del model
        gc.collect()
