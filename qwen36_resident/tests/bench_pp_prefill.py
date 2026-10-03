# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill time of the pipeline-parallel prefill (prefill/pp_prefill.py) for Qwen3.6-27B on QB2: all 64
layers, traced ticks, prompts of RESIDENT_PROMPTS tokens in chunks of RESIDENT_CHUNK."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.qwen36_resident.prefill.pp_prefill import PPPrefill
from models.experimental.qwen36_resident.tests import qwen36_weights as Q

OUT = os.environ.get("BENCH_OUT", "/tmp/resident_prefill.jsonl")


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_pp_prefill_time(mesh_device):
    layers = int(os.environ.get("RESIDENT_LAYERS", "64"))
    chunks = [int(c) for c in os.environ.get("RESIDENT_CHUNK", "1024").split(",")]
    prompts = [int(p) for p in os.environ.get("RESIDENT_PROMPTS", "8192").split(",")]
    ck = Q.Checkpoint()
    interval = ck.config["full_attention_interval"]
    t0 = time.time()
    C = max(chunks)
    pp = PPPrefill(mesh_device, ck, layers, interval, chunk=chunks, max_len=(max(prompts) + C - 1) // C * C)
    logger.info(f"weights placed in {time.time() - t0:.0f} s")
    pp.capture()
    g = torch.Generator().manual_seed(0)
    for T in prompts:
        ids = torch.randint(0, ck.config["vocab_size"], (T,), generator=g)
        for rep in range(2):
            pp.reset()
            ticks = []
            t1 = time.perf_counter()
            pp.run(ids, timings=ticks if rep else None)
            dt = time.perf_counter() - t1
        pp.reset()
        t1 = time.perf_counter()
        pp.run(ids)
        wall = time.perf_counter() - t1
        rec = dict(
            layers=layers,
            chunk=pp.C,
            prompt=T,
            prefill_s=wall,
            tick_ms=sorted(round(1e3 * b, 1) for _, b in ticks)[len(ticks) // 2],
            ticks=len(ticks),
        )
        logger.info(rec)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
