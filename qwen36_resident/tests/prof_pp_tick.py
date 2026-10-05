# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Untraced pipeline-parallel prefill ticks (4 chunks, twice) for the device profiler, e.g.
python -m tracy -p -r -m pytest qwen36_resident/tests/prof_pp_tick.py (RESIDENT_LAYERS, RESIDENT_CHUNK)."""
import os

import pytest
import torch

import ttnn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident import weights as Q


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_prof_tick(mesh_device):
    layers = int(os.environ.get("RESIDENT_LAYERS", "64"))
    C = int(os.environ.get("RESIDENT_CHUNK", "512"))
    ck = Q.Checkpoint()
    pp = PPPrefill(mesh_device, ck, layers, ck.config["full_attention_interval"], chunk=C, max_len=4 * C)
    ids = torch.randint(0, ck.config["vocab_size"], (4 * C,))
    for _ in range(2):
        pp.reset()
        pp.run(ids)
        ttnn.synchronize_device(mesh_device)
