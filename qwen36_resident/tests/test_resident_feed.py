# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The hub feeds the next step's token state from its greedy token: compared with the host doing it.

After a reference prompt runs through the decode step token by token, GEN tokens are generated greedily
twice: with the host reading the argmax and writing each token state, and with step(None) launched back
to back (no host write or wait between steps; the fed (token, position) word read asynchronously after
each launch). The token sequences must be identical, the hub's pick must equal the host's reduction of
the streamers' records on every host-fed step, and two fed runs must be identical (no host gaps: this is
where a cross-chip semaphore race would show).
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident import TT_METAL_HOME
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State

REFPT = os.environ.get("RESIDENT_REFPT", os.path.join(TT_METAL_HOME, "models/tt_transformers/tests/reference_outputs/Qwen3.6-27B.refpt"))
LAYERS = int(os.environ.get("RESIDENT_LAYERS", "64"))
PROMPT = int(os.environ.get("RESIDENT_PROMPT", "32"))
GEN = int(os.environ.get("RESIDENT_GEN", "128"))


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_feed(mesh_device):
    prompt = torch.load(REFPT, weights_only=False)["reference_tokens"][0][:PROMPT]
    ck = Q.Checkpoint()
    n = mesh_device.get_num_devices()
    d = Dims(n, mesh_device.dram_grid_size().x, vocab=ck.config["vocab_size"])
    interval = ck.config["full_attention_interval"]
    n_attn = sum((l + 1) % interval == 0 for l in range(LAYERS))
    emb = Q.embedding(ck)
    embed = ttnn.from_torch(
        emb.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w = Q.load(ck, d, LAYERS, interval)
    st = State(d, LAYERS - n_attn, n_attn, 0, max_pos=PROMPT + GEN + 64, zero=True)
    model = ResidentModel(mesh_device, d, w, st, LAYERS, interval, lm_head=True, embed=embed)
    del w
    tok = lambda t, pos: model.token_state(emb[int(t)].float(), pos)
    model.step(tok(0, 0))  # compile outside the trace
    model.reset()
    model.capture_trace()

    def run_prompt():
        model.reset()
        for pos in range(PROMPT):
            model.step(tok(prompt[pos], pos))
        ttnn.synchronize_device(mesh_device)

    # host-fed: the host reduces the records and writes every token state
    run_prompt()
    host, picks = [], []
    t0 = time.perf_counter()
    for k in range(GEN):
        t = model.argmax()
        picks.append(model.fed_token() == (t, PROMPT + k))
        host.append(t)
        model.step(tok(t, PROMPT + k))
        ttnn.synchronize_device(mesh_device)
    host_ms = 1e3 * (time.perf_counter() - t0) / GEN

    def run_fed():
        run_prompt()
        view = ttnn.get_device_tensors(model.out_t)[0]
        first = model.fed_token()
        reads = []
        t0 = time.perf_counter()
        for _ in range(GEN - 1):
            model.step(None)
            reads.append(ttnn.from_device(view, blocking=False))
        ttnn.synchronize_device(mesh_device)
        ms = 1e3 * (time.perf_counter() - t0) / (GEN - 1)
        got = [first] + [tuple(int(v) for v in ttnn.to_torch(r).reshape(-1)[:2]) for r in reads]
        return got, ms

    fed, fed_ms = run_fed()
    fed2, _ = run_fed()
    model.release_trace()
    fed_tokens = [t for t, _ in fed]
    logger.info(f"host-fed {host_ms:.2f} ms/step, fed {fed_ms:.2f} ms/step; first tokens {host[:12]}")
    assert all(picks), f"hub pick != host reduction at steps {[k for k, ok in enumerate(picks) if not ok][:8]}"
    assert [p for _, p in fed] == list(range(PROMPT, PROMPT + GEN)), "fed positions"
    diff = [k for k in range(GEN) if fed_tokens[k] != host[k]]
    assert not diff, f"fed tokens differ from host-fed at {diff[:8]}: {fed_tokens[:16]} vs {host[:16]}"
    assert fed == fed2, "two fed runs differ"
