# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device sampling of the fed token (sampling.hpp) against a host reference of the same integer arithmetic.

After a reference prompt, GEN tokens are generated per parameter set twice: host-fed (the host reads the logits,
draws the token with the reference and writes each token state) and fed (step(None) back to back). Every hub draw
must equal the reference's draw from the same logits, the fed run must repeat the host-fed tokens (the draw
depends only on the logits, the parameters and the position), and some draw must differ from the argmax (the
sampler is not greedy).
"""
import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from qwen36_resident import TT_METAL_HOME
from qwen36_resident import weights as Q
from qwen36_resident.model import TOP_K, Dims, ResidentModel, State

REFPT = os.environ.get("RESIDENT_REFPT", os.path.join(TT_METAL_HOME, "models/tt_transformers/tests/reference_outputs/Qwen3.6-27B.refpt"))
LAYERS = int(os.environ.get("RESIDENT_LAYERS", "64"))
PROMPT = int(os.environ.get("RESIDENT_PROMPT", "32"))
GEN = int(os.environ.get("RESIDENT_GEN", "64"))
CASES = [  # temperature, top_k, top_p, seed
    (1.0, 20, 0.95, 1234),  # the checkpoint's generation config
    (0.7, 32, 1.0, 7),
    (0.3, 5, 0.5, 99),
    (1.5, 8, 0.8, 2**32 - 1),
    (3.0, 32, 1.0, 5),  # flat enough that most draws leave the argmax
]
M32 = 0xFFFFFFFF


def key_of(b):
    return (~b & 0xFFFF) if b & 0x8000 else (b | 0x8000)


def q16(b):
    e = (b >> 7) & 0xFF
    if e == 0:
        return 0
    m = (b & 0x7F) | 0x80
    shift = e - 118
    v = (m << 22 if shift > 22 else m << shift) if shift >= 0 else (0 if shift < -31 else m >> -shift)
    return -v if b & 0x8000 else v


def exp_q16(d):
    t = (d * 94548) >> 16
    n, f = t >> 16, t & 0xFFFF
    p = 87
    for c in (630, 3638, 15743, 45426, 65536):
        p = c + (((p * f) & M32) >> 16)
    return 0 if n <= -31 else p >> -n


def mix(x):
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


def draw(logits, words, pos):
    """sampling::draw over the global top TOP_K of `logits` (bf16 values)"""
    bits = logits.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    keys = torch.where(bits & 0x8000 != 0, ~bits & 0xFFFF, bits | 0x8000)
    # largest keys first, the smaller index first on equal keys
    idx = torch.argsort(keys * len(keys) - torch.arange(len(keys)), descending=True)[:TOP_K]
    lk, li = [int(keys[i]) for i in idx], [int(i) for i in idx]
    inv_temp, top_k, top_p, seed = (int(v) for v in words[:4])
    k = min(top_k, TOP_K)
    if inv_temp == 0 or k <= 1:
        return li[0]
    bf = lambda key: (key & 0x7FFF) if key & 0x8000 else (~key & 0xFFFF)
    l0 = q16(bf(lk[0]))
    w = []
    for i in range(k):
        d = max(((q16(bf(lk[i])) - l0) * inv_temp) >> 16, -(64 << 16))
        w.append(exp_q16(d) if lk[i] else 0)
    limit = (sum(w) * top_p) >> 16
    kept, mass = 0, 0
    while kept < k and (kept == 0 or mass < limit):
        mass += w[kept]
        kept += 1
    target = (mix(seed ^ mix((pos + 0x9E3779B9) & M32)) * mass) >> 32
    for i in range(kept):
        if target < w[i]:
            return li[i]
        target -= w[i]
    return li[kept - 1]


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_resident_sampling(mesh_device):
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
    vocab = ck.config["vocab_size"]
    tok = lambda t, pos: model.token_state(emb[int(t)].float(), pos)
    model.step(tok(0, 0))  # compile outside the trace
    model.reset()
    model.capture_trace()

    def run_prompt():
        model.reset()
        for pos in range(PROMPT):
            model.step(tok(prompt[pos], pos))
        ttnn.synchronize_device(mesh_device)

    not_argmax = 0
    for temperature, top_k, top_p, seed in CASES:
        words = ResidentModel.sampling_words(temperature, top_k, top_p, seed)
        model.set_sampling(temperature, top_k, top_p, seed)
        run_prompt()
        host, bad, case_not_argmax = [], [], 0
        for k in range(GEN):
            logits = model.logits()[:, : d.vocab_chip].reshape(-1)[:vocab]
            ref = draw(logits, words, PROMPT - 1 + k)
            got = model.fed_token()
            if got != (ref, PROMPT + k):
                bad.append((k, got, ref))
            case_not_argmax += ref != int(logits.argmax())
            host.append(ref)
            model.step(tok(ref, PROMPT + k))
            ttnn.synchronize_device(mesh_device)

        run_prompt()
        view = ttnn.get_device_tensors(model.out_t)[0]
        fed = [model.fed_token()]
        reads = []
        for _ in range(GEN - 1):
            model.step(None)
            reads.append(ttnn.from_device(view, blocking=False))
        ttnn.synchronize_device(mesh_device)
        fed += [tuple(int(v) for v in ttnn.to_torch(r).reshape(-1)[:2]) for r in reads]
        not_argmax += case_not_argmax
        logger.info(f"T={temperature} k={top_k} p={top_p}: {case_not_argmax}/{GEN} draws not the argmax; {host[:12]}")
        assert not bad, f"hub draw != reference (step, (token, pos), ref): {bad[:8]}"
        assert fed == [(t, PROMPT + k) for k, t in enumerate(host)], "fed tokens differ from host-fed"
    model.release_trace()
    assert not_argmax > 0, "every draw was the argmax"
