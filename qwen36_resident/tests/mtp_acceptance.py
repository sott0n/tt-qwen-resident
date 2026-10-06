# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP draft acceptance on host from a resident decode dump (x and the argmax per position, teacher forced;
test_resident_accuracy with RESIDENT_DUMP writes one).

Convention (vLLM eagle/MTP): MTP entry at position p takes (embed(tok[p+1]), target hidden at p) at
rope position p and predicts tok[p+2]. Draft chain k>0 at position p+k takes (embed(d_k), MTP hidden at
p+k-1). Draft d_k is accepted if it equals the target's greedy token a[p+k] and all earlier ones were.
Usage: python -m qwen36_resident.tests.mtp_acceptance <dump.pt> [K]
"""
import sys

import torch
import torch.nn.functional as F

from qwen36_resident.weights import PREFIX, Checkpoint

torch.set_grad_enabled(False)
ck = Checkpoint()
idx, cfg = ck.index, ck.config


def get(name):
    return ck.get(name).float()


EPS = cfg["rms_norm_eps"]
HD, NQ, NKV = cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
ROT = int(HD * cfg["rope_parameters"]["partial_rotary_factor"])
THETA = cfg["rope_parameters"]["rope_theta"]


def norm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w)


W = {k[len("mtp.") :]: get(k) for k in idx if k.startswith("mtp.")}
L = {k[len("layers.0.") :]: v for k, v in W.items() if k.startswith("layers.0.")}


def rope(t, pos):
    inv = 1.0 / THETA ** (torch.arange(0, ROT, 2).float() / ROT)
    f = pos.float()[:, None] * inv[None]
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()  # [T, ROT]
    r, p = t[..., :ROT], t[..., ROT:]
    h = ROT // 2
    rot = torch.cat([-r[..., h:], r[..., :h]], -1)
    return torch.cat([r * cos[:, None] + rot * sin[:, None], p], -1)


def qkv(h, pos):
    """h [T, H] (after input_layernorm) -> q [T, NQ, HD], gate [T, NQ*HD], k, v [T, NKV, HD]"""
    T = h.shape[0]
    q, gate = (h @ L["self_attn.q_proj.weight"].T).view(T, NQ, 2 * HD).chunk(2, -1)
    q = rope(norm(q, L["self_attn.q_norm.weight"]), pos)
    k = rope(norm((h @ L["self_attn.k_proj.weight"].T).view(T, NKV, HD), L["self_attn.k_norm.weight"]), pos)
    v = (h @ L["self_attn.v_proj.weight"].T).view(T, NKV, HD)
    return q, gate.reshape(T, -1), k, v


def attend(q, K, V, mask):
    """q [T, NQ, HD], K/V [S, NKV, HD], mask [T, S] bool (True = visible)"""
    rep = NQ // NKV
    K, V = K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1)
    s = torch.einsum("tnd,snd->nts", q, K) * HD**-0.5
    s = s.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("nts,snd->tnd", s.softmax(-1), V).reshape(q.shape[0], -1)


def fc_in(e, h):
    return torch.cat([norm(e, W["pre_fc_norm_embedding.weight"]), norm(h, W["pre_fc_norm_hidden.weight"])], -1) @ W[
        "fc.weight"
    ].T


def layer_rest(u, a, gate):
    x = u + (a * torch.sigmoid(gate)) @ L["self_attn.o_proj.weight"].T
    h = norm(x, L["post_attention_layernorm.weight"])
    x = x + (F.silu(h @ L["mlp.gate_proj.weight"].T) * (h @ L["mlp.up_proj.weight"].T)) @ L["mlp.down_proj.weight"].T
    return norm(x, W["norm.weight"])


def check_against_hf(u, pos, out):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    c = Qwen3_5TextConfig(**cfg)
    c._attn_implementation = "sdpa"
    li = c.layer_types.index("full_attention")
    lay = Qwen3_5DecoderLayer(c, li).float().eval()
    lay.load_state_dict(L)
    pe = Qwen3_5TextRotaryEmbedding(c)(u, pos[None])
    y = lay(u[None], position_embeddings=pe, attention_mask=None, position_ids=pos[None])
    y = y[0] if isinstance(y, tuple) else y
    ref = norm(y[0], W["norm.weight"])
    err = (ref - out).abs().max() / ref.abs().max()
    print(f"MTP layer vs HF Qwen3_5DecoderLayer: max rel err {err:.2e}")
    assert err < 1e-3


def main():
    dump = torch.load(sys.argv[1])
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    toks, a = dump["tokens"].long(), torch.tensor(dump["argmax"])
    assert dump["pos"] == list(range(len(dump["pos"])))
    emb = get(f"{PREFIX}embed_tokens.weight")
    head = get("lm_head.weight")
    hn = norm(dump["x"].float(), get(f"{PREFIX}norm.weight")).bfloat16().float()
    agree = ((hn @ head.T).argmax(-1) == a).float().mean()
    print(f"host lm_head(norm(x)) argmax == device argmax: {100 * agree:.2f}%")

    N = len(a) - 1  # entries p = 0..N-1 use tok[p+1] and hidden p
    pos = torch.arange(N)
    u = fc_in(emb[toks[1 : N + 1]], hn[:N])
    q, gate, Kc, Vc = qkv(norm(u, L["input_layernorm.weight"]), pos)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    m = layer_rest(u, attend(q, Kc, Vc, causal), gate)
    check_against_hf(u[:256], pos[:256], m[:256])

    # draft chain from every p; d[k][p] predicts the token at p+k+2, compared with a[p+k+1]
    drafts = [(m @ head.T).argmax(-1)]
    hid = m
    ks, vs = [], []
    for k in range(1, K):
        P = torch.arange(N - k)
        pk = P + k
        uk = fc_in(emb[drafts[-1][: N - k]], hid[: N - k])
        qk, gk, kk, vk = qkv(norm(uk, L["input_layernorm.weight"]), pk)
        ks.append(kk)
        vs.append(vk)
        out = torch.empty(N - k, NQ * HD)
        for p in range(N - k):
            Kp = torch.cat([Kc[: p + 1]] + [kj[p : p + 1] for kj in ks])
            Vp = torch.cat([Vc[: p + 1]] + [vj[p : p + 1] for vj in vs])
            out[p] = attend(qk[p : p + 1], Kp, Vp, torch.ones(1, Kp.shape[0], dtype=torch.bool))[0]
        hid = layer_rest(uk, out, gk)
        drafts.append((hid @ head.T).argmax(-1))

    M = N - K
    hit = torch.stack([drafts[k][:M] == a[k + 1 : k + 1 + M] for k in range(K)])  # [K, M]
    run = hit.long().cumprod(0)
    print(f"scored positions: {M}")
    for k in range(K):
        print(f"draft {k + 1}: accepted {100 * run[k].float().mean():.1f}% (alone {100 * hit[k].float().mean():.1f}%)")
    for kk in range(1, K + 1):
        print(f"K={kk}: tokens per verify step {1 + run[:kk].sum(0).float().mean():.3f}")


main()
