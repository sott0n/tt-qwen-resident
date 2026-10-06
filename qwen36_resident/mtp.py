# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decode of Qwen3.6-27B with its MTP layer (one draft token per step), host driven.

The main model runs the verify step (ResidentModel(verify=True)): rows t at p and the draft d at p + 1.
Its greedy tokens a0, a1 accept the draft when a0 == d: the step then commits d and a1, else a0. The
MTP model is one full-attention decoder layer with its own KV cache and the main model's lm_head (folded
with mtp.norm), run as a verify step too: its entry at position q takes x0 = fc([norm_e(embed(tok[q + 1])),
norm_h(h_q)]) for the main model's final hidden h_q (after the final norm) and predicts tok[q + 2]. After
a main step at p it adds the entries p (and p + 1 when the draft was accepted) and its row on the kept
path drafts the next step's d. The fc runs on the host.
"""
import time

import torch

from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State, rmsnorm

MTP = "mtp."


class MtpCheckpoint:
    """the MTP layer as layer 0 of a checkpoint, its norm as the final norm"""

    def __init__(self, ck):
        self.ck, self.config = ck, ck.config

    def get(self, name):
        if name == f"{Q.PREFIX}norm.weight":
            name = f"{MTP}norm.weight"
        return self.ck.get(name)

    def layer(self, i, name):
        assert i == 0
        return self.ck.get(f"{MTP}layers.0.{name}")


def mtp_weights(ck, d):
    """ResidentModel weights of the MTP decoder layer (an attention layer) and its lm_head"""
    m = MtpCheckpoint(ck)
    w = dict(gdn=[], attn=[Q.attn_layer(m, 0, d)], mlp=[Q.mlp(m, 0, d)], out=[Q.out_proj(m, 0, d, True)])
    w["head"], w["vocab_per_chip"] = Q.lm_head(m, d)
    return w


class SpecDecoder:
    def __init__(self, mesh, ck, layers, max_pos, trace=True):
        n = mesh.get_num_devices()
        self.d = d = Dims(n, mesh.dram_grid_size().x, vocab=ck.config["vocab_size"])
        interval = ck.config["full_attention_interval"]
        n_attn = sum((l + 1) % interval == 0 for l in range(layers))
        w = Q.load(ck, d, layers, interval)
        self.main = ResidentModel(
            mesh, d, w, State(d, layers - n_attn, n_attn, 0, max_pos, zero=True), layers, interval, True, verify=True
        )
        del w
        self.draft = ResidentModel(
            mesh, d, mtp_weights(ck, d), State(d, 0, 1, 0, max_pos, zero=True), 1, 1, True, verify=True
        )
        self.mesh = mesh
        self.emb = Q.embedding(ck).float()
        self.norm_w = 1 + ck.get(f"{Q.PREFIX}norm.weight").float()
        self.norm_e = 1 + ck.get(f"{MTP}pre_fc_norm_embedding.weight").float()
        self.norm_h = 1 + ck.get(f"{MTP}pre_fc_norm_hidden.weight").float()
        self.fc_t = ck.get(f"{MTP}fc.weight").float().T.contiguous()  # [2 H, H]
        if trace:
            for m in (self.main, self.draft):
                m.step(m.token_state(torch.zeros(2, self.emb.shape[1]), [0, 1], ring=0, slot=0))
                m.reset()
                m.capture_trace()

    def _main(self, toks, p, ring, slot):
        m = self.main
        m.step(m.token_state(self.emb[toks], [p, p + 1], ring=ring, slot=slot))
        a = m.argmax()
        h = rmsnorm(m.x()) * self.norm_w  # the final norm, as the lm_head sees it
        return a, h.bfloat16().float()

    draft_t = []

    def _draft(self, toks, h, q, ring, slot):
        """MTP entries at q, q + 1 for (toks[i], h[i]); its argmax per row"""
        t0 = time.perf_counter()
        x0 = torch.cat([rmsnorm(self.emb[toks]) * self.norm_e, rmsnorm(h) * self.norm_h], dim=1) @ self.fc_t
        m = self.draft
        tok = m.token_state(x0.bfloat16().float(), [q, q + 1], ring=ring, slot=slot)
        t1 = time.perf_counter()
        m.step(tok)
        a = m.argmax()
        self.draft_t.append((t1 - t0, time.perf_counter() - t1))
        return a

    def generate(self, prompt, n_new):
        """greedy: the prompt's tokens, then n_new generated ones; returns (tokens, stats)"""
        seq = [int(t) for t in prompt]
        P = len(seq)
        assert P >= 2
        p = ring = slot = 0
        # the prompt in pairs; the pair holding position P - 2 continues from that row, so the main model has
        # seen positions < P - 1 and the draft row on the kept path predicts position P
        while p <= P - 2:
            accept = int(p + 1 <= P - 2)
            a, h = self._main([seq[p], seq[p + 1]], p, ring, slot)
            nxt = [seq[p + 1], seq[p + 2] if p + 2 < P else seq[p + 1]]
            d = self._draft(nxt, h, p, ring, slot)[accept]
            p, ring, slot = (p + 2, ring + 2, slot ^ 1) if accept else (p + 1, ring + 1, slot)
        assert p == P - 1
        steps = accepted = 0
        t0 = time.perf_counter()
        times, main_t = [], []
        self.draft_t = []
        while len(seq) < P + n_new:
            ts = time.perf_counter()
            a, h = self._main([seq[p], d], p, ring, slot)
            main_t.append(time.perf_counter() - ts)
            accept = int(a[0] == d)
            new = [d, a[1]] if accept else [a[0]]
            seq += new
            d = self._draft([a[0], a[1]], h, p, ring, slot)[accept]
            p, ring, slot = (p + 2, ring + 2, slot ^ 1) if accept else (p + 1, ring + 1, slot)
            steps += 1
            accepted += accept
            times.append((time.perf_counter() - ts, len(new)))
        total = time.perf_counter() - t0
        out = seq[: P + n_new]
        stats = dict(
            steps=steps,
            accept_rate=accepted / steps,
            tokens_per_step=(len(seq) - P) / steps,
            ms_per_step=1e3 * sorted(t for t, _ in times)[len(times) // 2],
            main_ms=1e3 * sorted(main_t)[len(main_t) // 2],
            draft_host_ms=1e3 * sorted(a for a, _ in self.draft_t)[len(self.draft_t) // 2],
            draft_device_ms=1e3 * sorted(b for _, b in self.draft_t)[len(self.draft_t) // 2],
            tok_s=(len(seq) - P) / total,
        )
        return out, stats

    def greedy(self, prompt, n_new):
        """plain greedy on the main model's verify step (row 1 repeats row 0's token and is never kept): the
        tokens speculative decode must reproduce"""
        seq = [int(t) for t in prompt]
        p = ring = slot = 0
        while len(seq) < len(prompt) + n_new:
            a, _ = self._main([seq[p], seq[p]], p, ring, slot)
            if p + 1 >= len(seq):
                seq.append(a[0])
            p, ring = p + 1, ring + 1
        return seq
