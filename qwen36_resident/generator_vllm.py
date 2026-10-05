# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter (vllm-tt-plugin) for the resident Qwen3.6-27B on a 1 x 4 Blackhole mesh.

One user: the prompt but its last token goes through the pipeline-parallel prefill, the state is handed to
the resident decode model, and the last prompt token runs one decode step for the first logits. The model
owns its KV cache, GDN state and conv history, so the plugin's paged KV cache is not used. Sampling runs on
the host from the full logits.
"""
import os

import torch
from loguru import logger
from vllm.transformers_utils.config import uses_mrope

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State
from qwen36_resident.prefill.pp_prefill import PPPrefill

PREFILL_CHUNKS = (128, 256, 512, 1024)


class Qwen36ResidentForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
    }

    def __init__(self, mesh, ck, max_seq_len, mrope):
        self.mesh, self.mrope = mesh, mrope
        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        self.vocab = cfg["vocab_size"]
        n = mesh.get_num_devices()
        self.d = Dims(n, mesh.dram_grid_size().x, vocab=self.vocab)
        self.emb = Q.embedding(ck)
        C = max(PREFILL_CHUNKS)
        max_len = -(-max_seq_len // C) * C
        n_attn = sum((l + 1) % interval == 0 for l in range(layers))
        st = State(self.d, layers - n_attn, n_attn, 0, max_pos=max_len + 64, zero=True)
        w = Q.load(ck, self.d, layers, interval)
        self.model = ResidentModel(mesh, self.d, w, st, layers, interval, lm_head=True)
        del w
        self.max_seq_len = max_seq_len
        # the order bench_ttft validates: decode trace, prefill traces, then the handoff kernel's compile
        self.model.step(self.model.token_state(self.emb[0].float(), 0))
        self.model.reset()
        self.model.capture_trace()
        self.pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
        self.pp.capture()
        self.pp.handoff(self.model, 1)

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, **kwargs):
        assert max_batch_size == 1, "the prefill hands its state to one decode user: serve with max_num_seqs=1"
        # the served revision: a cache filled for a pinned commit need not hold refs/main
        path = Q.checkpoint_dir(os.environ.get("HF_MODEL") or hf_config._name_or_path, getattr(hf_config, "_commit_hash", None))
        ck = Q.Checkpoint(path)
        logger.info(f"Building the resident Qwen3.6 model for up to {max_seq_len} tokens")
        return cls(mesh_device, ck, max_seq_len, mrope=uses_mrope(hf_config))

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=None, max_num_seqs=None, **kwargs):
        return int(max_model_len) * int(max_num_seqs or 1)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return []

    def warmup_model_prefill(self, *args, **kwargs):
        pass

    def warmup_model_decode(self, *args, **kwargs):
        pass

    def _step(self, token, pos):
        assert pos < self.max_seq_len, f"position {pos} is past max_model_len {self.max_seq_len}"
        self.model.step(self.model.token_state(self.emb[int(token)].float(), int(pos)))
        ttnn.synchronize_device(self.mesh)
        # per chip [n, padded vocab slice]: drop the padding, then concatenate in vocab order
        logits = self.model.logits()[:, : self.d.vocab_chip].reshape(-1)[: self.vocab]
        return logits.view(1, 1, -1)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        assert tokens.shape[0] == 1, "one user per prefill"
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        ids = tokens[0, :T]
        self.model.reset()
        self.pp.reset()
        if T > 1:
            self.pp.run(ids[: T - 1])
            self.pp.handoff(self.model, T - 1)
        logits = self._step(ids[T - 1], T - 1)
        # the runner keeps a per-request M-RoPE delta when the config has one; text only, so it is 0
        return (logits, torch.zeros(1, dtype=torch.long)) if self.mrope else logits

    def decode_forward(self, tokens, start_pos, *args, **kwargs):
        return self._step(tokens.reshape(-1)[0], start_pos.reshape(-1)[0])
