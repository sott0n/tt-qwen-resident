# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP speculative decode (hubs feeding each other, qwen36_resident.mtp) on chat-template prompts: short
requests of different kinds, and summaries of MTP_LONG tokens of tt-metal's tech reports. Per prompt the
draft acceptance, tokens per step and decode speed over MTP_NEW new tokens. MTP_PREFILL=1 (default): the
prompt goes through the pipeline-parallel prefill (generate_prefilled: TTFT, no prompt entries in the draft
model); 0: through the verify step two tokens at a time (draft entries for every prompt position, slow)."""
import glob
import json
import os

import pytest
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from qwen36_resident import TT_METAL_HOME
from qwen36_resident import weights as Q
from qwen36_resident.mtp import SpecDecoder

OUT = os.environ.get("BENCH_OUT", "/tmp/mtp_spec_bench.jsonl")
LAYERS = int(os.environ.get("RESIDENT_LAYERS", "64"))
NEW = int(os.environ.get("MTP_NEW", "256"))
PREFILL = os.environ.get("MTP_PREFILL", "1") == "1"
LONG = [int(n) for n in os.environ.get("MTP_LONG", "8192,32768").split(",") if n]
CHAT = [
    "Write a Python function that parses an ISO 8601 date string into a datetime, with error handling and tests.",
    "Explain how the attention mechanism in a transformer works, for a high school student.",
    "A train leaves at 3 pm at 60 km/h; a second one leaves the same station at 4 pm at 90 km/h on the same "
    "track. When and where does the second train catch up? Show your steps.",
    "日本の四季について、短いエッセイを書いてください。",
    "Summarize the key ideas of the theory of evolution in five bullet points.",
]


def prompts(tok):
    chat = lambda content: list(
        tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)["input_ids"]
    )
    out = [(f"chat{i}", chat(c)) for i, c in enumerate(CHAT)]
    files = sorted(glob.glob(f"{TT_METAL_HOME}/tech_reports/**/*.md", recursive=True))
    doc = tok("\n\n".join(open(f).read() for f in files)).input_ids
    for n in LONG:
        out.append((f"long{n}", chat(tok.decode(doc[:n]) + "\n\nSummarize the document above in detail.")))
    return out


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_mtp_spec_bench(mesh_device):
    ck = Q.Checkpoint()
    tok = AutoTokenizer.from_pretrained(ck.path)
    cases = prompts(tok)
    max_pos = max(len(p) for _, p in cases) + NEW + 64
    spec = SpecDecoder(mesh_device, ck, LAYERS, max_pos, prefill=PREFILL)
    for name, prompt in cases:
        parts = {}
        seq, stats = spec.generate_prefilled(prompt, NEW, parts) if PREFILL else spec.generate_fed(prompt, NEW)
        rec = dict(test="mtp_spec_bench", case=name, prefill=PREFILL, prompt=len(prompt), new=NEW, **stats)
        rec["ttft_parts"] = {k: round(v, 4) for k, v in parts.items()}
        logger.info(rec)
        logger.info(f"{name}: {tok.decode(seq[len(prompt):])[:300]!r}")
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
