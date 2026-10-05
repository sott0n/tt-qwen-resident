# tt-qwen-resident

Qwen3.6-27B (dense) on a Tenstorrent QB2 (4x Blackhole P150), with an execution model built for Tensix
instead of the op-by-op ttnn graph:

- **Resident decode** — one persistent program per chip runs a whole decode step (64 layers + lm_head),
  tensor-parallel over the 4 chips. Weights stream from DRAM through a ring in L1 at ~490 GB/s per chip;
  the GDN recurrence, attention and the cross-chip all-reduce run on dedicated cores alongside.
- **Pipeline-parallel prefill** — 16 layers per chip, the prompt flows chip to chip in chunks
  (128 to 1024 tokens, picked per prompt), traced; the state is handed to the decode model on device.

## Results (QB2, 64 layers, real weights)

Token accuracy vs the HF reference: top-1 / top-5 = 99.41 / 100 (batch 1, 8 and 32).

| | this repo | tt-metal `models/demos/blackhole/qwen36` |
|---|---|---|
| decode, batch 1 | 12.1 ms/step, ~75 tok/s | ~25 tok/s |
| TTFT 8k / 32k / 128k | 1.06 s / 4.49 s / 32.7 s | 1.24 s / 6.35 s / 34.4 s |
| decode at 32k / 128k / 256k | 69.7 / 54.2 / 42.9 tok/s | 24.3 / 22.9 / — tok/s |
| batch 8 | 374 tok/s (46.7 / user) | 165 tok/s (20.6 / user) |
| batch 32 | 374 tok/s (11.7 / user) | 404 tok/s (12.6 / user) |

Batches above 8 run as sub-batches of 8 (one launch each, one trace).

## Layout

```
qwen36_resident/
  kernels/   resident decode kernels (streamers, GDN heads, attention, hub)
  prefill/   pipeline-parallel prefill (pp_prefill.py) and its kernels
  tests/     model builder (resident_model.py), accuracy tests and benchmarks
patches/     changes to tt-metal's ttnn this code needs (applied to the submodule)
tt-metal/    submodule, pinned to the tt-metal commit this code is built against
```

## Setup

```bash
git clone --recurse-submodules git@github.com:sott0n/tt-qwen-resident.git   # or: git submodule update --init --recursive
cd tt-qwen-resident
./scripts/setup.sh          # applies patches/, builds tt-metal, creates its python env
source scripts/env.sh       # TT_METAL_HOME, PYTHONPATH, python env
```

`pip install ttnn` is not enough: the prefill uses a ttnn op change (`patches/`) and kernel APIs newer than
the published wheels. To use an existing tt-metal build of the pinned commit with the patches applied, set
`TT_METAL_HOME` to it before sourcing `scripts/env.sh`.

Weights: the Hugging Face checkpoint `Qwen/Qwen3.6-27B` in `~/.cache/huggingface/hub`. The accuracy tests read
the reference tokens from tt-metal (`models/tt_transformers/tests/reference_outputs/Qwen3.6-27B.refpt`, or
`RESIDENT_REFPT`).

## Running

```bash
pytest qwen36_resident/tests/test_resident_accuracy.py::test_resident_accuracy        # decode accuracy (RESIDENT_BATCH=8)
RESIDENT_PREFILL=1 pytest qwen36_resident/tests/test_resident_accuracy.py::test_resident_accuracy  # with prefill
RESIDENT_PROMPTS=511,8192,32767 pytest qwen36_resident/tests/bench_ttft.py           # TTFT and decode speed
RESIDENT_BATCHES=1,8,32 pytest qwen36_resident/tests/bench_resident_batch.py          # decode step per batch
RESIDENT_BATCH=8 pytest qwen36_resident/tests/bench_resident_model.py::test_resident_decode_steps  # vs torch, random weights
```

If a run hangs, reset the cards (`tt-smi -r 0,1,2,3`) before the next one.
