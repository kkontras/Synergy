# Tier1 ESNLI Debug Handoff

## Context
- Cluster: Tier1 Dodrio (Slurm-backed interactive session)
- Project/account: `2026_029`
- Main goal: make ESNLI data prep + smoke pipeline run reliably on Tier1

## What was failing

### 1) Interactive job confusion (Tier2 flags on Tier1)
- Tier2-style flags/account were used initially (`genius/wice`, `lp_biomed_mdv`).
- Correct Tier1 style was confirmed with Dodrio and project `2026_029`.

### 2) `download_esnli_data.sh` failures
- Old HF dataset script loading path broke:
  - `datasets.load_dataset("nlphuji/flickr30k", trust_remote_code=True)` no longer supported.
- Wrong fallback URL downloaded tiny metadata tar (few MB) instead of image archive.
- GitHub URL for e-ViL was stale (`multimodal-ai-lab/e-ViL` returned 404).

### 3) Smoke test runtime issues
- `ESNLI_CodeBook_v2.py` had long silent startup (cold HF/model load).
- API mismatch in `model.get_image_features(...)`:
  - `ValueError: too many values to unpack (expected 2)`.
- CUDA runtime issue in Qwen3-VL image feature path:
  - `RuntimeError: CUDA driver error: invalid argument` at `image_grid_thw.prod(...)`.

## Files changed

### Data download script
- `scripts/experiments/download_esnli_data.sh`
  - Added revision banner.
  - Switched Flickr30k download to HF zip path using `hf_hub_download(..., filename="flickr30k-images.zip")`.
  - Added image count sanity check (~31k jpg).
  - Fixed e-ViL URL:
    - primary: `https://github.com/maximek3/e-ViL/archive/refs/heads/main.zip`
    - fallback: `https://github.com/virginie-do/e-SNLI-VE/archive/refs/heads/master.zip`
  - Added `SKIP_IMAGES=1` support to skip re-downloading Flickr30k.

### New Tier1 smoke orchestrator
- `scripts/experiments/esnli_tier1_smoke.sh` (new)
  - Modes: `MODE=all|cache|train` (default now `all` for true end-to-end run).
  - Cache methods: `CACHE_METHOD=v2|legacy|both` (default now `v2` for minimal smoke runtime).
  - Model sets: `MODEL_SET=basic|extended`
  - Timestamped stage logging.
  - Uses unbuffered Python for live logs.
  - Forwards v2 heartbeat flag.
  - Adds explicit smoke limits:
    - `N_IMAGES=2` (v2), `LEGACY_MAX_SAMPLES=8` (legacy)
    - `SPLITS=validation` by default (single split)
    - `TRAIN_MAX_EPOCH=1`, `TRAIN_MODEL_LIMIT=1`, `TRAIN_BS=2`
    - no hard timeout cutoff; stages run until completion/failure
  - Adds proactive CUDA health probe + auto fallback:
    - probes CUDA with a tiny torch op
    - if probe fails/unavailable, switches smoke to CPU (`SMOKE_DEVICE=cpu`)
    - if probe passes, keeps GPU path (`SMOKE_DEVICE=cuda:0`)
    - dtype is auto-set to match device (`fp16/float16` on GPU, `fp32/float32` on CPU)
  - Forces HF caches under `${DATA_ROOT}/hf_cache` via:
    - `HF_HOME`
    - `HUGGINGFACE_HUB_CACHE`
    - `TRANSFORMERS_CACHE`
    - `HF_DATASETS_CACHE`

### ESNLI cache builders
- `mydatasets/ESNLI/ESNLI_CodeBook_v2.py`
  - Added tmux-friendly logging + heartbeat (`--heartbeat_every`, default 10).
  - Added startup/stage logs.
  - Updated e-ViL source URL to `maximek3/e-ViL`.
  - Added robust unpacker for `get_image_features(...)` return signature drift.
  - Added CUDA workaround:
    - if `get_image_features(..., image_grid_thw_cuda)` throws `CUDA driver error: invalid argument`,
      retry with CPU `image_grid_thw`.
  - Removed noisy exception re-wrap pattern.

- `mydatasets/ESNLI/ESNLI_CodeBook.py`
  - Added same robust `get_image_features(...)` unpacker.
  - Added same CUDA retry for `image_grid_thw`.

## Verified behavior observed
- Flickr30k download now succeeds from HF zip:
  - ~4.39GB zip fetched.
  - Extracted `31783` jpg files.
- e-ViL annotation download now succeeds from `maximek3/e-ViL`.
- `SKIP_IMAGES=1` works and avoids redownloading images.

## Current likely pain point
- First v2 cache run can still feel "stuck" before first batch:
  - heavy model load/cold-start on Tier1 filesystem/network.
  - Now instrumented; should print startup + heartbeat logs once processing starts.

## Recommended commands (fast smoke)

### 1) Ensure data step (without re-downloading images)
```bash
SKIP_IMAGES=1 bash ./scripts/experiments/download_esnli_data.sh "${DATA_ROOT}"
```

### 2) Cache-only smoke (both builders), with frequent heartbeat
```bash
MODE=cache bash scripts/experiments/esnli_tier1_smoke.sh
```

### 3) Explicit cache smoke command (same as defaults, shown for clarity)
```bash
MODE=cache CACHE_METHOD=both SPLITS=validation N_IMAGES=2 LEGACY_MAX_SAMPLES=8 HEARTBEAT_EVERY=1 bash scripts/experiments/esnli_tier1_smoke.sh
```

### 4) Train-only smoke (after v2 cache exists, still fail-fast)
```bash
MODE=train MODEL_SET=basic TRAIN_MODEL_LIMIT=1 TRAIN_MAX_EPOCH=1 bash scripts/experiments/esnli_tier1_smoke.sh
```

## Expected runtime and behavior
- There is no forced cutoff; runtime depends on model/cache warmness and cluster I/O.
- For faster checks, keep `SPLITS=validation`, `N_IMAGES=2`, and `TRAIN_MODEL_LIMIT=1`.
- CUDA behavior:
  - If CUDA is healthy, smoke runs on GPU automatically.
  - If CUDA is unavailable or misconfigured, smoke logs `CUDA_STATUS=cpu_fallback` and continues on CPU instead of crashing.

## If another Codex continues
- First verify node is running the updated checkout:
```bash
pwd
readlink -f scripts/experiments/esnli_tier1_smoke.sh
grep -n "HEARTBEAT_EVERY\\|HF_HOME\\|CACHE_METHOD" scripts/experiments/esnli_tier1_smoke.sh
grep -n "heartbeat_every\\|_unpack_image_feature_outputs\\|maximek3/e-ViL" mydatasets/ESNLI/ESNLI_CodeBook_v2.py
```
- Then tail logs during run:
```bash
tail -f /path/to/job.log
```
- If failures persist, capture full traceback with the exact command/env block used.
