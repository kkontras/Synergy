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
  - Modes: `MODE=all|cache|train`
  - Cache methods: `CACHE_METHOD=v2|legacy|both`
  - Model sets: `MODEL_SET=basic|extended`
  - Timestamped stage logging.
  - Uses unbuffered Python for live logs.
  - Forwards v2 heartbeat flag.
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

## Recommended commands (current)

### 1) Ensure data step (without re-downloading images)
```bash
SKIP_IMAGES=1 bash ./scripts/experiments/download_esnli_data.sh "${DATA_ROOT}"
```

### 2) Cache-only smoke (both builders), with frequent heartbeat
```bash
MODE=cache CACHE_METHOD=both GPU=0 HEARTBEAT_EVERY=1 bash scripts/experiments/esnli_tier1_smoke.sh
```

### 3) Train-only smoke (after cache exists)
```bash
MODE=train MODEL_SET=basic GPU=0 bash scripts/experiments/esnli_tier1_smoke.sh
```

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
