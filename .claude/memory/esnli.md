# ESNLI Investigation Notes

## Task
e-SNLI-VE (visual entailment): image + hypothesis → entailment / neutral / contradiction (3-class).
Dataset: e-ViL / e-SNLI-VE using Flickr30k images + SNLI hypotheses.
Backbone: Qwen3-VL-2B-Instruct.

## Pipeline
1. **Cache builder** (`ESNLI_CodeBook_v2.py`): runs Qwen3-VL over the data, saves per-sample:
   - `input_ids`, `attention_mask`, `position_ids` (3,1,L), `input_embeds` (1,L,2048)
   - `visual_pos_masks` (image token positions), `deepstack_visual_embeds` (K,64,2048)
   - `token_type_ids`, `masks` dict: {image, hint, cls, other}
2. **Dataloader**: `ESNLI_MemmapDataloader` (line 780, `ESNLI_CB.py`) using `ESNLI_ShardedLazyDataset`
   - Reads manifests lazily, LRU caches 2 shards in RAM
   - Extracts `hint_mask` from `masks["hint"]` (or fallback to all-non-image tokens)
   - collate: `memmap_collate()` pads to (B,T) and produces `hint_mask` in batch["data"]
3. **Models** (`Synergy_Models_Dec.py`):
   - `QwenVL_ScienceQA_Cached` (line 5507): combined — passes full pre-cached embeds directly
   - `QwenVL_ScienceQA_Cached_Text` (line 5881): text-only — masks image tokens in attention
   - `QwenVL_ScienceQA_Cached_Image` (line 6262): image-only — zeroes hint tokens in input_embeds, masks attention
   - `QwenVL_ScienceQA_Cached_MCR` (line 7261): combined-only, MCR bias infusion (degenerate w/o perturbation outputs)
   - `QwenVL_ScienceQA_Cached_MMPareto` (line 7635): dual-forward — returns preds/features/losses for {combined, c, g}
   - `QwenVL_ScienceQA_Cached_DnR` (line 8731): subclass of MMPareto, pops ce_loss_combined from losses (avoid double-count)
   - `QwenVL_ScienceQA_Cached_ReconBoost` (line 8753): same as DnR but for ReconBoost handler
   - `QwenVL_ScienceQA_Cached_RMask` (line 8774): subclass of combined, randomly masks image or text during training

## The CLS Bug (fixed in v2)
- **v1 cache**: `<CLS>` was a literal text string in the prompt → tokenized as 3 ordinary tokens `<`, `CLS`, `>` (IDs ~27, ~87716, ~397). Image-only model could exploit these tokens, leading to artificially high accuracy.
- **v2 fix**: `<CLS>` registered as `additional_special_tokens`, gets unique ID ≥ 151669 (original Qwen3-VL-2B vocab size). Cache now stores `masks["cls"]` and `masks["hint"]` correctly.
- Smoke test STEP 2 checks: hint_mask coverage > 80% AND CLS-as-special-token > 80%.

## Smoke Test Status (1000 samples each split)
Cache: `/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_v2_smoke/` — EXISTS (train/val/test built).

Checkpoints in `/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ESNLI/smoke/`:
- `smoke_image_fold0_lr0.0001_wd0.00001_bs2.pth.tar` — **image-only model only** (combined + text-only NEVER trained)
- Run: epoch=0, step=260 (1 epoch over 1000 train samples, bs=2)

### Smoke Test Val/Test Class Distribution (1000 samples)
From early collapse steps (step 20-100), model collapsing to single class reveals:
- Class 0 (entailment): ~36%
- Class 1 (neutral): ~25%
- Class 2 (contradiction): ~39%
→ Slightly imbalanced (not 1/3 each); contradiction over-represented.

### Image-Only Model Results
| Step | Train acc | Val acc | Val F1 (per-class)         |
|------|-----------|---------|----------------------------|
| 20   | 28.6%     | 36.2%   | [0.53, 0.00, 0.00]         |
| 60   | 29.5%     | 38.7%   | [0.00, 0.01, 0.56]         |
| 140  | 30.5%     | 47.4%   | [0.39, 0.00, 0.60]         |
| 160  | 31.7%     | **60.5%** | [0.69, 0.48, 0.65]      |
| 220  | 38.2%     | 64.8%   | [0.70, 0.33, 0.73]         |
| 260  | 42.9%     | **65.4%** | [0.70, 0.54, 0.72]      |

Best val accuracy: **65.4%** at step 260.
Best test accuracy: **65.6%** at step 260.

## The Core Problem: Image-Only Gets ~65%
Random chance on a balanced 3-class problem = 33%. Getting 65%+ for image-only is too high — it means the model can discriminate labels WITHOUT seeing the hypothesis text. This undermines the synergy claim.

### Key Observation: Train acc < Val acc (43% vs 65%)
This is unusual (normally train > val). Possible reasons:
- Val set is slightly easier (class distribution)
- Model's pre-trained visual representations are very strong in eval mode (no dropout)
- LR warm restart at step ~140 caused a sudden jump to 60%+ on val

### Candidate Root Causes Investigated
1. **CLS bug** (v1 issue, fixed in v2): Confirmed fixed. New CLS ID ≥ 151669 and masks["hint"] built correctly.
2. **hint_mask BPE suffix-merge bug** (CONFIRMED, FIXED in ESNLI_CodeBook_v2.py): The prompt template places the hypothesis before 6 trailing `\n` characters. BPE merges the last hypothesis char with those newlines into one token (e.g. `".\n\n\n\n\n\n"` → token `.ĊĊĊĊĊĊ`). The old `_find_subseq` search for `"Hypothesis:{text}"` failed because the last token differed. **Fix**: added candidates with 2–8 trailing `\n` to `label_block` in `build_token_type_ids`. **Caches nano and smoke MUST BE REBUILT** — both contain all-zero `masks["hint"]`.
3. **Position IDs recomputation** (in `QwenVL_ScienceQA_Cached_Image.forward`, lines 6572-6575): Computes new_pos as cumsum of non-hint text positions on shape (3,B,T) tensor. This is likely INCORRECT for Qwen3-VL's 3D RoPE (height/width/depth positions for image tokens), but unclear if this is the cause of high accuracy.
4. **Pre-trained Qwen3-VL bias**: The backbone was pre-trained on visual QA tasks including NLI-like tasks. Its visual representations might inherently correlate with ESNLI labels even without text. The `deepstack_visual_embeds` are deep visual features injected at multiple transformer layers — purely visual, but may encode rich enough semantics.
5. **1e-5 instead of 0**: Image-only model sets hint positions to `1e-5` (not exactly zero). Attention mask DOES zero these out, so this is likely not the cause.

### Fixes Applied (session 2)
- [x] **position_ids fix (image-only)**: Replaced wrong cumsum approach (lines 6573–6575) with `hint_expanded = hint_mask.bool().unsqueeze(0).expand_as(position_ids); new_pos = position_ids.masked_fill(hint_expanded, 0)`. Preserves 3D RoPE spatial coords.
- [x] **position_ids fix (text-only)**: Same fix at lines 6184–6186 using `image_mask`.
- [x] **hint_mask diagnostic**: Added `warnings.warn` if `hint_mask.float().mean() < 0.01` in image-only forward, so masking failures surface immediately.
- [x] **Frozen baseline run**: Added STEP 4 to `esnli_smoke_train_only.sh` using new `configs/ESNLI/smoke_image_frozen.json` (no LoRA, fresh linear head). Saves as `smoke_image_frozen_*.pth.tar`.
- [x] **Summary table updated**: Shows combined / image-only (LoRA) / text-only / img-frozen columns with diagnosis guide.

### Next Steps / Open Questions
- [ ] **REBUILD nano cache**: `rm -rf cache_v2_nano && GPU=N bash scripts/experiments/esnli_nano.sh`
- [ ] **REBUILD smoke cache**: `rm -rf cache_v2_smoke` then rebuild (rerun the cache step of esnli_smoke_test.sh)
- [ ] After rebuild: retrain all 4 models on smoke to get valid accuracy numbers
- [ ] The 65% image-only accuracy on smoke was INVALID — model was seeing full hypothesis (hint_mask all zeros)
- [ ] Check if SAME image appears with different labels across splits (data leakage)

## Config Files
### Smoke (1k samples, local ESAT)
- `configs/ESNLI/default_config_esnli_smoke.json` — smoke default (cache_root=cache_v2_smoke, bs=4)
- `configs/ESNLI/smoke_lora.json` — combined | `smoke_image_lora.json` — image-only
- `configs/ESNLI/smoke_text_lora.json` — text-only | `smoke_image_frozen.json` — frozen baseline

### Full (401k/14k/14k samples, MIB cluster)
- `configs/ESNLI/default_config_esnli_cache_mib.json` — full MIB default
  (cache_root=`/scratch/kkontras/ESNLI/cache_v2_full`, validate_every=12500, n_steps_stop=10, max_epoch=10)
- `configs/ESNLI/full_lora.json` — combined (save: `esnli_full_combined_{}.pth.tar`)
- `configs/ESNLI/full_image_lora.json` — image-only LoRA (save: `esnli_full_image_{}.pth.tar`)
- `configs/ESNLI/full_text_lora.json` — text-only LoRA (save: `esnli_full_text_{}.pth.tar`)
- `configs/ESNLI/full_image_frozen.json` — frozen baseline (save: `esnli_full_image_frozen_{}.pth.tar`)
- `configs/ESNLI/full_mcr.json` — MCR (save: `esnli_full_mcr_{}.pth.tar`)
- `configs/ESNLI/full_mmpareto.json` — MMPareto_Qwen (save: `esnli_full_mmpareto_{}.pth.tar`)
- `configs/ESNLI/full_dnr.json` — DnR (save: `esnli_full_dnr_{}.pth.tar`)
- `configs/ESNLI/full_reconboost.json` — ReconBoost (save: `esnli_full_reconboost_{}.pth.tar`)
- `configs/ESNLI/full_rmask.json` — RMask p=0.15/0.15 (save: `esnli_full_rmask_{}.pth.tar`)

### Old full configs (pre-v2, avoid using)
- `configs/ESNLI/cache_lora.json`, `cache_image_lora.json`, `cache_text_lora.json`, `cache_zero_image.json`

## Experiment Scripts
- `scripts/experiments/esnli_smoke_test.sh` — builds v2 smoke cache (1k/1k/1k) + trains all 4 models
- `scripts/experiments/esnli_smoke_train_only.sh` — trains all 4 models (assumes smoke cache exists)
- `scripts/experiments/esnli_nano.sh` — 5-image pipeline sanity check
- `scripts/experiments/esnli_full_mib.sh` — full pipeline on MIB (cache build + all 4 trainings)
  - MODE=cache / MODE=train / MODE=all (default)
  - Full dataset: train=401,717  val=14,339  test=14,740

## Full Dataset Info
- e-ViL source, Flickr30k images + SNLI hypotheses, 3-class (entailment/neutral/contradiction)
- train: 401,717 | val: 14,339 | test: 14,740
- Cache: /scratch/kkontras/ESNLI/cache_v2_full (shard_size=4096 → ~98 train shards)
- Checkpoints: /scratch/kkontras/checkpoints/synergy/ESNLI/full/
