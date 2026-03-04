# Synergy Project Memory

## Project Overview
Multimodal learning research codebase. Main focus: Synergy (both modalities needed, neither is sufficient alone). Datasets: CREMAD, ScienceQA, FactorCL (MOSI/MOSEI/Mustard/UR-Funny), ESNLI (e-SNLI-VE).

## Key Paths
- **Repo**: `/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy`
- **ESNLI data**: `/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/`
- **ESNLI v2 smoke cache**: `/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_v2_smoke/`
- **ESNLI full cache (v1)**: `/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_qwen3_vl_2b_nocls_vis/`
- **ESNLI checkpoints**: `/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ESNLI/`

## Key Files
- `mydatasets/ESNLI/ESNLI_CodeBook_v2.py` — v2 cache builder (proper CLS special token)
- `mydatasets/ESNLI/ESNLI_CB.py` — dataloader; contains `ESNLI_MemmapDataloader` (line 780, shard-lazy version wins)
- `mydatasets/ESNLI/ESNLI_CB_MEM.py` — memmap-backed version (commented out in `__init__.py`)
- `models/Synergy_Models_Dec.py` — all model classes; ESNLI models at lines 5506/5880/6259
- `scripts/experiments/esnli_smoke_test.sh` — smoke test script
- `configs/ESNLI/default_config_esnli_smoke.json` — smoke test base config

## Detailed Notes
- See `esnli.md` for ESNLI investigation details.
- See `factorcl_experiments.md` for FactorCL experiment structure (MOSI/MOSEI sweep rules, RMask suite, script quirks, TODOs).
