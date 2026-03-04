# FactorCL Experiments — MOSI / MOSEI

## Script Locations
- `scripts/experiments/mosi_vt.sh` — MOSI VT (Video+Text), full pipeline with RMask
- `scripts/experiments/mosi_vta.sh` — MOSI VTA (Video+Text+Audio), release methods only
- `scripts/experiments/mosei_vt.sh` — MOSEI VT, release methods + RMask (syn/VT)
- `scripts/experiments/mosei_vta.sh` — MOSEI VTA, release methods only
- `scripts/experiments/mosi_vt_full_pipeline.sh` — orchestrates prep → ceu → methods → table

## Config Locations
- `configs/FactorCL/Mosi/syn/VT/` — MOSI VT syn configs (RMask, unimodals, MCR, etc.)
- `configs/FactorCL/Mosi/release/VTA/` — MOSI VTA release method configs
- `configs/FactorCL/Mosei/syn/VT/` — MOSEI VT syn configs (synprom_RMask.json, synprom_RMask_nopre.json, MCR.json)
- `configs/FactorCL/Mosei/release/VT/` — MOSEI VT release method configs
- `configs/FactorCL/Mosei/release/VTA/` — MOSEI VTA release method configs
- **No syn/VTA configs exist** for either MOSI or MOSEI → RMask not yet available for 3-modality combos

## Per-Method Sweep Rules
Not all methods run the same sweep. The established pattern:
- **MCR**: fixed lr/wd + sweep `l × multil` (0.001,0.01,0.1,1 × 0.01,0.1,1)
- **MMPareto / AGM / OGM / MLB**: fixed lr/wd + sweep `alpha` (0.5,1.0,1.5,2.0,3.0,5.0)
- **DnR**: fixed lr/wd + sweep `alpha × kmepoch` (mosi_vt only)
- **ReconBoost**: fixed lr/wd + sweep `alpha × stages × w1` (mosi_vt only)
- **ens / joint_training / multiloss / pre_finetuned / pre_frozen**: sweep `lr × wd` grid (0.001,0.0001 × 0.001,0.0001)
- **RMask methods**: fixed lr/wd (from best nopre run) + method-specific sweep (see below)

## RMask Suite (mosi_vt and mosei_vt)
All 6 variants, all using fixed `RMASK_LR` / `RMASK_WD` (best from nopre sweep):
1. `synprom_RMask_nopre` — plain run, no extra params
2. `synprom_RMask` (base) — `--l 0`
3. `synprom_RMask_learned` — base cfg + `--perturb learned --perturb_fill ema` + sweep `l × lsparse`
4. `synprom_RMask_random` — base cfg + `--perturb random --perturb_fill ema` + sweep `l × pmin`
5. `synprom_RMask_nopre_learned` — nopre cfg + `--perturb learned --perturb_fill ema` + sweep `l × lsparse`
6. `synprom_RMask_nopre_random` — nopre cfg + `--perturb random --perturb_fill ema` + sweep `l × pmin`

Sweep grids:
- `l` (learned/random): 0.001, 0.01, 0.1, 1
- `lsparse`: 0.001, 0.01, 0.1, 1, 3, 5, 10
- `pmin`: 0.1, 0.3, 0.5, 0.7, 0.9

mosi_vt.sh: variants 5 & 6 (nopre_learned, nopre_random) added as virtual entries in METHODS array with case-dispatch blocks using `SYNPROM_RMASK_NOPRE_CFG`.

## mosi_vt.sh Script Quirks
- `MODE=methods` also triggers `do_ceu` and `do_patch` guards → need `SKIP_CEU=1 SKIP_PATCH=1` to run methods only
- Or bypass entirely with direct Python calls (more reliable)
- `METHOD_FIXED_LR` / `METHOD_FIXED_WD` auto-loaded from `artifacts/reports/mosi_vt_workflow_state.json`

## Deleted Configs
- `configs/FactorCL/Mosei/release/VT/MCR_NoiseInput.json` — deleted
- `configs/FactorCL/Mosei/release/VT/MCR_NoiseLatent.json` — deleted
- `configs/FactorCL/Mosei/release/VT/MCR_ZeroInput.json` — deleted
- `configs/FactorCL/Mosei/release/VT/MCR_ZeroLatent.json` — deleted
These were also removed from the METHODS array in `mosei_vt.sh`.

## TODO
- [ ] **Add RMask for 3-modality (VTA) combos**: create `configs/FactorCL/Mosi/syn/VTA/` and `configs/FactorCL/Mosei/syn/VTA/` with `synprom_RMask.json` and `synprom_RMask_nopre.json` adapted for 3 modalities, then add the full 6-variant RMask suite to `mosi_vta.sh` and `mosei_vta.sh`.
