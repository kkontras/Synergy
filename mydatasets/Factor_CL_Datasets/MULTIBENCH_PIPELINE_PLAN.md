# MultiBench / FactorCL Pipeline Plan (Analytical)

## Objective
Design a reproducible pipeline that downloads, builds, validates, and reports prepared artifacts for MultiBench affect datasets, while guaranteeing Synergy-compatible dataloader outputs.

## Scope
- In scope:
  - MOSI, MOSEI, UR-Funny, MUStARD
  - Unified preparation CLI
  - Manifest-driven dataset configuration
  - Validation and compatibility gates
  - Config/script wiring fixes
- Out of scope:
  - Model architecture changes
  - Trainer loop changes
  - New task definitions beyond current FactorCL usage

## Constraints
1. Keep existing training/evaluation entrypoints unchanged.
2. Preserve internal modality mapping used by FactorCL (`c`, `f`, `g`).
3. Enforce same structural batch format as other Synergy datasets.
4. Make the pipeline idempotent and safe to rerun.

## Architecture Decisions
1. Single CLI entrypoint:
   - `scripts/experiments/factorcl_affect_prepare.py`
   - Subcommands: `download`, `build`, `validate`, `stats`, `all`
2. Manifest-driven behavior:
   - Dataset-specific behavior configured in one manifest file
   - No dataset-specific hardcoded paths in CLI core
3. Shared normalization layer:
   - Each builder outputs canonical fields before serialization
4. Contract-first gating:
   - Build output is invalid unless contract + compatibility checks pass

## Canonical Data Contract (Summary)
1. Splits: `train`, `valid`, `test`
2. External modalities: `vision`, `audio`, `text`
3. Internal aliases: `c=vision`, `f=audio`, `g=text`
4. Sequence tensor shape target: `[N, T, D]` before batching
5. Batch tensor shape target: `[B, T, D]` for each modality
6. Label shape policy: choose one and enforce globally (`[N,1]` preferred)
7. Required fields: `id`, `label`, modalities, split assignment
8. Prohibited values: NaN/Inf in modalities and labels

## Execution Phases

### Phase 0: Contract + Validator Foundation
Deliverables:
1. Contract doc (see `MULTIBENCH_SYNERGY_CONTRACT.yaml`)
2. Shared validators library (shape, key, dtype, NaN/Inf, split completeness)
3. Compatibility validator against existing Synergy loader batch structure

Exit criteria:
1. A synthetic tiny artifact passes all validators.
2. Validator failure messages clearly indicate root cause and split/sample.

### Phase 1: Manifest + CLI Skeleton
Deliverables:
1. Manifest template (see `MULTIBENCH_MANIFEST_TEMPLATE.yaml`)
2. CLI skeleton with all subcommands and logging
3. Run metadata output (timestamp, git commit, manifest hash)

Exit criteria:
1. `download/build/validate/stats/all` parse and execute with dry-run support.
2. `all` runs strict stage ordering and halts on first gate failure.

### Phase 2: Dataset Builders (Incremental)
Order:
1. MOSEI
2. MOSI
3. UR-Funny
4. MUStARD

Per-dataset builder responsibilities:
1. Read raw source format
2. Map to canonical intermediate schema
3. Apply padding/truncation/mask policy
4. Serialize unified artifact for `get_data.py` consumption

Exit criteria per dataset:
1. Builder succeeds on clean raw data.
2. Artifact passes all validators.
3. One-batch smoke load works for all splits.

### Phase 3: Wiring + Regression Safety
Deliverables:
1. Fix script/config drift (e.g., MOSEI prepare config filename mismatch)
2. Add smoke commands to preparation scripts
3. Add minimal CI-compatible check (if applicable in this repo workflow)

Exit criteria:
1. Existing training command paths still run.
2. No API changes required in `FactorCL_Dataloader` consumers.

## Validation Gates (Must Pass)
1. Split gate:
   - `train/valid/test` present and non-empty
2. Modality gate:
   - required modalities exist for all samples
3. Shape gate:
   - rank and axis ordering conform to contract
4. Numeric gate:
   - no NaN/Inf
5. Label gate:
   - shape/dtype valid for configured task type
6. Feature gate:
   - modality `D` matches config-driven expectations
7. Compatibility gate:
   - one sampled batch structurally matches Synergy dataloader conventions

## Risk Register and Mitigations
1. Risk: dataset-specific preprocessing divergence
   - Mitigation: strict canonical intermediate schema before serialization
2. Risk: label semantics mismatch (`ur-funny` vs internal checks)
   - Mitigation: explicit `task_type` and `label_semantics` fields in manifest
3. Risk: text embedding drift (GloVe/BERT confusion)
   - Mitigation: explicit manifest field + validator on feature dimension
4. Risk: silent corruption from partial downloads
   - Mitigation: expected file checksums/sizes and retry logic

## Milestones
1. M1: Contract + validators + manifest template complete
2. M2: CLI skeleton + MOSEI builder + smoke load
3. M3: MOSI + UR-Funny builders
4. M4: MUStARD builder + full documentation
5. M5: script/config cleanup and final acceptance run

## Definition of Done
1. `factorcl_affect_prepare.py all --dataset <name>` succeeds for each supported dataset.
2. Prepared artifacts load via existing `FactorCL_Dataloader` without codepath changes in trainers.
3. Compatibility gate verifies format parity with existing Synergy dataset batch conventions.
4. Report artifacts produced: sample counts, shapes, label distribution, dropped sample reasons.

