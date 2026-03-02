#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV_PATH="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${CONDA_ENV_PATH}" || true
fi
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="python"; fi

GPU="${1:-0}"

DEFAULT_CONFIG="./configs/FactorCL/Mosi/default_config_mosi_VT.json"
RELEASE_DIR="./configs/FactorCL/Mosi/release/VT"
SYN_DIR="./configs/FactorCL/Mosi/syn/VT"

LR="${LR:-0.0005}"
WD="${WD:-0.001}"

MCR_L_CSV="${MCR_L_CSV:-0.001,0.01,0.1,1}"
MCR_MULTIL_CSV="${MCR_MULTIL_CSV:-0.01,0.1,1}"
MMPARETO_ALPHA_CSV="${MMPARETO_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"
RMASK_LS_CSV="${RMASK_LS_CSV:-0}"
RMASK_LEARNED_L_CSV="${RMASK_LEARNED_L_CSV:-0.001,0.01,0.1,1}"
RMASK_LEARNED_LSPARSE_CSV="${RMASK_LEARNED_LSPARSE_CSV:-0.001,0.01,0.1,1,3,5,10}"
RMASK_RANDOM_L_CSV="${RMASK_RANDOM_L_CSV:-0.001,0.01,0.1,1}"
RMASK_RANDOM_PMIN_CSV="${RMASK_RANDOM_PMIN_CSV:-0.1,0.3,0.5,0.7,0.9}"

IFS=',' read -r -a MCR_LS              <<< "${MCR_L_CSV}"
IFS=',' read -r -a MCR_MULTILS         <<< "${MCR_MULTIL_CSV}"
IFS=',' read -r -a MMPARETO_ALPHAS     <<< "${MMPARETO_ALPHA_CSV}"
IFS=',' read -r -a RMASK_LS            <<< "${RMASK_LS_CSV}"
IFS=',' read -r -a RMASK_LEARNED_LS    <<< "${RMASK_LEARNED_L_CSV}"
IFS=',' read -r -a RMASK_LEARNED_LSPARSES <<< "${RMASK_LEARNED_LSPARSE_CSV}"
IFS=',' read -r -a RMASK_RANDOM_LS     <<< "${RMASK_RANDOM_L_CSV}"
IFS=',' read -r -a RMASK_RANDOM_PMINS  <<< "${RMASK_RANDOM_PMIN_CSV}"

TMPDIR_SWEEP="/tmp/mosi_full_sweep_$$"
mkdir -p "${TMPDIR_SWEEP}"
trap 'rm -rf "${TMPDIR_SWEEP}"' EXIT

run_show() {
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/show.py \
    --default_config "${DEFAULT_CONFIG}" --fold 0 --validate_with accuracy "$@"
}

parse_and_track() {
  local dir="$1" label="$2"; shift 2
  local out
  out="$(run_show "$@" 2>&1)" || true
  "${PYTHON_BIN}" - "${dir}" "${label}" "${out}" <<'PY'
import re, sys, math, statistics, os, json

dir_, label, text = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dir_, exist_ok=True)

ANSI   = re.compile(r"\x1b\[[0-9;]*m")
VAL_RE = re.compile(r"\bAcc_combined:\s*([0-9]+(?:\.[0-9]+)?)")
TST_RE = re.compile(r"\bTest_Acc_combined:\s*([0-9]+(?:\.[0-9]+)?)")
CEU_RE = re.compile(r"\bT_CEU_S:\s*([0-9]+(?:\.[0-9]+)?)")

clean = ANSI.sub("", text)
vals  = [float(m) for m in VAL_RE.findall(clean)]
tests = [float(m) for m in TST_RE.findall(clean)]
ceus  = [float(m) for m in CEU_RE.findall(clean)]

def ms(xs):
    if not xs: return float("nan"), float("nan")
    return statistics.mean(xs), statistics.pstdev(xs)

vm, vs = ms(vals)
tm, ts = ms(tests)
cm, cs = ms(ceus)

if not vals:
    print(f"  {label:<60}  MISSING")
    sys.exit(0)

print(f"  {label:<60}  val={vm:5.2f}±{vs:.2f}  test={tm:5.2f}±{ts:.2f}  ceu={cm:5.2f}±{cs:.2f}")
sys.stdout.flush()

record = {"label": label, "val_mean": vm, "val_std": vs,
          "test_mean": tm, "test_std": ts, "ceu_mean": cm, "ceu_std": cs}
with open(os.path.join(dir_, "results.jsonl"), "a") as f:
    f.write(json.dumps(record) + "\n")
PY
}

print_top3() {
  local dir="$1"
  "${PYTHON_BIN}" - "${dir}" <<'PY'
import sys, json, math, os

dir_ = sys.argv[1]
path = os.path.join(dir_, "results.jsonl")
if not os.path.exists(path):
    print("  no results"); sys.exit(0)

records = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line: records.append(json.loads(line))

ok = [r for r in records if not math.isnan(r["val_mean"])]
if not ok:
    print("  all missing"); sys.exit(0)

def fmt(r):
    ceu = f"{r['ceu_mean']:.2f}±{r['ceu_std']:.2f}" if not math.isnan(r["ceu_mean"]) else "   NA  "
    return (f"    val={r['val_mean']:.2f}±{r['val_std']:.2f}"
            f"  test={r['test_mean']:.2f}±{r['test_std']:.2f}"
            f"  ceu={ceu}"
            f"  [{r['label']}]")

def top3(key, rows):
    return sorted(rows, key=lambda r: r[key], reverse=True)[:3]

print("  Top-3 val_acc:")
for r in top3("val_mean", ok): print(fmt(r))

print("  Top-3 test_acc:")
for r in top3("test_mean", ok): print(fmt(r))

ceu_ok = [r for r in ok if not math.isnan(r["ceu_mean"])]
if ceu_ok:
    print("  Top-3 ceu_synergy:")
    for r in top3("ceu_mean", ceu_ok): print(fmt(r))
else:
    print("  Top-3 ceu_synergy: NA")
PY
}

section() {
  echo ""
  echo "============================================================"
  echo " $1"
  echo "============================================================"
}

# ------------------------------------------------------------------ #
section "MCR"
D="${TMPDIR_SWEEP}/MCR"; mkdir -p "${D}"
for l in "${MCR_LS[@]}"; do
  for multil in "${MCR_MULTILS[@]}"; do
    parse_and_track "${D}" "l=${l} multil=${multil}" \
      --config "${RELEASE_DIR}/MCR.json" --lr "${LR}" --wd "${WD}" --l "${l}" --multil "${multil}"
  done
done
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "MCR_NoiseInput"
D="${TMPDIR_SWEEP}/MCR_NoiseInput"; mkdir -p "${D}"
for l in "${MCR_LS[@]}"; do
  for multil in "${MCR_MULTILS[@]}"; do
    parse_and_track "${D}" "l=${l} multil=${multil}" \
      --config "${RELEASE_DIR}/MCR_NoiseInput.json" --lr "${LR}" --wd "${WD}" --l "${l}" --multil "${multil}"
  done
done
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "MCR_NoiseLatent"
D="${TMPDIR_SWEEP}/MCR_NoiseLatent"; mkdir -p "${D}"
for l in "${MCR_LS[@]}"; do
  for multil in "${MCR_MULTILS[@]}"; do
    parse_and_track "${D}" "l=${l} multil=${multil}" \
      --config "${RELEASE_DIR}/MCR_NoiseLatent.json" --lr "${LR}" --wd "${WD}" --l "${l}" --multil "${multil}"
  done
done
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "MCR_ZeroInput"
D="${TMPDIR_SWEEP}/MCR_ZeroInput"; mkdir -p "${D}"
for l in "${MCR_LS[@]}"; do
  for multil in "${MCR_MULTILS[@]}"; do
    parse_and_track "${D}" "l=${l} multil=${multil}" \
      --config "${RELEASE_DIR}/MCR_ZeroInput.json" --lr "${LR}" --wd "${WD}" --l "${l}" --multil "${multil}"
  done
done
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "MCR_ZeroLatent"
D="${TMPDIR_SWEEP}/MCR_ZeroLatent"; mkdir -p "${D}"
for l in "${MCR_LS[@]}"; do
  for multil in "${MCR_MULTILS[@]}"; do
    parse_and_track "${D}" "l=${l} multil=${multil}" \
      --config "${RELEASE_DIR}/MCR_ZeroLatent.json" --lr "${LR}" --wd "${WD}" --l "${l}" --multil "${multil}"
  done
done
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "MLB"
D="${TMPDIR_SWEEP}/MLB"; mkdir -p "${D}"
parse_and_track "${D}" "lr=${LR} wd=${WD}" \
  --config "${RELEASE_DIR}/MLB.json" --lr "${LR}" --wd "${WD}"
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "MMPareto"
D="${TMPDIR_SWEEP}/MMPareto"; mkdir -p "${D}"
for alpha in "${MMPARETO_ALPHAS[@]}"; do
  parse_and_track "${D}" "alpha=${alpha}" \
    --config "${RELEASE_DIR}/MMPareto.json" --lr "${LR}" --wd "${WD}" --alpha "${alpha}"
done
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "OGM"
D="${TMPDIR_SWEEP}/OGM"; mkdir -p "${D}"
parse_and_track "${D}" "lr=${LR} wd=${WD}" \
  --config "${RELEASE_DIR}/OGM.json" --lr "${LR}" --wd "${WD}"
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "Ensemble"
D="${TMPDIR_SWEEP}/ens"; mkdir -p "${D}"
parse_and_track "${D}" "l=0 lr=${LR} wd=${WD}" \
  --config "${RELEASE_DIR}/ens.json" --lr "${LR}" --wd "${WD}" --l 0
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "Joint Training"
D="${TMPDIR_SWEEP}/joint_training"; mkdir -p "${D}"
parse_and_track "${D}" "lr=${LR} wd=${WD}" \
  --config "${RELEASE_DIR}/joint_training.json" --lr "${LR}" --wd "${WD}"
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "Multi-Loss"
D="${TMPDIR_SWEEP}/multiloss"; mkdir -p "${D}"
parse_and_track "${D}" "lr=${LR} wd=${WD}" \
  --config "${RELEASE_DIR}/multiloss.json" --lr "${LR}" --wd "${WD}"
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "Pre Fine-tuned"
D="${TMPDIR_SWEEP}/pre_finetuned"; mkdir -p "${D}"
parse_and_track "${D}" "lr=${LR} wd=${WD}" \
  --config "${RELEASE_DIR}/pre_finetuned.json" --lr "${LR}" --wd "${WD}"
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "Pre Frozen"
D="${TMPDIR_SWEEP}/pre_frozen"; mkdir -p "${D}"
parse_and_track "${D}" "lr=${LR} wd=${WD}" \
  --config "${RELEASE_DIR}/pre_frozen.json" --lr "${LR}" --wd "${WD}"
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "SynProm RMask (no-pre)"
D="${TMPDIR_SWEEP}/synprom_nopre"; mkdir -p "${D}"
parse_and_track "${D}" "lr=${LR} wd=${WD}" \
  --config "${SYN_DIR}/synprom_RMask_nopre.json" --lr "${LR}" --wd "${WD}"
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "SynProm RMask (base, l sweep)"
D="${TMPDIR_SWEEP}/synprom_base"; mkdir -p "${D}"
for l in "${RMASK_LS[@]}"; do
  parse_and_track "${D}" "l=${l}" \
    --config "${SYN_DIR}/synprom_RMask.json" --lr "${LR}" --wd "${WD}" --l "${l}"
done
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "SynProm Learned (l x lsparse sweep)"
D="${TMPDIR_SWEEP}/synprom_learned"; mkdir -p "${D}"
for l in "${RMASK_LEARNED_LS[@]}"; do
  for lsparse in "${RMASK_LEARNED_LSPARSES[@]}"; do
    parse_and_track "${D}" "l=${l} lsparse=${lsparse}" \
      --config "${SYN_DIR}/synprom_RMask.json" --lr "${LR}" --wd "${WD}" \
      --l "${l}" --perturb learned --perturn learned --perturb_fill ema --perturb_lsparse "${lsparse}"
  done
done
print_top3 "${D}"

# ------------------------------------------------------------------ #
section "SynProm Random (l x pmin sweep)"
D="${TMPDIR_SWEEP}/synprom_random"; mkdir -p "${D}"
for l in "${RMASK_RANDOM_LS[@]}"; do
  for pmin in "${RMASK_RANDOM_PMINS[@]}"; do
    parse_and_track "${D}" "l=${l} pmin=${pmin}" \
      --config "${SYN_DIR}/synprom_RMask.json" --lr "${LR}" --wd "${WD}" \
      --l "${l}" --perturb random --perturn random --perturb_fill ema --perturb_pmin "${pmin}"
  done
done
print_top3 "${D}"

echo ""
echo "Done."
