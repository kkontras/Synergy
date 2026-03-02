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

DEFAULT_CONFIG="./configs/FactorCL/Mosi/default_config_mosi_VT.json"
BASE_CFG="./configs/FactorCL/Mosi/syn/VT/synprom_RMask.json"
NOPRE_CFG="./configs/FactorCL/Mosi/syn/VT/synprom_RMask_nopre.json"

LR="${LR:-0.0005}"
WD="${WD:-0.001}"

RMASK_LS_CSV="${RMASK_LS_CSV:-0}"
RMASK_LEARNED_L_CSV="${RMASK_LEARNED_L_CSV:-0.001,0.01,0.1,1}"
RMASK_LEARNED_LSPARSE_CSV="${RMASK_LEARNED_LSPARSE_CSV:-0.001,0.01,0.1,1,3,5,10}"
RMASK_RANDOM_L_CSV="${RMASK_RANDOM_L_CSV:-0.001,0.01,0.1,1}"
RMASK_RANDOM_PMIN_CSV="${RMASK_RANDOM_PMIN_CSV:-0.1,0.3,0.5,0.7,0.9}"

IFS=',' read -r -a RMASK_LS        <<< "${RMASK_LS_CSV}"
IFS=',' read -r -a RMASK_LEARNED_LS       <<< "${RMASK_LEARNED_L_CSV}"
IFS=',' read -r -a RMASK_LEARNED_LSPARSES <<< "${RMASK_LEARNED_LSPARSE_CSV}"
IFS=',' read -r -a RMASK_RANDOM_LS        <<< "${RMASK_RANDOM_L_CSV}"
IFS=',' read -r -a RMASK_RANDOM_PMINS     <<< "${RMASK_RANDOM_PMIN_CSV}"

GPU="${1:-0}"

run_show() {
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/show.py \
    --default_config "${DEFAULT_CONFIG}" --fold 0 --validate_with accuracy "$@"
}

parse_and_track() {
  # Usage: parse_and_track <label> <flags...>
  # Runs show.py, parses val_acc/test_acc/ceu across folds, prints one line,
  # and writes to tmp files for best tracking.
  local label="$1"; shift
  local output
  output="$(run_show "$@" 2>&1)" || true

  "${PYTHON_BIN}" - "${label}" "${output}" <<'PY'
import re, sys, math, statistics, os

label = sys.argv[1]
text  = sys.argv[2]

ANSI   = re.compile(r"\x1b\[[0-9;]*m")
VAL_RE = re.compile(r"\bAcc_combined:\s*([0-9]+(?:\.[0-9]+)?)")
TST_RE = re.compile(r"\bTest_Acc_combined:\s*([0-9]+(?:\.[0-9]+)?)")
CEU_RE = re.compile(r"\bT_CEU_S:\s*([0-9]+(?:\.[0-9]+)?)")

clean = ANSI.sub("", text)
vals  = [float(m) for m in VAL_RE.findall(clean)]
tests = [float(m) for m in TST_RE.findall(clean)]
ceus  = [float(m) for m in CEU_RE.findall(clean)]

if not vals:
    print(f"  {label:<60}  MISSING")
    sys.exit(0)

def ms(xs):
    if not xs: return float("nan"), float("nan")
    return statistics.mean(xs), statistics.pstdev(xs)

vm, vs = ms(vals)
tm, ts = ms(tests)
cm, cs = ms(ceus)

print(
    f"  {label:<60}"
    f"  val={vm:5.2f}±{vs:.2f}"
    f"  test={tm:5.2f}±{ts:.2f}"
    f"  ceu={cm:5.2f}±{cs:.2f}"
)

# Write to tmp tracking files (append one line each: mean std label)
tmp = "/tmp/mosi_rmask_sweep"
os.makedirs(tmp, exist_ok=True)
with open(f"{tmp}/val.txt",  "a") as f: f.write(f"{vm}\t{vs}\t{label}\n")
with open(f"{tmp}/test.txt", "a") as f: f.write(f"{tm}\t{ts}\t{label}\n")
with open(f"{tmp}/ceu.txt",  "a") as f: f.write(f"{cm}\t{cs}\t{label}\n")
PY
}

# Clear tmp tracking files
rm -rf /tmp/mosi_rmask_sweep && mkdir -p /tmp/mosi_rmask_sweep

echo "============================================================"
echo "SynProm RMask sweep  lr=${LR}  wd=${WD}"
echo "============================================================"

echo ""
echo "--- synprom_RMask_nopre ---"
parse_and_track "nopre lr=${LR} wd=${WD}" \
  --config "${NOPRE_CFG}" --lr "${LR}" --wd "${WD}"

echo ""
echo "--- synprom_RMask (base, l sweep) ---"
for l in "${RMASK_LS[@]}"; do
  parse_and_track "base l=${l} lr=${LR} wd=${WD}" \
    --config "${BASE_CFG}" --lr "${LR}" --wd "${WD}" --l "${l}"
done

echo ""
echo "--- synprom_RMask_learned (l x lsparse sweep) ---"
for l in "${RMASK_LEARNED_LS[@]}"; do
  for lsparse in "${RMASK_LEARNED_LSPARSES[@]}"; do
    parse_and_track "learned l=${l} lsparse=${lsparse} lr=${LR} wd=${WD}" \
      --config "${BASE_CFG}" --lr "${LR}" --wd "${WD}" \
      --l "${l}" --perturb learned --perturn learned --perturb_fill ema --perturb_lsparse "${lsparse}"
  done
done

echo ""
echo "--- synprom_RMask_random (l x pmin sweep) ---"
for l in "${RMASK_RANDOM_LS[@]}"; do
  for pmin in "${RMASK_RANDOM_PMINS[@]}"; do
    parse_and_track "random l=${l} pmin=${pmin} lr=${LR} wd=${WD}" \
      --config "${BASE_CFG}" --lr "${LR}" --wd "${WD}" \
      --l "${l}" --perturb random --perturn random --perturb_fill ema --perturb_pmin "${pmin}"
  done
done

echo ""
echo "============================================================"
echo "BESTS"
echo "============================================================"
"${PYTHON_BIN}" - <<'PY'
import os, math

tmp = "/tmp/mosi_rmask_sweep"

def best_of(filename, maximize=True):
    path = os.path.join(tmp, filename)
    if not os.path.exists(path):
        return None
    best_val, best_row = None, None
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            mean = float(parts[0])
            if math.isnan(mean): continue
            if best_val is None or (maximize and mean > best_val) or (not maximize and mean < best_val):
                best_val = mean
                best_row = parts
    return best_row

for criterion, fname in [("Best val acc ", "val.txt"), ("Best test acc", "test.txt"), ("Best CEU-S   ", "ceu.txt")]:
    row = best_of(fname)
    if row is None:
        print(f"{criterion}: no data")
    else:
        mean, std, label = float(row[0]), float(row[1]), row[2]
        print(f"{criterion}: {mean:5.2f}±{std:.2f}  [{label}]")
PY
