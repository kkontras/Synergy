#!/usr/bin/env bash
# mosei_all_methods_ckpt_mode_sweep.sh
#
# Sweep all 3 best-model checkpoints (loss / accuracy / syn_accuracy) for
# every available .pth.tar file under CKPT_DIR, for all MOSEI VT methods.
#
# Usage:
#   bash scripts/experiments/mosei_all_methods_ckpt_mode_sweep.sh [DEVICE]
#
# The optional first argument overrides the default DEVICE (cuda:0).

set -euo pipefail

# Activate the project conda environment
CONDA_ENV="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Always run from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_DIR="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Rmask/MOSEI/VT"
DEFAULT_CFG="configs/FactorCL/Mosei/default_config_mosei_VT_syn.json"
DEVICE="${1:-cuda:0}"
SCRIPT="scripts/analysis/eval_all_ckpt_modes.py"

# Map method prefix -> config file
# NOTE: SynIB_RMask_nopre must be listed before SynIB_RMask to avoid the
# shorter prefix matching nopre checkpoints first.
declare -A METHOD_CFG=(
  ["DnR"]="configs/FactorCL/Mosei/syn/VT/DnR.json"
  ["MCR"]="configs/FactorCL/Mosei/syn/VT/MCR.json"
  ["MMPareto"]="configs/FactorCL/Mosei/syn/VT/MMPareto.json"
  ["ReconBoost"]="configs/FactorCL/Mosei/syn/VT/ReconBoost.json"
  ["SynIB_RMask_nopre"]="configs/FactorCL/Mosei/syn/VT/synprom_RMask_nopre.json"
  ["SynIB_RMask"]="configs/FactorCL/Mosei/syn/VT/synprom_RMask.json"
)

# Process methods in an order that avoids prefix collisions at glob time.
METHODS=("DnR" "MCR" "MMPareto" "ReconBoost" "SynIB_RMask_nopre" "SynIB_RMask")

total_ckpts=0
total_skipped=0
declare -a processed_ckpts=()

echo "======================================================================="
echo "MOSEI all-methods checkpoint-mode sweep"
echo "  CKPT_DIR   : ${CKPT_DIR}"
echo "  DEFAULT_CFG: ${DEFAULT_CFG}"
echo "  DEVICE     : ${DEVICE}"
echo "======================================================================="

for method in "${METHODS[@]}"; do
  cfg="${METHOD_CFG[$method]}"

  # SynIB_RMask (no suffix) should NOT match SynIB_RMask_nopre files.
  # We handle this by iterating nopre first, then excluding nopre files
  # when processing the plain SynIB_RMask prefix.
  pattern="${CKPT_DIR}/${method}_*.pth.tar"

  found_any=0
  for ckpt in ${pattern}; do
    # Glob returned the literal pattern when there are no matches
    [ -f "$ckpt" ] || continue

    # For plain SynIB_RMask, skip files that contain "_nopre" in the name
    # (those belong to SynIB_RMask_nopre and are already processed above).
    if [ "$method" = "SynIB_RMask" ]; then
      basename_ckpt="$(basename "$ckpt")"
      if [[ "$basename_ckpt" == SynIB_RMask_nopre* ]]; then
        continue
      fi
    fi

    # Extract fold number from filename.
    # Handles both "fold0" and "foldfold0" patterns:
    #   grep -oP '(?<=fold)\d' matches the first digit that immediately
    #   follows the substring "fold" — works for both naming conventions.
    fold=$(echo "$ckpt" | grep -oP '(?<=fold)\d' | head -1)
    if [ -z "$fold" ]; then
      echo "[WARN] Could not extract fold from: $ckpt — skipping"
      (( total_skipped++ )) || true
      continue
    fi

    echo ""
    echo "-----------------------------------------------------------------------"
    echo "Method : ${method}"
    echo "Config : ${cfg}"
    echo "Fold   : ${fold}"
    echo "File   : $(basename "$ckpt")"
    echo "-----------------------------------------------------------------------"

    python "${SCRIPT}" \
      --checkpoint "$ckpt" \
      --config "$cfg" \
      --default_config "$DEFAULT_CFG" \
      --fold "$fold" \
      --device "$DEVICE"

    (( total_ckpts++ )) || true
    processed_ckpts+=("${method}|fold${fold}|$(basename "$ckpt")")
    found_any=1
  done

  if [ "$found_any" -eq 0 ]; then
    echo "[INFO] No checkpoints found for method=${method} (pattern: ${pattern})"
  fi
done

echo ""
echo "======================================================================="
echo "Sweep complete."
echo "  Processed : ${total_ckpts} checkpoint(s)"
echo "  Skipped   : ${total_skipped} checkpoint(s)"
echo ""
echo "Summary of processed checkpoints:"
for entry in "${processed_ckpts[@]}"; do
  echo "  ${entry}"
done

# -----------------------------------------------------------------------
# Optional: collect and print a cross-checkpoint summary table.
# Reads post_test_results_{mode} from each processed checkpoint and
# prints best synergy (acc["combined"]) per method × mode.
# -----------------------------------------------------------------------
echo ""
echo "======================================================================="
echo "Cross-checkpoint summary (combined acc per method × ckpt-mode)"
echo "======================================================================="

python - <<'PYEOF'
import os, sys, glob, re, torch
import numpy as np

CKPT_DIR = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Rmask/MOSEI/VT"
METHODS  = ["DnR", "MCR", "MMPareto", "ReconBoost", "SynIB_RMask_nopre", "SynIB_RMask"]
MODES    = ["loss", "accuracy", "syn_accuracy"]

rows = []
for method in METHODS:
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, method + "_*.pth.tar")))
    for ckpt in ckpts:
        bname = os.path.basename(ckpt)
        if method == "SynIB_RMask" and bname.startswith("SynIB_RMask_nopre"):
            continue
        try:
            cp = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [WARN] Could not load {bname}: {e}")
            continue
        fold_m = re.search(r'fold(\d)', bname)
        fold = fold_m.group(1) if fold_m else "?"
        row = {"method": method, "fold": fold, "file": bname}
        for mode in MODES:
            key = f"post_test_results_{mode}"
            res = cp.get(key, {})
            if "acc" in res and "combined" in res["acc"]:
                row[f"{mode}_acc"] = "{:.1f}".format(res["acc"]["combined"] * 100)
            else:
                row[f"{mode}_acc"] = "-"
            # CEU synergy stored back by eval script
            if "ceu_syn" in res and "synergy" in res["ceu_syn"]:
                v = res["ceu_syn"]["synergy"]
                row[f"{mode}_ceu"] = "{:.1f}".format(v) if not np.isnan(v) else "-"
            else:
                row[f"{mode}_ceu"] = "-"
        rows.append(row)

if not rows:
    print("  (no results found yet — run the sweep first)")
    sys.exit(0)

col_w = 10
hdr = "{:<30}{:<6}".format("method", "fold")
for m in MODES:
    hdr += "{:<{w}}".format(m[:4]+"_acc", w=col_w)
    hdr += "{:<{w}}".format(m[:4]+"_ceu", w=col_w)
print(hdr)
print("-" * len(hdr))
for r in rows:
    line = "{:<30}{:<6}".format(r["method"], r["fold"])
    for m in MODES:
        line += "{:<{w}}".format(r.get(f"{m}_acc", "-"), w=col_w)
        line += "{:<{w}}".format(r.get(f"{m}_ceu", "-"), w=col_w)
    print(line)
PYEOF
