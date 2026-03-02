#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
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
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

DEFAULT_CONFIG="${DEFAULT_CONFIG:-./configs/FactorCL/Mustard/default_config_mustard_VT.json}"
RELEASE_DIR="${RELEASE_DIR:-./configs/FactorCL/Mustard/release/VT}"
OUT_DIR="${OUT_DIR:-./artifacts/reports}"
mkdir -p "${OUT_DIR}"

# Keep defaults aligned with scripts/experiments/mustard_vt.sh
METHOD_RUN_MODE="${METHOD_RUN_MODE:-all}"                    # all | single
METHOD_TARGET="${METHOD_TARGET:-MMPareto.json}"              # used when METHOD_RUN_MODE=single
METHOD_FIXED_LR="${METHOD_FIXED_LR:-0.0005}"
METHOD_FIXED_WD="${METHOD_FIXED_WD:-0.001}"
SYNPROM_RMASK_SWEEP_LRWD="${SYNPROM_RMASK_SWEEP_LRWD:-0}"    # 1 => sweep METHOD_LRS/WDS for synprom_RMask
SYNPROM_RMASK_LS_CSV="${SYNPROM_RMASK_LS_CSV:-0.001,0.01,0.1,1}"
METHOD_LRS_CSV="${METHOD_LRS_CSV:-0.001,0.0001}"
METHOD_WDS_CSV="${METHOD_WDS_CSV:-0.001,0.0001}"
SELECT_METHODS_CSV="${SELECT_METHODS_CSV:-}"                 # optional comma-separated subset

export PYTHON_BIN DEFAULT_CONFIG RELEASE_DIR OUT_DIR
export METHOD_RUN_MODE METHOD_TARGET METHOD_FIXED_LR METHOD_FIXED_WD
export SYNPROM_RMASK_SWEEP_LRWD SYNPROM_RMASK_LS_CSV METHOD_LRS_CSV METHOD_WDS_CSV
export SELECT_METHODS_CSV

"${PYTHON_BIN}" - <<'PY'
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(os.getcwd())
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")
DEFAULT_CONFIG = os.environ["DEFAULT_CONFIG"]
RELEASE_DIR = os.environ["RELEASE_DIR"]
OUT_DIR = Path(os.environ["OUT_DIR"])
METHOD_RUN_MODE = os.environ.get("METHOD_RUN_MODE", "all")
METHOD_TARGET = os.environ.get("METHOD_TARGET", "MMPareto.json")
METHOD_FIXED_LR = os.environ.get("METHOD_FIXED_LR", "0.0005")
METHOD_FIXED_WD = os.environ.get("METHOD_FIXED_WD", "0.001")
SYNPROM_RMASK_SWEEP_LRWD = os.environ.get("SYNPROM_RMASK_SWEEP_LRWD", "0")
SYNPROM_RMASK_LS = [x.strip() for x in os.environ.get("SYNPROM_RMASK_LS_CSV", "0.001,0.01,0.1,1").split(",") if x.strip()]
METHOD_LRS = [x.strip() for x in os.environ.get("METHOD_LRS_CSV", "0.001,0.0001").split(",") if x.strip()]
METHOD_WDS = [x.strip() for x in os.environ.get("METHOD_WDS_CSV", "0.001,0.0001").split(",") if x.strip()]
SELECT_METHODS_CSV = os.environ.get("SELECT_METHODS_CSV", "").strip()

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
VAL_ACC_RE = re.compile(r"\bAcc_combined:\s*([0-9]+(?:\.[0-9]+)?)\b")
TEST_ACC_RE = re.compile(r"\bTest_Acc_combined:\s*([0-9]+(?:\.[0-9]+)?)\b")
TEST_CEU_RE = re.compile(r"\bT_CEU_S:\s*([0-9]+(?:\.[0-9]+)?)\b")
MISSING_RE = re.compile(r"We could not load ")


def mean_std(vals):
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.pstdev(vals))


def latex_pair(ceu_mean_pct, ceu_std_pct, acc_mean_pct, acc_std_pct):
    if any(math.isnan(v) for v in [ceu_mean_pct, ceu_std_pct, acc_mean_pct, acc_std_pct]):
        return "NA"
    return (
        f"{ceu_mean_pct:.1f}" + r"{\tiny$\pm$}" + f"{ceu_std_pct:.1f}" +
        "/" +
        f"{acc_mean_pct:.1f}" + r"{\tiny$\pm$}" + f"{acc_std_pct:.1f}"
    )


def parse_show_output(text):
    clean = ANSI_RE.sub("", text)
    val_accs = []
    test_accs = []
    test_ceus = []
    for line in clean.splitlines():
        # Per-fold lines contain uppercase Acc_combined / Test_Acc_combined.
        if "Acc_combined:" in line:
            m = VAL_ACC_RE.search(line)
            if m:
                val_accs.append(float(m.group(1)))
        if "Test_Acc_combined:" in line:
            m = TEST_ACC_RE.search(line)
            if m:
                test_accs.append(float(m.group(1)))
        if "T_CEU_S:" in line:
            m = TEST_CEU_RE.search(line)
            if m:
                test_ceus.append(float(m.group(1)))

    if not val_accs and MISSING_RE.search(clean):
        return None
    if not val_accs:
        return {"error": "Could not parse validation accuracy from show.py output", "raw": clean}

    val_mean, val_std = mean_std(val_accs)
    test_acc_mean, test_acc_std = mean_std(test_accs) if test_accs else (math.nan, math.nan)
    test_ceu_mean, test_ceu_std = mean_std(test_ceus) if test_ceus else (math.nan, math.nan)
    return {
        "val_acc_fold_pct": val_accs,
        "test_acc_fold_pct": test_accs,
        "test_ceu_synergy_fold_raw": test_ceus,  # raw in [0,1] from show.py
        "val_acc_mean_pct": val_mean,
        "val_acc_std_pct": val_std,
        "test_acc_mean_pct": test_acc_mean,
        "test_acc_std_pct": test_acc_std,
        "test_ceu_synergy_mean_raw": test_ceu_mean,
        "test_ceu_synergy_std_raw": test_ceu_std,
        "test_ceu_synergy_mean_pct": (100.0 * test_ceu_mean) if not math.isnan(test_ceu_mean) else math.nan,
        "test_ceu_synergy_std_pct": (100.0 * test_ceu_std) if not math.isnan(test_ceu_std) else math.nan,
        "raw_output": clean,
    }


def run_show(config_path, extra_flags):
    cmd = [
        PYTHON_BIN,
        "scripts/entrypoints/show.py",
        "--config", config_path,
        "--default_config", DEFAULT_CONFIG,
        "--fold", "0",  # show.py interprets non-None fold as "iterate all 3 folds"
        "--validate_with", "accuracy",
    ] + extra_flags
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    parsed = parse_show_output(proc.stdout)
    return proc.returncode, parsed, proc.stdout, cmd


def fmt_flags(flags):
    pairs = []
    i = 0
    while i < len(flags):
        k = flags[i]
        if i + 1 < len(flags) and flags[i + 1].startswith("--") is False:
            pairs.append(f"{k}={flags[i+1]}")
            i += 2
        else:
            pairs.append(k)
            i += 1
    return " ".join(pairs)


def candidate_specs():
    rel = lambda name: str(Path(RELEASE_DIR) / name)
    methods = [
        "MMPareto.json",
        "DnR.json",
        "MCR.json",
        "ReconBoost.json",
        "synprom_RMask.json",
        "synprom_RMask_nonpre.json",
        "synprom_RMask_learned.json",
        "synprom_RMask_random.json",
    ]

    if SELECT_METHODS_CSV:
        wanted = {m.strip() if m.strip().endswith(".json") else f"{m.strip()}.json"
                  for m in SELECT_METHODS_CSV.split(",") if m.strip()}
        methods = [m for m in methods if m in wanted]

    if METHOD_RUN_MODE == "single":
        target = METHOD_TARGET if METHOD_TARGET.endswith(".json") else f"{METHOD_TARGET}.json"
        methods = [m for m in methods if m == target]

    specs = []
    for m in methods:
        if m == "DnR.json":
            for alpha in ["0.5", "1.0", "1.5", "2.0", "3.0", "5.0"]:
                for kmpe in ["1", "3", "5", "10"]:
                    flags = ["--lr", METHOD_FIXED_LR, "--wd", METHOD_FIXED_WD, "--alpha", alpha, "--kmepoch", kmpe]
                    specs.append((m, rel(m), flags))
        elif m == "MCR.json":
            for l in ["0.001", "0.01", "0.1", "1"]:
                for multil in ["0.01", "0.1", "1"]:
                    flags = ["--lr", METHOD_FIXED_LR, "--wd", METHOD_FIXED_WD, "--l", l, "--multil", multil]
                    specs.append((m, rel(m), flags))
        elif m == "MMPareto.json":
            for alpha in ["0.5", "1.0", "1.5", "2.0", "3.0", "5.0"]:
                flags = ["--lr", METHOD_FIXED_LR, "--wd", METHOD_FIXED_WD, "--alpha", alpha]
                specs.append((m, rel(m), flags))
        elif m == "ReconBoost.json":
            for alpha in ["0.5", "1.0", "1.5", "2.0", "3.0", "5.0"]:
                for recon_stages in ["1", "4", "10"]:
                    for w1 in ["1", "3", "5", "10"]:
                        flags = [
                            "--lr", METHOD_FIXED_LR, "--wd", METHOD_FIXED_WD,
                            "--alpha", alpha,
                            "--recon_weight1", w1, "--recon_weight2", "1",
                            "--recon_epochstages", recon_stages,
                            "--recon_ensemblestages", recon_stages,
                        ]
                        specs.append((m, rel(m), flags))
        elif m == "synprom_RMask_learned.json":
            for l in ["0.001", "0.01", "0.1", "1"]:
                for lsparse in ["0.001", "0.01", "0.1", "1", "3", "5", "10"]:
                    flags = [
                        "--lr", METHOD_FIXED_LR, "--wd", METHOD_FIXED_WD,
                        "--l", l,
                        "--perturb", "learned", "--perturn", "learned",
                        "--perturb_fill", "ema", "--perturb_lsparse", lsparse,
                    ]
                    specs.append((m, rel("synprom_RMask.json"), flags))
        elif m == "synprom_RMask_random.json":
            for l in ["0.001", "0.01", "0.1", "1"]:
                for pmin in ["0.1", "0.3", "0.5", "0.7", "0.9"]:
                    flags = [
                        "--lr", METHOD_FIXED_LR, "--wd", METHOD_FIXED_WD,
                        "--l", l,
                        "--perturb", "random", "--perturn", "random",
                        "--perturb_fill", "ema", "--perturb_pmin", pmin,
                    ]
                    specs.append((m, rel("synprom_RMask.json"), flags))
        elif m == "synprom_RMask.json":
            # Vanilla fusion (RMask architecture, no IB): fixed at l=0, no perturb
            flags = ["--lr", METHOD_FIXED_LR, "--wd", METHOD_FIXED_WD, "--l", "0"]
            specs.append((m, rel(m), flags))
        elif m == "synprom_RMask_nonpre.json":
            # Vanilla fusion (no pretrained encoders): trained without --l, use fixed lr/wd
            flags = ["--lr", METHOD_FIXED_LR, "--wd", METHOD_FIXED_WD]
            specs.append((m, rel(m), flags))
        else:
            for lr in METHOD_LRS:
                for wd in METHOD_WDS:
                    flags = ["--lr", lr, "--wd", wd]
                    specs.append((m, rel(m), flags))
    return specs


def method_order():
    return [
        "MMPareto.json",
        "DnR.json",
        "MCR.json",
        "ReconBoost.json",
        "synprom_RMask.json",
        "synprom_RMask_nonpre.json",
        "synprom_RMask_learned.json",
        "synprom_RMask_random.json",
    ]


all_specs = candidate_specs()
if not all_specs:
    print("No methods selected.", file=sys.stderr)
    sys.exit(1)

grouped = {}
for method_name, config_path, flags in all_specs:
    grouped.setdefault(method_name, []).append((config_path, flags))

selected_methods = [m for m in method_order() if m in grouped]
print("Selecting best hyperparameters by mean validation accuracy across folds (via show.py --fold 0)")
print(f"Methods: {', '.join(selected_methods)}")
print("")
sys.stdout.flush()

results = []
for method_name in selected_methods:
    best = None
    tried = 0
    found = 0
    total_cands = len(grouped[method_name])
    print(f"[{method_name}] evaluating {total_cands} candidates")
    sys.stdout.flush()
    for idx, (config_path, flags) in enumerate(grouped[method_name], start=1):
        tried += 1
        t0 = time.time()
        print(f"  [{idx}/{total_cands}] {fmt_flags(flags) or '(no extra flags)'}")
        sys.stdout.flush()
        rc, parsed, raw, cmd = run_show(config_path, flags)
        elapsed = time.time() - t0
        if parsed is None:
            print(f"    missing checkpoint(s) [{elapsed:.1f}s]")
            sys.stdout.flush()
            continue
        if isinstance(parsed, dict) and "error" in parsed:
            print(f"  parse error for: {fmt_flags(flags)}")
            print(parsed["error"])
            print(f"    returncode={rc} [{elapsed:.1f}s]")
            sys.stdout.flush()
            continue
        found += 1
        score = parsed["val_acc_mean_pct"]
        print(
            f"    val_acc={parsed['val_acc_mean_pct']:.2f}±{parsed['val_acc_std_pct']:.2f} "
            f"test_acc={parsed['test_acc_mean_pct']:.2f}±{parsed['test_acc_std_pct']:.2f} "
            f"test_ceu_synergy={parsed['test_ceu_synergy_mean_pct']:.2f}±{parsed['test_ceu_synergy_std_pct']:.2f} "
            f"[{elapsed:.1f}s]"
        )
        sys.stdout.flush()
        # tie-breaker: lower val std, then higher test acc
        key = (score, -parsed["val_acc_std_pct"], parsed.get("test_acc_mean_pct", float("-inf")))
        candidate = {
            "method": method_name,
            "config": config_path,
            "flags": flags,
            "flags_str": fmt_flags(flags),
            "metrics": parsed,
            "show_cmd": " ".join(cmd),
        }
        if best is None:
            best = (key, candidate)
        else:
            if key > best[0]:
                best = (key, candidate)
    if best is None:
        print(f"  no checkpoints found / parsed for {method_name}")
        results.append({
            "method": method_name,
            "status": "missing",
            "candidates_tried": tried,
            "candidates_found": found,
        })
        continue
    cand = best[1]
    m = cand["metrics"]
    print(
        "  best:",
        cand["flags_str"],
        f"| val_acc={m['val_acc_mean_pct']:.2f}±{m['val_acc_std_pct']:.2f}",
        f"| test_acc={m['test_acc_mean_pct']:.2f}±{m['test_acc_std_pct']:.2f}",
        f"| test_ceu_synergy={m['test_ceu_synergy_mean_pct']:.2f}±{m['test_ceu_synergy_std_pct']:.2f}",
    )
    results.append({
        "method": method_name,
        "status": "ok",
        "candidates_tried": tried,
        "candidates_found": found,
        "selected_config": cand["config"],
        "selected_flags": cand["flags"],
        "selected_flags_str": cand["flags_str"],
        "show_cmd": cand["show_cmd"],
        **m,
        "latex_pair": latex_pair(
            m["test_ceu_synergy_mean_pct"],
            m["test_ceu_synergy_std_pct"],
            m["test_acc_mean_pct"],
            m["test_acc_std_pct"],
        ),
    })
    print("")

print("\nFinal table (rows = methods)")
header = [
    "method",
    "latex_pair",
    "(ceu/acc)",
    "val_acc_mean_pct",
    "val_acc_std_pct",
    "selected_hparams",
]
print("\t".join(header))
for r in results:
    if r.get("status") != "ok":
        print(f"{r['method'].replace('.json','')}\tNA\tNA\tNA\tNA\tMISSING")
        continue
    print("\t".join([
        r["method"].replace(".json", ""),
        r["latex_pair"],
        f"{r['test_ceu_synergy_mean_pct']:.3f}±{r['test_ceu_synergy_std_pct']:.3f}/"
        f"{r['test_acc_mean_pct']:.3f}±{r['test_acc_std_pct']:.3f}",
        f"{r['val_acc_mean_pct']:.3f}",
        f"{r['val_acc_std_pct']:.3f}",
        r["selected_flags_str"],
    ]))

print("\nLaTeX-only lines:")
for r in results:
    if r.get("status") != "ok":
        print(f"{r['method'].replace('.json','')}: NA")
        continue
    print(f"{r['method'].replace('.json','')}: {r['latex_pair']}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = OUT_DIR / f"mustard_vt_method_table_{stamp}.json"
tsv_path = OUT_DIR / f"mustard_vt_method_table_{stamp}.tsv"
with json_path.open("w") as f:
    json.dump(results, f, indent=2)

with tsv_path.open("w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(header)
    for r in results:
        if r.get("status") != "ok":
            writer.writerow([r["method"].replace(".json",""), "NA", "NA", "NA", "NA", "MISSING"])
            continue
        writer.writerow([
            r["method"].replace(".json", ""),
            r["latex_pair"],
            f"{r['test_ceu_synergy_mean_pct']:.3f}±{r['test_ceu_synergy_std_pct']:.3f}/"
            f"{r['test_acc_mean_pct']:.3f}±{r['test_acc_std_pct']:.3f}",
            f"{r['val_acc_mean_pct']:.3f}",
            f"{r['val_acc_std_pct']:.3f}",
            r["selected_flags_str"],
        ])

print("")
print(f"Saved JSON: {json_path}")
print(f"Saved TSV : {tsv_path}")
PY
