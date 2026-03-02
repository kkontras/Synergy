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

DEFAULT_CONFIG="${DEFAULT_CONFIG:-./configs/FactorCL/Mosi/default_config_mosi_VT.json}"
SYN_DIR="${SYN_DIR:-./configs/FactorCL/Mosi/syn/VT}"
OUT_DIR="${OUT_DIR:-./artifacts/reports}"
mkdir -p "${OUT_DIR}"

# Stage 1: unimodal lr/wd selection
UNIMODAL_LRS_CSV="${UNIMODAL_LRS_CSV:-0.001,0.0005,0.0001,0.00005}"
UNIMODAL_WDS_CSV="${UNIMODAL_WDS_CSV:-0.001,0.0001,0.00001}"
# Stage 2: RMask_nopre lr/wd selection (defaults to same grid)
RMASK_NOPRE_LRS_CSV="${RMASK_NOPRE_LRS_CSV:-${UNIMODAL_LRS_CSV}}"
RMASK_NOPRE_WDS_CSV="${RMASK_NOPRE_WDS_CSV:-${UNIMODAL_WDS_CSV}}"

# Stage 3: method selection at fixed lr/wd from best RMask_nopre
METHOD_RUN_MODE="${METHOD_RUN_MODE:-all}"      # all | single
METHOD_TARGET="${METHOD_TARGET:-MMPareto}"
SELECT_METHODS_CSV="${SELECT_METHODS_CSV:-}"

# Method hyperparameter grids (matching Mustard/URFunny)
MCR_L_CSV="${MCR_L_CSV:-0.001,0.01,0.1,1}"
MCR_MULTIL_CSV="${MCR_MULTIL_CSV:-0.01,0.1,1}"
MMPARETO_ALPHA_CSV="${MMPARETO_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"
DNR_ALPHA_CSV="${DNR_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"
DNR_KMEPOCH_CSV="${DNR_KMEPOCH_CSV:-1,3,5,10}"
RECONBOOST_ALPHA_CSV="${RECONBOOST_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"
RECONBOOST_STAGES_CSV="${RECONBOOST_STAGES_CSV:-1,4,10}"
RECONBOOST_W1_CSV="${RECONBOOST_W1_CSV:-1,3,5,10}"
RMASK_LS_CSV="${RMASK_LS_CSV:-0}"
RMASK_LEARNED_L_CSV="${RMASK_LEARNED_L_CSV:-0.001,0.01,0.1,1}"
RMASK_LEARNED_LSPARSE_CSV="${RMASK_LEARNED_LSPARSE_CSV:-0.001,0.01,0.1,1,3,5,10}"
RMASK_RANDOM_L_CSV="${RMASK_RANDOM_L_CSV:-0.001,0.01,0.1,1}"
RMASK_RANDOM_PMIN_CSV="${RMASK_RANDOM_PMIN_CSV:-0.1,0.3,0.5,0.7,0.9}"

export PYTHON_BIN DEFAULT_CONFIG SYN_DIR OUT_DIR
export UNIMODAL_LRS_CSV UNIMODAL_WDS_CSV RMASK_NOPRE_LRS_CSV RMASK_NOPRE_WDS_CSV
export METHOD_RUN_MODE METHOD_TARGET SELECT_METHODS_CSV
export MCR_L_CSV MCR_MULTIL_CSV MMPARETO_ALPHA_CSV
export DNR_ALPHA_CSV DNR_KMEPOCH_CSV
export RECONBOOST_ALPHA_CSV RECONBOOST_STAGES_CSV RECONBOOST_W1_CSV
export RMASK_LS_CSV RMASK_LEARNED_L_CSV RMASK_LEARNED_LSPARSE_CSV
export RMASK_RANDOM_L_CSV RMASK_RANDOM_PMIN_CSV

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
SYN_DIR = Path(os.environ["SYN_DIR"])
OUT_DIR = Path(os.environ["OUT_DIR"])
UNIMODAL_LRS = [x.strip() for x in os.environ.get("UNIMODAL_LRS_CSV", "0.001,0.0005,0.0001,0.00005").split(",") if x.strip()]
UNIMODAL_WDS = [x.strip() for x in os.environ.get("UNIMODAL_WDS_CSV", "0.001,0.0001,0.00001").split(",") if x.strip()]
RMASK_NOPRE_LRS = [x.strip() for x in os.environ.get("RMASK_NOPRE_LRS_CSV", "").split(",") if x.strip()] or UNIMODAL_LRS
RMASK_NOPRE_WDS = [x.strip() for x in os.environ.get("RMASK_NOPRE_WDS_CSV", "").split(",") if x.strip()] or UNIMODAL_WDS
METHOD_RUN_MODE = os.environ.get("METHOD_RUN_MODE", "all")
METHOD_TARGET = os.environ.get("METHOD_TARGET", "MMPareto").strip()
SELECT_METHODS_CSV = os.environ.get("SELECT_METHODS_CSV", "").strip()
MCR_LS = [x.strip() for x in os.environ.get("MCR_L_CSV", "0.001,0.01,0.1,1").split(",") if x.strip()]
MCR_MULTILS = [x.strip() for x in os.environ.get("MCR_MULTIL_CSV", "0.01,0.1,1").split(",") if x.strip()]
MMPARETO_ALPHAS = [x.strip() for x in os.environ.get("MMPARETO_ALPHA_CSV", "0.5,1.0,1.5,2.0,3.0,5.0").split(",") if x.strip()]
DNR_ALPHAS = [x.strip() for x in os.environ.get("DNR_ALPHA_CSV", "0.5,1.0,1.5,2.0,3.0,5.0").split(",") if x.strip()]
DNR_KMEPOCHS = [x.strip() for x in os.environ.get("DNR_KMEPOCH_CSV", "1,3,5,10").split(",") if x.strip()]
RECONBOOST_ALPHAS = [x.strip() for x in os.environ.get("RECONBOOST_ALPHA_CSV", "0.5,1.0,1.5,2.0,3.0,5.0").split(",") if x.strip()]
RECONBOOST_STAGES = [x.strip() for x in os.environ.get("RECONBOOST_STAGES_CSV", "1,4,10").split(",") if x.strip()]
RECONBOOST_W1S = [x.strip() for x in os.environ.get("RECONBOOST_W1_CSV", "1,3,5,10").split(",") if x.strip()]
RMASK_LS = [x.strip() for x in os.environ.get("RMASK_LS_CSV", "0").split(",") if x.strip()]
RMASK_LEARNED_LS = [x.strip() for x in os.environ.get("RMASK_LEARNED_L_CSV", "0.001,0.01,0.1,1").split(",") if x.strip()]
RMASK_LEARNED_LSPARSES = [x.strip() for x in os.environ.get("RMASK_LEARNED_LSPARSE_CSV", "0.001,0.01,0.1,1,3,5,10").split(",") if x.strip()]
RMASK_RANDOM_LS = [x.strip() for x in os.environ.get("RMASK_RANDOM_L_CSV", "0.001,0.01,0.1,1").split(",") if x.strip()]
RMASK_RANDOM_PMINS = [x.strip() for x in os.environ.get("RMASK_RANDOM_PMIN_CSV", "0.1,0.3,0.5,0.7,0.9").split(",") if x.strip()]

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
    val_accs, test_accs, test_ceus = [], [], []
    for line in clean.splitlines():
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
        "test_ceu_synergy_fold_raw": test_ceus,
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
        "--config", str(config_path),
        "--default_config", DEFAULT_CONFIG,
        "--fold", "0",
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
    return proc.returncode, parse_show_output(proc.stdout), proc.stdout, cmd


def fmt_flags(flags):
    out = []
    i = 0
    while i < len(flags):
        k = flags[i]
        if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
            out.append(f"{k}={flags[i+1]}")
            i += 2
        else:
            out.append(k)
            i += 1
    return " ".join(out)


def sweep_best_lrwd(label, config_path, lrs, wds):
    print(f"[{label}] sweeping {len(lrs) * len(wds)} lr/wd candidates")
    sys.stdout.flush()
    best = None
    found = 0
    total = len(lrs) * len(wds)
    idx = 0
    for lr in lrs:
        for wd in wds:
            idx += 1
            flags = ["--lr", lr, "--wd", wd]
            t0 = time.time()
            print(f"  [{idx}/{total}] {fmt_flags(flags)}")
            sys.stdout.flush()
            rc, parsed, raw, cmd = run_show(config_path, flags)
            dt = time.time() - t0
            if parsed is None:
                print(f"    missing checkpoint(s) [{dt:.1f}s]")
                continue
            if isinstance(parsed, dict) and "error" in parsed:
                print(f"    parse error [{dt:.1f}s]")
                print(parsed["error"])
                continue
            found += 1
            key = (parsed["val_acc_mean_pct"], -parsed["val_acc_std_pct"], parsed["test_acc_mean_pct"])
            print(
                f"    val_acc={parsed['val_acc_mean_pct']:.2f}±{parsed['val_acc_std_pct']:.2f} "
                f"test_acc={parsed['test_acc_mean_pct']:.2f}±{parsed['test_acc_std_pct']:.2f} "
                f"test_ceu_synergy={parsed['test_ceu_synergy_mean_pct']:.2f}±{parsed['test_ceu_synergy_std_pct']:.2f} [{dt:.1f}s]"
            )
            cand = {
                "label": label,
                "status": "ok",
                "selected_config": str(config_path),
                "selected_flags": flags,
                "selected_flags_str": fmt_flags(flags),
                "show_cmd": " ".join(cmd),
                **parsed,
            }
            if best is None or key > best[0]:
                best = (key, cand)
            sys.stdout.flush()
    if best is None:
        print(f"  no checkpoints found / parsed for {label}\n")
        return {"label": label, "status": "missing", "selected_config": str(config_path), "candidates_total": total, "candidates_found": found}
    out = best[1]
    out["candidates_total"] = total
    out["candidates_found"] = found
    out["lr"] = out["selected_flags"][1]
    out["wd"] = out["selected_flags"][3]
    out["latex_pair"] = latex_pair(out["test_ceu_synergy_mean_pct"], out["test_ceu_synergy_std_pct"], out["test_acc_mean_pct"], out["test_acc_std_pct"])
    print(
        f"  best: {out['selected_flags_str']} | "
        f"val_acc={out['val_acc_mean_pct']:.2f}±{out['val_acc_std_pct']:.2f} | "
        f"test_acc={out['test_acc_mean_pct']:.2f}±{out['test_acc_std_pct']:.2f} | "
        f"test_ceu_synergy={out['test_ceu_synergy_mean_pct']:.2f}±{out['test_ceu_synergy_std_pct']:.2f}\n"
    )
    sys.stdout.flush()
    return out


def method_order():
    rows = [
        "MMPareto.json",
        "DnR.json",
        "MCR.json",
        "ReconBoost.json",
        "MLB.json",
        "ens.json",
        "joint_training.json",
        "synprom_RMask_nopre.json",
        "synprom_RMask.json",
        "synprom_RMask_learned.json",
        "synprom_RMask_random.json",
    ]
    return rows


def display_name(method_name):
    alias = {
        "synprom_RMask.json": "SynProm RMask",
        "synprom_RMask_nopre.json": "SynProm RMask no-pre",
        "synprom_RMask_learned.json": "SynProm Learned",
        "synprom_RMask_random.json": "SynProm Random",
    }
    return alias.get(method_name, method_name.replace(".json", ""))


def candidate_specs(fixed_lr, fixed_wd):
    base_synprom = SYN_DIR / "synprom_RMask.json"

    methods = method_order()
    if SELECT_METHODS_CSV:
        wanted = {m.strip() if m.strip().endswith('.json') else f"{m.strip()}.json" for m in SELECT_METHODS_CSV.split(',') if m.strip()}
        methods = [m for m in methods if m in wanted]
    if METHOD_RUN_MODE == "single":
        target = METHOD_TARGET if METHOD_TARGET.endswith('.json') else f"{METHOD_TARGET}.json"
        methods = [m for m in methods if m == target]

    specs = []
    for m in methods:
        cfg = SYN_DIR / m
        if m == "MMPareto.json":
            for alpha in MMPARETO_ALPHAS:
                specs.append((m, str(cfg), ["--lr", fixed_lr, "--wd", fixed_wd, "--alpha", alpha]))
        elif m == "DnR.json":
            for alpha in DNR_ALPHAS:
                for kmpe in DNR_KMEPOCHS:
                    specs.append((m, str(cfg), ["--lr", fixed_lr, "--wd", fixed_wd, "--alpha", alpha, "--kmepoch", kmpe]))
        elif m == "MCR.json":
            for l in MCR_LS:
                for multil in MCR_MULTILS:
                    specs.append((m, str(cfg), ["--lr", fixed_lr, "--wd", fixed_wd, "--l", l, "--multil", multil]))
        elif m == "ReconBoost.json":
            for alpha in RECONBOOST_ALPHAS:
                for stages in RECONBOOST_STAGES:
                    for w1 in RECONBOOST_W1S:
                        specs.append((m, str(cfg), [
                            "--lr", fixed_lr, "--wd", fixed_wd,
                            "--alpha", alpha,
                            "--recon_weight1", w1, "--recon_weight2", "1",
                            "--recon_epochstages", stages, "--recon_ensemblestages", stages,
                        ]))
        elif m == "synprom_RMask_learned.json":
            for l in RMASK_LEARNED_LS:
                for lsparse in RMASK_LEARNED_LSPARSES:
                    specs.append((m, str(base_synprom), [
                        "--lr", fixed_lr, "--wd", fixed_wd, "--l", l,
                        "--perturb", "learned", "--perturn", "learned",
                        "--perturb_fill", "ema", "--perturb_lsparse", lsparse,
                    ]))
        elif m == "synprom_RMask_random.json":
            for l in RMASK_RANDOM_LS:
                for pmin in RMASK_RANDOM_PMINS:
                    specs.append((m, str(base_synprom), [
                        "--lr", fixed_lr, "--wd", fixed_wd, "--l", l,
                        "--perturb", "random", "--perturn", "random",
                        "--perturb_fill", "ema", "--perturb_pmin", pmin,
                    ]))
        elif m == "synprom_RMask.json":
            for l in RMASK_LS:
                specs.append((m, str(base_synprom), ["--lr", fixed_lr, "--wd", fixed_wd, "--l", l]))
        elif m == "ens.json":
            specs.append((m, str(cfg), ["--lr", fixed_lr, "--wd", fixed_wd, "--l", "0"]))
        elif m == "synprom_RMask_nopre.json":
            specs.append((m, str(cfg), ["--lr", fixed_lr, "--wd", fixed_wd]))
        else:
            # MLB, joint_training: fixed lr/wd only
            specs.append((m, str(cfg), ["--lr", fixed_lr, "--wd", fixed_wd]))
    return specs


# Stage 1: best unimodals
print("Step 1/3: Select best unimodal lr/wd (by mean val acc across folds)")
print("")
unimodal_cfgs = [
    ("unimodal_video", SYN_DIR / "unimodal_video.json"),
    ("unimodal_text", SYN_DIR / "unimodal_text.json"),
    ("unimodal_audio", SYN_DIR / "unimodal_audio.json"),
]
unimodal_results = []
for label, cfg in unimodal_cfgs:
    if not cfg.exists():
        print(f"  skipping {label}: config not found at {cfg}")
        continue
    unimodal_results.append(sweep_best_lrwd(label, cfg, UNIMODAL_LRS, UNIMODAL_WDS))

# Stage 2: best rmask_nopre lr/wd
print("Step 2/3: Select best synprom_RMask_nopre lr/wd (fixed optimizer for methods)")
print("")
rmask_nopre_cfg = SYN_DIR / "synprom_RMask_nopre.json"
if not rmask_nopre_cfg.exists():
    print(f"Missing config: {rmask_nopre_cfg}", file=sys.stderr)
    sys.exit(1)
rmask_nopre_best = sweep_best_lrwd("synprom_RMask_nopre", rmask_nopre_cfg, RMASK_NOPRE_LRS, RMASK_NOPRE_WDS)
if rmask_nopre_best.get("status") != "ok":
    print("Could not determine fixed lr/wd from synprom_RMask_nopre.", file=sys.stderr)
    sys.exit(1)
fixed_lr = rmask_nopre_best["lr"]
fixed_wd = rmask_nopre_best["wd"]

# Stage 3: method-specific sweeps at fixed lr/wd
print("Step 3/3: Select best hyperparameters per method at fixed lr/wd from synprom_RMask_nopre")
print(f"Fixed lr/wd = {fixed_lr}/{fixed_wd}")
print("")
all_specs = candidate_specs(fixed_lr, fixed_wd)
if not all_specs:
    print("No methods selected.", file=sys.stderr)
    sys.exit(1)

grouped = {}
for method_name, cfg, flags in all_specs:
    grouped.setdefault(method_name, []).append((cfg, flags))
selected_methods = [m for m in method_order() if m in grouped]
print(f"Methods: {', '.join(selected_methods)}")
print("")
sys.stdout.flush()

results = []
for method_name in selected_methods:
    cands = grouped[method_name]
    print(f"[{method_name}] evaluating {len(cands)} candidates")
    sys.stdout.flush()
    best = None
    found = 0
    for idx, (cfg, flags) in enumerate(cands, start=1):
        t0 = time.time()
        print(f"  [{idx}/{len(cands)}] {fmt_flags(flags) or '(no extra flags)'}")
        sys.stdout.flush()
        rc, parsed, raw, cmd = run_show(cfg, flags)
        dt = time.time() - t0
        if parsed is None:
            print(f"    missing checkpoint(s) [{dt:.1f}s]")
            continue
        if isinstance(parsed, dict) and "error" in parsed:
            print(f"    parse error [{dt:.1f}s]")
            print(parsed["error"])
            continue
        found += 1
        key = (parsed["val_acc_mean_pct"], -parsed["val_acc_std_pct"], parsed["test_acc_mean_pct"])
        print(
            f"    val_acc={parsed['val_acc_mean_pct']:.2f}±{parsed['val_acc_std_pct']:.2f} "
            f"test_acc={parsed['test_acc_mean_pct']:.2f}±{parsed['test_acc_std_pct']:.2f} "
            f"test_ceu_synergy={parsed['test_ceu_synergy_mean_pct']:.2f}±{parsed['test_ceu_synergy_std_pct']:.2f} [{dt:.1f}s]"
        )
        sys.stdout.flush()
        cand = {
            "method": method_name,
            "selected_config": cfg,
            "selected_flags": flags,
            "selected_flags_str": fmt_flags(flags),
            "show_cmd": " ".join(cmd),
            **parsed,
        }
        if best is None or key > best[0]:
            best = (key, cand)
    if best is None:
        print(f"  no checkpoints found / parsed for {method_name}\n")
        results.append({"method": method_name, "status": "missing"})
        continue
    r = best[1]
    r["status"] = "ok"
    r["fixed_lr_from_rmask_nopre"] = fixed_lr
    r["fixed_wd_from_rmask_nopre"] = fixed_wd
    r["latex_pair"] = latex_pair(r["test_ceu_synergy_mean_pct"], r["test_ceu_synergy_std_pct"], r["test_acc_mean_pct"], r["test_acc_std_pct"])
    print(
        "  best:", r["selected_flags_str"],
        f"| val_acc={r['val_acc_mean_pct']:.2f}±{r['val_acc_std_pct']:.2f}",
        f"| test_acc={r['test_acc_mean_pct']:.2f}±{r['test_acc_std_pct']:.2f}",
        f"| ceu_s={r['test_ceu_synergy_mean_pct']:.2f}±{r['test_ceu_synergy_std_pct']:.2f}",
    )
    print("")
    sys.stdout.flush()
    results.append(r)

print("\nSelected unimodals (best lr/wd by val acc)")
print("unimodal\tval_acc_mean_pct\tval_acc_std_pct\ttest_acc_mean_pct\ttest_acc_std_pct\tselected_hparams")
for r in unimodal_results:
    name = r.get("label", "unimodal")
    if r.get("status") != "ok":
        print(f"{name}\tNA\tNA\tNA\tNA\tMISSING")
        continue
    print(
        f"{name}\t{r['val_acc_mean_pct']:.3f}\t{r['val_acc_std_pct']:.3f}\t"
        f"{r['test_acc_mean_pct']:.3f}\t{r['test_acc_std_pct']:.3f}\t{r['selected_flags_str']}"
    )

print("\nFixed optimizer pair from synprom_RMask_nopre")
print(f"lr={fixed_lr} wd={fixed_wd}")

print("\nFinal table (rows = methods)")
print("method\tlatex_pair\t(ceu/acc)\tval_acc_mean_pct\tval_acc_std_pct\tselected_hparams")
for r in results:
    name = display_name(r["method"])
    if r.get("status") != "ok":
        print(f"{name}\tNA\tNA\tNA\tNA\tMISSING")
        continue
    print(
        f"{name}\t{r['latex_pair']}\t"
        f"{r['test_ceu_synergy_mean_pct']:.3f}±{r['test_ceu_synergy_std_pct']:.3f}/"
        f"{r['test_acc_mean_pct']:.3f}±{r['test_acc_std_pct']:.3f}\t"
        f"{r['val_acc_mean_pct']:.3f}\t{r['val_acc_std_pct']:.3f}\t"
        f"{r['selected_flags_str']}"
    )

print("\nLaTeX-only lines:")
for r in results:
    name = display_name(r["method"])
    if r.get("status") != "ok":
        print(f"{name}: NA")
    else:
        print(f"{name}: {r['latex_pair']}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = OUT_DIR / f"mosi_vt_method_table_{stamp}.json"
tsv_path = OUT_DIR / f"mosi_vt_method_table_{stamp}.tsv"
with json_path.open("w") as f:
    json.dump({
        "selection_policy": {
            "step1": "best unimodals by val acc over lr/wd",
            "step2": "best synprom_RMask_nopre by val acc over lr/wd",
            "step3": "best per-method hyperparams at fixed lr/wd from step2",
            "fixed_lr": fixed_lr,
            "fixed_wd": fixed_wd,
        },
        "unimodals": unimodal_results,
        "rmask_nopre_best": rmask_nopre_best,
        "methods": results,
    }, f, indent=2)
with tsv_path.open("w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["method", "latex_pair", "test_ceu_synergy_mean_pct", "test_ceu_synergy_std_pct",
                     "test_acc_mean_pct", "test_acc_std_pct", "val_acc_mean_pct", "val_acc_std_pct",
                     "selected_hparams", "selected_config"])
    for r in results:
        if r.get("status") != "ok":
            writer.writerow([display_name(r["method"]), "NA", "NA", "NA", "NA", "NA", "NA", "NA", "MISSING", "MISSING"])
            continue
        writer.writerow([
            display_name(r["method"]),
            r["latex_pair"],
            f"{r['test_ceu_synergy_mean_pct']:.3f}",
            f"{r['test_ceu_synergy_std_pct']:.3f}",
            f"{r['test_acc_mean_pct']:.3f}",
            f"{r['test_acc_std_pct']:.3f}",
            f"{r['val_acc_mean_pct']:.3f}",
            f"{r['val_acc_std_pct']:.3f}",
            r["selected_flags_str"],
            r["selected_config"],
        ])

print("")
print(f"Saved JSON: {json_path}")
print(f"Saved TSV : {tsv_path}")
PY
