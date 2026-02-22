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

GPU="${1:-0}"
MODE="${2:-run}"    # run | dry

DEFAULT_CONFIG="${DEFAULT_CONFIG:-./configs/FactorCL/URFunny/default_config_ur_funny_VT.json}"
RELEASE_DIR="${RELEASE_DIR:-./configs/FactorCL/URFunny/release/VT}"
FOLDS_CSV="${FOLDS_CSV:-0,1,2}"
ONLY="${ONLY:-}"    # comma-separated subset, e.g. ONLY="ens,MCR"

"${PYTHON_BIN}" - "${GPU}" "${MODE}" "${DEFAULT_CONFIG}" "${RELEASE_DIR}" "${FOLDS_CSV}" "${ONLY}" <<'PY'
import argparse
import contextlib
import io
import itertools
import json
import os
import statistics
import subprocess
import sys

from scripts.entrypoints.show import print_search

gpu = sys.argv[1]
mode = sys.argv[2]
default_config = sys.argv[3]
release_dir = sys.argv[4]
folds_csv = sys.argv[5]
only_csv = sys.argv[6]
folds = [int(x) for x in folds_csv.split(",") if x.strip() != ""]
only = set([x.strip() for x in only_csv.split(",") if x.strip()]) if only_csv else None


def base_namespace():
    return argparse.Namespace(
        config=None,
        default_config=default_config,
        fold=None,
        alpha=None,
        validate_with="accuracy",
        transform_type=None,
        trasform_before=None,
        tanh_mode=None,
        tanh_mode_beta=None,
        regby=None,
        clip=None,
        batch_size=None,
        l=None,
        multil=None,
        l_diffsq=None,
        lib=None,
        ratio_us=None,
        kmepoch=None,
        num_samples=None,
        pow=None,
        nstep=None,
        contrcoeff=None,
        kde_coeff=None,
        etube=None,
        temperature=None,
        contr_type=None,
        shuffle_type=None,
        num_classes=None,
        base_alpha=None,
        alpha_var=None,
        base_beta=None,
        beta_var=None,
        optim_method=None,
        ilr_c=None,
        ilr_g=None,
        mmcosine_scaling=None,
        ending_epoch=None,
        load_ongoing=None,
        commonlayers=None,
        recon_weight1=None,
        recon_weight2=None,
        recon_epochstages=None,
        recon_ensemblestages=None,
        lr=None,
        wd=None,
        mm=None,
        cls=None,
        printing=False,
        ironic_rate=None,
        perturb=None,
        perturb_fill=None,
        perturb_pmax=None,
        perturb_pmin=None,
        perturb_lsparse=None,
        pre=False,
        frozen=False,
        tdqm_disable=False,
        start_over=False,
    )


def to_float(v):
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    try:
        return float(v)
    except Exception:
        return None


def extract_acc(metric_dict):
    if not isinstance(metric_dict, dict):
        return None
    acc = metric_dict.get("acc")
    if not isinstance(acc, dict):
        return None
    return to_float(acc.get("combined"))


def evaluate_fold(candidate, fold):
    ns = base_namespace()
    for k, v in candidate.items():
        setattr(ns, k, v)
    ns.fold = str(fold)

    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            val_metrics, test_metrics = print_search(
                config_path=ns.config,
                default_config_path=ns.default_config,
                args=ns,
            )
    except Exception:
        return None

    val_acc = extract_acc(val_metrics)
    test_acc = extract_acc(test_metrics)
    if val_acc is None:
        return None
    return val_acc, test_acc


def grid(base, dims):
    keys = list(dims.keys())
    for vals in itertools.product(*(dims[k] for k in keys)):
        c = dict(base)
        for k, v in zip(keys, vals):
            c[k] = v
        yield c


lrwd = {"lr": ["0.001", "0.0001", "0.00005", "0.00001"], "wd": ["0.001", "0.0001", "0.00001"]}
method_specs = [
    ("unimodal_video", "unimodal_video.json", lrwd),
    ("unimodal_text", "unimodal_text.json", lrwd),
    ("synprom_RMask_nopre", "synprom_RMask_nopre.json", lrwd),
    ("ens", "ens.json", {"lr": ["0.001"], "wd": ["0.001"], "l": ["0"]}),
    (
        "synprom_RMask_learned",
        "synprom_RMask.json",
        {
            "lr": ["0.001"],
            "wd": ["0.001"],
            "l": ["0.001", "0.01", "0.1", "1"],
            "perturb": ["learned"],
            "perturb_fill": ["ema"],
            "perturb_lsparse": ["0.001", "0.01", "0.1", "1", "3", "5", "10"],
        },
    ),
    (
        "synprom_RMask_random",
        "synprom_RMask.json",
        {
            "lr": ["0.001"],
            "wd": ["0.001"],
            "l": ["0.001", "0.01", "0.1", "1"],
            "perturb": ["random"],
            "perturb_fill": ["ema"],
            "perturb_pmin": ["0.1", "0.3", "0.5", "0.7", "0.9"],
        },
    ),
    (
        "MCR",
        "MCR.json",
        {"lr": ["0.001"], "wd": ["0.001"], "l": ["0.001", "0.01", "0.1", "1"], "multil": ["0.01", "0.1", "1"]},
    ),
    ("MMPareto", "MMPareto.json", {"lr": ["0.001"], "wd": ["0.001"], "alpha": ["0.5", "1.0", "1.5", "2.0", "3.0", "5.0"]}),
    (
        "DnR",
        "DnR.json",
        {"lr": ["0.001"], "wd": ["0.001"], "alpha": ["0.5", "1.0", "1.5", "2.0", "3.0", "5.0"], "kmepoch": ["1", "3", "5", "10"]},
    ),
    (
        "ReconBoost",
        "ReconBoost.json",
        {
            "lr": ["0.001"],
            "wd": ["0.001"],
            "alpha": ["0.5", "1.0", "1.5", "2.0", "3.0", "5.0"],
            "recon_weight1": ["1", "3", "5", "10"],
            "recon_weight2": ["1"],
            "recon_epochstages": ["1", "4", "10"],
            "recon_ensemblestages": ["1", "4", "10"],
        },
    ),
]

methods = []
for name, cfg_file, dims in method_specs:
    if only is not None and name not in only:
        continue
    base = {
        "config": os.path.join(release_dir, cfg_file),
        "default_config": default_config,
        "validate_with": "accuracy",
    }
    methods.append((name, list(grid(base, dims))))

if not methods:
    print("No methods selected.")
    sys.exit(1)

print(f"Selection folds: {folds}")
best = {}

for method_name, candidates in methods:
    print("")
    print(f"[{method_name}] evaluating {len(candidates)} candidates")
    best_score = None
    best_payload = None

    for cand in candidates:
        val_list = []
        test_list = []
        ok = True
        for fold in folds:
            out = evaluate_fold(cand, fold)
            if out is None:
                ok = False
                break
            v_acc, t_acc = out
            val_list.append(v_acc)
            test_list.append(t_acc)
        if not ok:
            continue

        mean_val = statistics.fmean(val_list)
        if best_score is None or mean_val > best_score:
            best_score = mean_val
            best_payload = {
                "candidate": cand,
                "val_list": val_list,
                "test_list": test_list,
                "mean_val": mean_val,
                "mean_test": statistics.fmean([x for x in test_list if x is not None]) if any(x is not None for x in test_list) else None,
            }

    if best_payload is None:
        print("  no complete checkpoints found")
        continue

    best[method_name] = best_payload
    val_pct = [round(v * 100, 2) for v in best_payload["val_list"]]
    test_pct = [None if t is None else round(t * 100, 2) for t in best_payload["test_list"]]
    mean_val_pct = round(best_payload["mean_val"] * 100, 2)
    mean_test_pct = None if best_payload["mean_test"] is None else round(best_payload["mean_test"] * 100, 2)

    print(f"  best mean val acc: {mean_val_pct}%")
    print(f"  val acc list (%): {val_pct}")
    print(f"  test acc list (%): {test_pct}")
    print(f"  mean test acc: {mean_test_pct}%")
    printable_args = {k: v for k, v in best_payload['candidate'].items() if k not in ('default_config',)}
    print(f"  best args: {json.dumps(printable_args, sort_keys=True)}")

if not best:
    print("")
    print("No runnable methods with complete folds found.")
    sys.exit(1)

print("")
print("=== Summary (best per method) ===")
for method_name in best:
    item = best[method_name]
    mv = round(item["mean_val"] * 100, 2)
    mt = None if item["mean_test"] is None else round(item["mean_test"] * 100, 2)
    print(f"{method_name}: mean_val={mv}% mean_test={mt}%")

if mode == "dry":
    print("")
    print("Dry mode: skip running show.py on fold 0.")
    sys.exit(0)

print("")
print("=== Running show.py on fold 0 for each best setup ===")

show_order = [
    "config",
    "default_config",
    "fold",
    "lr",
    "wd",
    "alpha",
    "l",
    "multil",
    "kmepoch",
    "recon_weight1",
    "recon_weight2",
    "recon_epochstages",
    "recon_ensemblestages",
    "perturb",
    "perturb_fill",
    "perturb_pmin",
    "perturb_lsparse",
    "validate_with",
]

for method_name, payload in best.items():
    cand = dict(payload["candidate"])
    cand["fold"] = "0"
    cli = [sys.executable, "scripts/entrypoints/show.py"]
    for k in show_order:
        if k in cand and cand[k] is not None:
            cli.extend([f"--{k}", str(cand[k])])
    print(f"\n[{method_name}] {' '.join(cli)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(cli, env=env, check=False)
PY
