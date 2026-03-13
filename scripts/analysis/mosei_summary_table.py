"""
Aggregate MOSEI checkpoint sweep results into 3 tables (one per ckpt-mode).

For each mode (loss / accuracy / syn_accuracy):
  - Group checkpoints by (method, config) — config = filename with fold stripped
  - Average combined_acc and ceu_syn% over folds
  - For each method, pick the best config by avg combined_acc
  - Print the table

Usage (from project root):
    python scripts/analysis/mosei_summary_table.py [--sort_by combined_acc|ceu_syn]
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

CKPT_DIR = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Rmask/MOSEI/VT"
BASELINES = ["unimodal_video", "unimodal_text", "late", "ens"]
METHODS   = ["DnR", "MCR", "MMPareto", "ReconBoost",
             "SynIB_RMask_nopre_learned", "SynIB_RMask_nopre_random",
             "SynIB_RMask_learned", "SynIB_RMask_random"]
MODES    = ["loss", "accuracy", "syn_accuracy"]


def config_key(bname, method):
    """Strip fold from filename to get the hyperparameter fingerprint."""
    # handles both fold0 and foldfold0
    return re.sub(r"fold(fold)?\d+_?", "", bname)


BASELINE_GLOBS = {
    "unimodal_video": ("unimodal_video_", lambda b: True),
    "unimodal_text":  ("unimodal_text_",  lambda b: True),
    "late":           ("late_",           lambda b: True),
    "ens":            ("ens_",            lambda b: True),
}

# Map logical method name -> (glob prefix, filename filter fn)
METHOD_GLOBS = {
    "DnR":                      ("DnR_",                 lambda b: True),
    "MCR":                      ("MCR_",                 lambda b: True),
    "MMPareto":                 ("MMPareto_",            lambda b: True),
    "ReconBoost":               ("ReconBoost_",          lambda b: True),
    "SynIB_RMask_nopre_learned":("SynIB_RMask_nopre_",  lambda b: "perturblearned" in b),
    "SynIB_RMask_nopre_random": ("SynIB_RMask_nopre_",  lambda b: "perturbrandom"  in b),
    "SynIB_RMask_learned":      ("SynIB_RMask_",        lambda b: not b.startswith("SynIB_RMask_nopre") and "perturblearned" in b),
    "SynIB_RMask_random":       ("SynIB_RMask_",        lambda b: not b.startswith("SynIB_RMask_nopre") and "perturbrandom"  in b),
}


def load_all(methods):
    """
    Returns: data[method][cfg_key][fold] = {mode: {combined_acc, ceu_syn, f1, k}}
    """
    data = defaultdict(lambda: defaultdict(dict))

    for method in methods:
        prefix, keep = METHOD_GLOBS[method]
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, prefix + "*.pth.tar")))
        ckpts = [c for c in ckpts if keep(os.path.basename(c))]

        for ckpt in ckpts:
            bname = os.path.basename(ckpt)
            fold_m = re.search(r"fold(\d)", bname)
            if not fold_m:
                continue
            fold = int(fold_m.group(1))
            cfg  = config_key(bname, method)

            try:
                cp = torch.load(ckpt, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  [WARN] {bname}: {e}")
                continue

            fold_data = {}
            for mode in MODES:
                res = cp.get(f"post_test_results_{mode}", {})
                if not res:
                    continue
                entry = {}
                if "acc" in res and "combined" in res["acc"]:
                    entry["combined_acc"] = float(res["acc"]["combined"]) * 100
                if "f1" in res and "combined" in res["f1"]:
                    entry["f1"] = float(res["f1"]["combined"]) * 100
                if "k" in res and "combined" in res["k"]:
                    entry["k"] = float(res["k"]["combined"])
                if "ceu_syn" in res and "synergy" in res["ceu_syn"]:
                    entry["ceu_syn"] = float(res["ceu_syn"]["synergy"])
                    # syn_grp% is constant across modes for same fold, store anyway
                    ceu = res["ceu_syn"]
                    total_n = sum(ceu.get(g + "_n", 0) for g in ("synergy", "cue_mod0", "cue_mod1", "coexistence"))
                    entry["syn_grp_pct"] = 100.0 * ceu["synergy_n"] / total_n if total_n > 0 else float("nan")
                if entry:
                    fold_data[mode] = entry

            if fold_data:
                data[method][cfg][fold] = fold_data

    return data


def aggregate(data, mode, sort_by="combined_acc"):
    """
    For a given mode, return per-method best config rows aggregated over folds.
    Row keys: method, best_cfg, n_folds, combined_acc ± std, ceu_syn ± std, f1, k, syn_grp_pct
    """
    rows = []
    for method in METHODS:
        if method not in data:
            continue
        cfg_stats = []
        for cfg, fold_dict in data[method].items():
            fold_entries = [fold_dict[f][mode] for f in fold_dict if mode in fold_dict[f]]
            if not fold_entries:
                continue
            def mean_std(key):
                vals = [e[key] for e in fold_entries if key in e]
                if not vals:
                    return float("nan"), float("nan")
                return float(np.mean(vals)), float(np.std(vals))

            ca_mean, ca_std = mean_std("combined_acc")
            cs_mean, cs_std = mean_std("ceu_syn")
            f1_mean, _      = mean_std("f1")
            k_mean,  _      = mean_std("k")
            sg_mean, _      = mean_std("syn_grp_pct")

            cfg_stats.append({
                "cfg": cfg,
                "n_folds": len(fold_entries),
                "combined_acc": ca_mean,
                "combined_acc_std": ca_std,
                "ceu_syn": cs_mean,
                "ceu_syn_std": cs_std,
                "f1": f1_mean,
                "k": k_mean,
                "syn_grp_pct": sg_mean,
            })

        if not cfg_stats:
            continue

        # pick best config by sort_by metric
        valid = [s for s in cfg_stats if not np.isnan(s[sort_by])]
        if not valid:
            continue
        best = max(valid, key=lambda s: s[sort_by])
        best["method"] = method
        rows.append(best)

    return rows


def fmt(v, decimals=1, suffix=""):
    if np.isnan(v):
        return "-"
    return f"{v:.{decimals}f}{suffix}"


def load_baselines():
    """
    Load baseline models (unimodal_video, unimodal_text, late, ens).
    Returns: rows list with same keys as aggregate(), one row per baseline.
    Only loss mode is used for unimodal/late (they have 3 state dicts).
    ens has a single post_test_results with CEU already computed.
    """
    rows = []
    for name, (prefix, keep) in BASELINE_GLOBS.items():
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, prefix + "*.pth.tar")))
        ckpts = [c for c in ckpts if keep(os.path.basename(c))]

        fold_entries = []
        for ckpt in ckpts:
            bname = os.path.basename(ckpt)
            fold_m = re.search(r"fold(\d)", bname)
            if not fold_m:
                continue
            try:
                cp = torch.load(ckpt, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  [WARN] {bname}: {e}")
                continue

            if name == "ens":
                # ens: single post_test_results with pre-computed CEU
                res = cp.get("post_test_results", {})
                entry = {}
                if "acc" in res and "combined" in res["acc"]:
                    entry["combined_acc"] = float(res["acc"]["combined"]) * 100
                if "f1" in res and "combined" in res["f1"]:
                    entry["f1"] = float(res["f1"]["combined"]) * 100
                if "k" in res and "combined" in res["k"]:
                    entry["k"] = float(res["k"]["combined"])
                # CEU stored as fraction 0-1 in training pipeline format
                ceu = res.get("ceu", {}).get("combined", {})
                if "synergy" in ceu:
                    entry["ceu_syn"] = float(ceu["synergy"]) * 100
                    entry["syn_grp_pct"] = float("nan")  # not stored for ens
                if entry:
                    fold_entries.append(entry)
            else:
                # unimodal / late: use loss-best state dict results
                res = cp.get("post_test_results_loss", {})
                if not res:
                    res = cp.get("post_test_results", {})  # fallback
                entry = {}
                if "acc" in res and "combined" in res["acc"]:
                    entry["combined_acc"] = float(res["acc"]["combined"]) * 100
                if "f1" in res and "combined" in res["f1"]:
                    entry["f1"] = float(res["f1"]["combined"]) * 100
                if "k" in res and "combined" in res["k"]:
                    entry["k"] = float(res["k"]["combined"])
                if "ceu_syn" in res and "synergy" in res["ceu_syn"]:
                    entry["ceu_syn"] = float(res["ceu_syn"]["synergy"])
                    ceu = res["ceu_syn"]
                    total_n = sum(ceu.get(g + "_n", 0) for g in ("synergy", "cue_mod0", "cue_mod1", "coexistence"))
                    entry["syn_grp_pct"] = 100.0 * ceu["synergy_n"] / total_n if total_n > 0 else float("nan")
                if entry:
                    fold_entries.append(entry)

        if not fold_entries:
            continue

        def mean_std(key):
            vals = [e[key] for e in fold_entries if key in e]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals))

        ca_mean, ca_std = mean_std("combined_acc")
        cs_mean, cs_std = mean_std("ceu_syn")
        f1_mean, _      = mean_std("f1")
        k_mean,  _      = mean_std("k")
        sg_mean, _      = mean_std("syn_grp_pct")

        rows.append({
            "method": name,
            "n_folds": len(fold_entries),
            "combined_acc": ca_mean, "combined_acc_std": ca_std,
            "ceu_syn": cs_mean,      "ceu_syn_std": cs_std,
            "f1": f1_mean, "k": k_mean, "syn_grp_pct": sg_mean,
        })
    return rows



def print_table(rows, mode, sort_by, baseline_rows=None):
    print(f"\n{'='*90}")
    print(f"  Mode: {mode}   (best config selected by: {sort_by})")
    print(f"{'='*90}")
    hdr = (f"{'Method':<28}{'Folds':<7}"
           f"{'Acc%':>8}{'±':>4}{'CEU_syn%':>10}{'±':>4}"
           f"{'F1%':>7}{'Kappa':>7}{'SynGrp%':>9}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    if baseline_rows:
        for r in baseline_rows:
            print(
                f"{r['method']:<28}{r['n_folds']:<7}"
                f"{fmt(r['combined_acc']):>8}{fmt(r['combined_acc_std']):>4}"
                f"{fmt(r['ceu_syn']):>10}{fmt(r['ceu_syn_std']):>4}"
                f"{fmt(r['f1']):>7}{fmt(r['k'], decimals=3):>7}"
                f"{fmt(r['syn_grp_pct']):>9}"
            )
        print(sep)
    for r in rows:
        print(
            f"{r['method']:<28}{r['n_folds']:<7}"
            f"{fmt(r['combined_acc']):>8}{fmt(r['combined_acc_std']):>4}"
            f"{fmt(r['ceu_syn']):>10}{fmt(r['ceu_syn_std']):>4}"
            f"{fmt(r['f1']):>7}{fmt(r['k'], decimals=3):>7}"
            f"{fmt(r['syn_grp_pct']):>9}"
        )
    print(sep)


def main(args):
    print("Loading checkpoints ...")
    data = load_all(METHODS)
    baseline_rows = load_baselines()

    for mode in MODES:
        rows = aggregate(data, mode, sort_by=args.sort_by)
        print_table(rows, mode, args.sort_by, baseline_rows=baseline_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort_by", default="combined_acc",
                        choices=["combined_acc", "ceu_syn"],
                        help="Metric used to pick best hyperparameter config per method")
    main(parser.parse_args())
