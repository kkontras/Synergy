"""
Evaluate all 3 best-model checkpoints (loss / accuracy / syn_accuracy) from a single
.pth.tar file and write the results back into the checkpoint.

Usage:
    python scripts/analysis/eval_all_ckpt_modes.py \
        --checkpoint <path.pth.tar> \
        --config <method_config.json> \
        --default_config configs/FactorCL/Mosei/default_config_mosei_VT_syn.json \
        --fold 0 \
        --device cuda:0
"""

import argparse
import os
import pickle
import sys

import numpy as np

# Ensure the project root is on the path regardless of CWD
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator

STATE_DICT_KEYS = ["loss", "accuracy", "syn_accuracy"]


def reform_state_dict(sd):
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("parametrizations.weight.original0", "weight_g"): v for k, v in sd.items()}
    sd = {k.replace("parametrizations.weight.original1", "weight_v"): v for k, v in sd.items()}
    return sd


def load_ceu_pkl(config):
    """Load the test CEU pkl if configured in config.model.ceu.test."""
    try:
        ceu_path = config.model.ceu.get("test", None)
    except Exception:
        return None
    if ceu_path is None or not os.path.exists(ceu_path):
        return None
    with open(ceu_path, "rb") as f:
        return pickle.load(f)["folds"]


def compute_ceu_synergy(metrics, ceu_folds, fold):
    """
    Compute CEU synergy from Validator metrics + unimodal CEU pkl.

    Uses the same logic as General_Evaluator.ceu():
      - mod0_preds = ceu_folds[fold]        (first unimodal, e.g. video)
      - mod1_preds = ceu_folds[fold + 3]    (second unimodal, e.g. text)
      - synergy = accuracy on samples where both unimodals were wrong

    Returns dict with synergy / cue_mod0 / cue_mod1 / coexistence, or None on failure.
    """
    if ceu_folds is None:
        return None
    if "total_preds" not in metrics or "combined" not in metrics["total_preds"]:
        return None

    try:
        mod0_preds   = np.array(ceu_folds[fold]["preds_combined"])
        mod0_targets = np.array(ceu_folds[fold]["targets"])
        mod1_preds   = np.array(ceu_folds[fold + 3]["preds_combined"])
        mod1_targets = np.array(ceu_folds[fold + 3]["targets"])
    except KeyError as e:
        print("[CEU] Missing fold key: {}".format(e))
        return None

    mm_logits  = np.array(metrics["total_preds"]["combined"])   # (N, C)
    mm_targets = np.array(metrics["total_preds_target"])        # (N,)

    # Sanity check: targets must align across all three
    if not (len(mm_targets) == len(mod0_targets) == len(mod1_targets)):
        print("[CEU] Target length mismatch: mm={} mod0={} mod1={}".format(
            len(mm_targets), len(mod0_targets), len(mod1_targets)))
        return None
    if not ((mm_targets == mod0_targets).all() and (mod0_targets == mod1_targets).all()):
        print("[CEU] Target values do not align across modalities")
        return None

    mod0_correct = (mod0_preds.argmax(-1) == mod0_targets)   # (N,) bool
    mod1_correct = (mod1_preds.argmax(-1) == mod1_targets)   # (N,)
    mm_correct   = (mm_logits.argmax(-1)  == mm_targets)     # (N,)

    groups = {
        "synergy":     (~mod0_correct) & (~mod1_correct),
        "cue_mod0":    ( mod0_correct) & (~mod1_correct),
        "cue_mod1":    (~mod0_correct) & ( mod1_correct),
        "coexistence": ( mod0_correct) & ( mod1_correct),
    }

    result = {}
    for name, mask in groups.items():
        n = mask.sum()
        result[name] = float(mm_correct[mask].sum() / n * 100.0) if n > 0 else float("nan")
        result[name + "_n"] = int(n)

    return result


def _fmt_acc(metrics, key):
    if "acc" in metrics and key in metrics["acc"]:
        return "{:.1f}%".format(metrics["acc"][key] * 100)
    return "-"


def _fmt_loss(metrics, key):
    if "loss" in metrics and key in metrics["loss"]:
        return "{:.4f}".format(metrics["loss"][key])
    return "-"


def _fmt_ceu(ceu, key):
    if ceu is not None and key in ceu and not np.isnan(ceu[key]):
        return "{:.1f}%".format(ceu[key])
    return "-"


def print_results_table(results, ceu_results):
    """Print table: mode | combined_acc | combined_loss | ceu_syn | other pred heads."""
    # Collect pred keys — put combined first
    pred_keys = []
    for mode, metrics in results.items():
        if "acc" in metrics:
            for k in metrics["acc"]:
                if k not in pred_keys:
                    pred_keys.append(k)
    if "combined" in pred_keys:
        pred_keys.insert(0, pred_keys.pop(pred_keys.index("combined")))

    col_w = 14
    header = "{:<14}".format("mode")
    header += "{:<{w}}".format("ceu_syn%", w=col_w)
    header += "{:<{w}}".format("syn_grp%", w=col_w)
    for k in pred_keys:
        header += "{:<{w}}".format(k + "_acc", w=col_w)
        header += "{:<{w}}".format(k + "_loss", w=col_w)

    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for mode in STATE_DICT_KEYS:
        if mode not in results:
            print("{:<14}".format(mode) + "  (skipped)")
            continue
        metrics = results[mode]
        ceu = ceu_results.get(mode)
        row = "{:<14}".format(mode)
        row += "{:<{w}}".format(_fmt_ceu(ceu, "synergy"), w=col_w)
        if ceu and all(k + "_n" in ceu for k in ("synergy", "cue_mod0", "cue_mod1", "coexistence")):
            total_n = sum(ceu[k + "_n"] for k in ("synergy", "cue_mod0", "cue_mod1", "coexistence"))
            syn_pct = "{:.1f}%".format(100.0 * ceu["synergy_n"] / total_n) if total_n > 0 else "-"
        else:
            syn_pct = "-"
        row += "{:<{w}}".format(syn_pct, w=col_w)
        for k in pred_keys:
            row += "{:<{w}}".format(_fmt_acc(metrics, k), w=col_w)
            row += "{:<{w}}".format(_fmt_loss(metrics, k), w=col_w)
        print(row)
    print(sep + "\n")


def main(args):
    checkpoint_path = args.checkpoint
    fold = int(args.fold)
    device = args.device

    print("=" * 70)
    print("Checkpoint: {}".format(os.path.basename(checkpoint_path)))
    print("Fold: {}  Device: {}".format(fold, device))
    print("=" * 70)

    # Build importer — fold is passed so _load_encoder formats encoder dirs
    importer = Importer(
        config_name=args.config,
        default_files=args.default_config,
        device=device,
        fold=fold,
    )

    # Set fold in dataset config (both common locations)
    try:
        importer.config.dataset.data_split.fold = fold
    except Exception:
        pass
    try:
        importer.config.dataset.fold = fold
    except Exception:
        pass

    # Load checkpoint early to check if all results already exist
    print("Loading checkpoint ...")
    importer.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Skip if every present state dict already has a post_test_results entry
    modes_with_sd = [m for m in STATE_DICT_KEYS if "best_model_{}_state_dict".format(m) in importer.checkpoint]
    modes_done    = [m for m in modes_with_sd if "post_test_results_{}".format(m) in importer.checkpoint]
    if set(modes_done) == set(modes_with_sd) and not args.force:
        print("All results already present — skipping (use --force to re-run)")
        results = {m: importer.checkpoint["post_test_results_{}".format(m)] for m in modes_done}
        ceu_results = {m: results[m]["ceu_syn"] for m in modes_done if "ceu_syn" in results[m]}
        print_results_table(results, ceu_results)
        return results

    # Load CEU pkl (test split) if configured
    ceu_folds = load_ceu_pkl(importer.config)
    if ceu_folds is not None:
        print("CEU pkl loaded ({} folds)".format(len(ceu_folds)))
    else:
        print("CEU pkl not found — ceu_syn will be '-'")

    # Build untrained model (encoders are initialised with pretrained weights here)
    print("Building model ...")
    model = importer.get_model(return_model="untrained_model")

    # Build dataloaders
    data_loader = importer.get_dataloaders()

    validator = Validator(
        model=model,
        data_loader=data_loader,
        config=importer.config,
        device=device,
    )

    results = {}
    ceu_results = {}
    for mode in STATE_DICT_KEYS:
        sd_key = "best_model_{}_state_dict".format(mode)
        if sd_key not in importer.checkpoint:
            print("[SKIP] {} not found in checkpoint".format(sd_key))
            continue

        print("\n--- Evaluating: {} ---".format(sd_key))
        model.load_state_dict(reform_state_dict(importer.checkpoint[sd_key]))
        model.to(device).eval()

        test_results = validator.get_results(set="Test", print_results=False)
        results[mode] = test_results

        # Compute CEU synergy
        ceu = compute_ceu_synergy(test_results, ceu_folds, fold)
        if ceu is not None:
            ceu_results[mode] = ceu
            test_results["ceu_syn"] = ceu  # store in results dict too

        # Write back into checkpoint
        result_key = "post_test_results_{}".format(mode)
        importer.checkpoint[result_key] = test_results
        print("Saved {} → checkpoint".format(result_key))

    # Persist updated checkpoint
    torch.save(importer.checkpoint, checkpoint_path)
    print("\nCheckpoint saved: {}".format(checkpoint_path))

    # Summary table
    print_results_table(results, ceu_results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep all 3 best-model checkpoints in a .pth.tar file")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth.tar checkpoint")
    parser.add_argument("--config", required=True, help="Method config JSON path")
    parser.add_argument("--default_config", required=True, help="Default config JSON path")
    parser.add_argument("--fold", required=True, type=int, help="Fold index")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--force", action="store_true", help="Re-run even if results already exist")
    args = parser.parse_args()

    main(args)
