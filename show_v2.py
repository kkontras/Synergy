from colorama import Fore
from utils.config import setup_logger
from posthoc.Helpers.Helper_Importer import Importer
import numpy as np
import argparse
from collections import defaultdict


def get_args():
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser(description="Model evaluation utility")

    # main configs
    parser.add_argument('--config', help="Path to config file(s), comma-separated")
    parser.add_argument('--default_config', help="Path to default config file")

    # optional parameters (extend as needed)
    arg_keys = [
        "fold", "alpha", "validate_with", "transform_type", "trasform_before",
        "tanh_mode", "tanh_mode_beta", "regby", "clip", "batch_size", "l",
        "multil", "l_diffsq", "lib", "ratio_us", "kmepoch", "num_samples",
        "pow", "nstep", "contrcoeff", "kde_coeff", "etube", "temperature",
        "contr_type", "shuffle_type", "num_classes", "base_alpha", "alpha_var",
        "base_beta", "beta_var", "ilr_c", "ilr_g", "mmcosine_scaling",
        "ending_epoch", "recon_weight1", "recon_weight2", "recon_epochstages",
        "recon_ensemblestages", "perturb", "lr", "wd", "mm", "cls"
    ]
    for arg in arg_keys:
        parser.add_argument(f'--{arg}', required=False)

    # flags
    parser.add_argument('--no-printing', action='store_true', help="Return acc/std for bash parsing")
    parser.add_argument('--tdqm_disable', action='store_true')
    parser.add_argument('--pre', action='store_true', default=False)
    parser.add_argument('--frozen', action='store_true', default=False)
    parser.add_argument('--start_over', action='store_true', default=False)

    return parser.parse_args()


def append_arg_suffix(m, args, key):
    val = getattr(args, key, None)
    if "fold" in key and val is not None:
        m += f"fold{val}"
        return m
    if val not in [None, "None"]:
        m += f"_{key}{val}"
    return m


def format_metrics(val_metrics, test_metric):
    message = ""
    if "current_epoch" in val_metrics:
        message += Fore.GREEN + f"Epoch: {val_metrics['current_epoch']}  "
    if "steps_no_improve" in val_metrics:
        message += Fore.YELLOW + f"No improve: {val_metrics['steps_no_improve']}  "

    if "acc" in val_metrics and "combined" in val_metrics["acc"]:
        acc = val_metrics["acc"]["combined"] * 100
        message += Fore.CYAN + f"Val_Acc: {acc:.1f}  "

    if test_metric and "acc" in test_metric and "combined" in test_metric["acc"]:
        test_acc = test_metric["acc"]["combined"] * 100
        message += Fore.MAGENTA + f"Test_Acc: {test_acc:.1f}  "

    if "synergy_gap_uni" in test_metric:
        message += Fore.LIGHTBLUE_EX + f"SyG_Uni: {test_metric['synergy_gap_uni']:.2f}  "
    if "synergy_gap_ens" in test_metric:
        message += Fore.LIGHTBLUE_EX + f"SyG_Ens: {test_metric['synergy_gap_ens']:.2f}  "

    return message + Fore.RESET


def print_search(config_path, default_config_path, args):
    setup_logger()
    importer = Importer(config_name=config_path, default_files=default_config_path, device="cuda:0")

    # build suffix
    keys_to_append = [
        "fold", "alpha", "recon_weight1", "recon_weight2", "recon_epochstages",
        "recon_ensemblestages", "tanh_mode", "num_classes", "tanh_mode_beta",
        "transform_type", "trasform_before", "regby", "clip", "l", "multil",
        "l_diffsq", "lib", "ratio_us", "kmepoch", "mmcosine_scaling",
        "ending_epoch", "num_samples", "pow", "nstep", "contrcoeff",
        "kde_coeff", "etube", "temperature", "shuffle_type", "contr_type",
        "validate_with", "base_alpha", "alpha_var", "base_beta", "beta_var", "perturb",
        "lr", "wd", "mm", "cls", "batch_size"
    ]

    m = ""
    for k in keys_to_append:
        m = append_arg_suffix(m, args, k)
    if getattr(args, "ilr_c", None) and getattr(args, "ilr_g", None):
        m += f"_ilrcg{args.ilr_c}_{args.ilr_g}"
    if getattr(args, "pre", False):
        m += "_pre"
    if getattr(args, "frozen", False):
        m += "_frozen"

    importer.config.model.save_dir = importer.config.model.save_dir.format(m)

    try:
        importer.load_checkpoint()
    except Exception:
        if not args.no_printing:
            print(f"❌ Could not load: {importer.config.model.save_dir}")
        return {}, {}

    val_metrics, test_metric = importer.print_progress(
        multi_fold_results={}, verbose=False, latex_version=False
    )

    if not args.no_printing:
        print(Fore.WHITE + importer.config.model.save_dir.split("/")[-1])
        print(format_metrics(val_metrics, test_metric))

    return val_metrics, test_metric


def print_mean(m: dict, val=True, printing=True):
    agg = {}
    for fold in m:
        for metric, val_metric in m[fold].items():
            if isinstance(val_metric, dict):
                agg.setdefault(metric, defaultdict(list))
                for pred, value in val_metric.items():
                    agg[metric][pred].append(value)
            else:
                agg.setdefault(metric, []).append(val_metric)

    color = Fore.RED if val else Fore.GREEN
    message = color + ("Val " if val else "Test ") + Fore.RESET

    mean_acc = std_acc = None

    if "acc" in agg and "combined" in agg["acc"]:
        mean_acc = np.mean(agg["acc"]["combined"])
        std_acc = np.std(agg["acc"]["combined"])
        message += Fore.CYAN + f"Acc: {mean_acc*100:.1f} ± {std_acc*100:.1f} "

    if "synergy_gap_uni" in agg:
        mean_uni = np.mean(agg["synergy_gap_uni"])
        std_uni = np.std(agg["synergy_gap_uni"])
        message += Fore.LIGHTBLUE_EX + f"SyG_Uni: {mean_uni:.2f} ± {std_uni:.2f}  "

    if "synergy_gap_ens" in agg:
        mean_ens = np.mean(agg["synergy_gap_ens"])
        std_ens = np.std(agg["synergy_gap_ens"])
        message += Fore.LIGHTBLUE_EX + f"SyG_Ens: {mean_ens:.2f} ± {std_ens:.2f}  "

    if printing:
        print(message)

    return mean_acc, std_acc


if __name__ == "__main__":
    args = get_args()
    configs = args.config.split(",")
    val, test = {}, {}

    for i in range(3):
        args.fold = i
        v, t = print_search(config_path=args.config, default_config_path=args.default_config, args=args)
        val[i] = v
        test[i] = t

    try:
        mean_val, std_val = print_mean(val, val=True, printing=not args.no_printing)
        mean_test, std_test = print_mean(test, val=False, printing=not args.no_printing)

        if args.no_printing:
            # Output for bash capture: "<mean_acc> <std_acc>"
            print(f"{mean_test*100:.2f} {std_test*100:.2f}")
        else:
            print(f"Final Test: {mean_test*100:.1f} ± {std_test*100:.1f}")
    except Exception as e:
        if not args.no_printing:
            print("⚠️ Error computing mean:", e)
