import argparse
import numpy as np
import warnings
from dataclasses import dataclass
from itertools import combinations

from colorama import Fore
from openTSNE import TSNE
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TkAgg")

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from scipy.spatial import procrustes

# -------------------------------------------------------
# Project imports
# -------------------------------------------------------
from utils.config import setup_logger
from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator

import os
import numpy as np
# -------------------------------------------------------
# Feature comparison metrics
# -------------------------------------------------------

@dataclass
class FeatureComparisonResult:
    pair: str
    dist_corr: float
    cka: float
    cosine_sim: float
    cca_mean_corr: float
    procrustes_disp: float


def compute_pairwise_dist_corr(X, Y):
    return pearsonr(pdist(X), pdist(Y))[0]


def compute_cka(X, Y):
    Kx, Ky = linear_kernel(X), linear_kernel(Y)
    Kx -= Kx.mean(axis=0, keepdims=True)
    Ky -= Ky.mean(axis=0, keepdims=True)
    return np.sum(Kx * Ky) / (np.linalg.norm(Kx, 'fro') * np.linalg.norm(Ky, 'fro'))


def compute_cosine_mean(X, Y):
    return cosine_similarity(X, Y).mean()


def compute_cca_mean_corr(X, Y, n_components=10):
    n = min(n_components, X.shape[1], Y.shape[1])
    cca = CCA(n_components=n).fit(X, Y)
    Xc, Yc = cca.transform(X, Y)
    return np.mean([np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(n)])


def compute_procrustes(X, Y):
    _, _, disp = procrustes(X, Y)
    return disp


def compare_feature_spaces(features: dict):
    results = []
    for a, b in combinations(features.keys(), 2):
        Xa, Xb = features[a], features[b]
        results.append(FeatureComparisonResult(
            pair=f"{a} vs {b}",
            dist_corr=compute_pairwise_dist_corr(Xa, Xb),
            cka=compute_cka(Xa, Xb),
            cosine_sim=compute_cosine_mean(Xa, Xb),
            cca_mean_corr=compute_cca_mean_corr(Xa, Xb),
            procrustes_disp=compute_procrustes(Xa, Xb)
        ))
    return results


def print_feature_comparisons(results):
    header = f"{'Pair':<22} {'DistCorr':>9} {'CKA':>8} {'CosSim':>8} {'CCA':>8} {'Procrustes':>11}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r.pair:<22} {r.dist_corr:9.3f} {r.cka:8.3f} {r.cosine_sim:8.3f} {r.cca_mean_corr:8.3f} {r.procrustes_disp:11.2e}")
    print("-" * len(header))


# -------------------------------------------------------
# Linear probing (train on train, eval on val)
# -------------------------------------------------------

@dataclass
class LinearProbeResult:
    name: str
    train_acc: float
    val_acc: float



def save_embeddings(path, features, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **features, labels=labels)


def load_embeddings(path):
    data = np.load(path)
    labels = data["labels"]
    features = {k: data[k] for k in data.files if k != "labels"}
    return features, labels


def evaluate_linear_probe(train_X, train_y, val_X, val_y):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    clf = LogisticRegression(max_iter=2000)

    clf.fit(train_X, train_y)

    train_preds = clf.predict(train_X)
    val_preds   = clf.predict(val_X)

    train_acc = (train_preds == train_y).mean()
    val_acc   = (val_preds   == val_y).mean()

    return clf, train_preds, val_preds, train_acc, val_acc

def run_linear_probing(train_dict, val_dict, y_train, y_val):
    results = []
    probes = {}

    for name in train_dict.keys():
        clf, tr_preds, val_preds, tr_acc, val_acc = \
            evaluate_linear_probe(train_dict[name], y_train, val_dict[name], y_val)

        probes[name] = {
            "clf": clf,
            "train_preds": tr_preds,
            "val_preds": val_preds
        }

        results.append(LinearProbeResult(name, tr_acc, val_acc))

    return results, probes

def compute_correct_intersections_from_preds(probes, y_val):
    n_val = len(y_val)

    correct = {
        name: set(np.where(info["val_preds"] == y_val)[0])
        for name, info in probes.items()
    }

    intersections = {
        "z1 ∩ z2": correct["z1"] & correct["z2"],
        "z1 ∩ z1_tilde": correct["z1"] & correct["z1_tilde"],
        "z2 ∩ z2_tilde": correct["z2"] & correct["z2_tilde"],
        "z1_tilde ∩ z2_tilde": correct["z1_tilde"] & correct["z2_tilde"],
        "z1 ∩ z2_tilde": correct["z1"] & correct["z2_tilde"],
        "z1_tilde ∩ z2": correct["z1_tilde"] & correct["z2"],
        "all 4": correct["z1"] & correct["z2"] & correct["z1_tilde"] & correct["z2_tilde"],
    }

    return {"correct": correct, "intersections": intersections, "n_val": n_val}


def print_linear_probe_results(results):
    header = f"{'Embedding':<15} {'Train Acc':>10} {'Val Acc':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:<15} {r.train_acc:10.3f} {r.val_acc:10.3f}")
    print("-" * len(header))

def print_correct_intersections(res):
    n_val = res["n_val"]

    print("\n=== Correct Prediction Rates (per embedding) ===")
    for name, corr in res["correct"].items():
        cnt = len(corr)
        pct = 100.0 * cnt / n_val
        print(f"{name}: {cnt:4d} correct ({pct:6.2f}%)")

    print("\n=== Shared Correct Predictions (intersections) ===")
    for name, inter in res["intersections"].items():
        cnt = len(inter)
        pct = 100.0 * cnt / n_val
        print(f"{name}: {cnt:4d} samples ({pct:6.2f}%)")


# -------------------------------------------------------
# t-SNE visualization
# -------------------------------------------------------

def tsne_plot(z1, z2, z1_tilde, z2_tilde, y, perplexity=30, separate_tsne=True):
    print(Fore.CYAN + "\n🔹 Running t-SNE…" + Fore.RESET)

    ts1 = TSNE(n_components=2, perplexity=perplexity).fit(z1)
    ts2 = TSNE(n_components=2, perplexity=perplexity).fit(z2)

    z1_e = np.asarray(ts1)
    z2_e = np.asarray(ts2)

    if separate_tsne:
        z1_t_e = TSNE(n_components=2, perplexity=perplexity).fit(z1_tilde)
        z2_t_e = TSNE(n_components=2, perplexity=perplexity).fit(z2_tilde)
    else:
        z1_t_e = ts1.transform(z1_tilde)
        z2_t_e = ts2.transform(z2_tilde)

    cmap = ListedColormap([
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999"
    ])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    labels = [
        "z₁",
        "z₂",
        "ẑ₁ (separate t-SNE)" if separate_tsne else "ẑ₁ → z₁ t-SNE",
        "ẑ₂ (separate t-SNE)" if separate_tsne else "ẑ₂ → z₂ t-SNE"
    ]

    data = [z1_e, z2_e, z1_t_e, z2_t_e]

    for ax, emb, title in zip(axs.ravel(), data, labels):
        ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap, s=8)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("t-SNE embeddings", fontsize=16)
    plt.tight_layout()
    plt.show()



# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
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
    parser.add_argument('--no_printing', action='store_true', help="Disable printing results")
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

def main():
    args = get_args()
    args.fold = 0

    setup_logger()
    # Load config + checkpoint
    importer = Importer(
        config_name=args.config,
        default_files=args.default_config,
        device="cuda:0"
    )

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
    if getattr(args, "no_printing", False):
        args.printing = False

    importer.config.model.save_dir = importer.config.model.save_dir.format(m)

    exp_name = os.path.basename(importer.config.model.save_dir)
    exp_dir = f"./experiments/{exp_name}"
    train_path = os.path.join(exp_dir, "embeddings_train.npz")
    val_path   = os.path.join(exp_dir, "embeddings_val.npz")

    try:
        importer.load_checkpoint()
    except:
        print(Fore.RED + f"❌ Could not load checkpoint: {importer.config.model.save_dir}" + Fore.RESET)
        return

    dataloaders = importer.get_dataloaders()
    model = importer.get_model(return_model="best_model")

    validator = Validator(
        model=model,
        data_loader=dataloaders,
        config=importer.config,
        device="cuda:0"
    )

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(Fore.YELLOW + "Loading cached embeddings..." + Fore.RESET)
        features_train, y_train = load_embeddings(train_path)
        features_val,   y_val   = load_embeddings(val_path)

    else:
        print(Fore.GREEN + "Extracting embeddings..." + Fore.RESET)

        features_train, y_train = validator.get_features(set="Train")
        features_val,   y_val   = validator.get_features(set="Validation")

        os.makedirs(exp_dir, exist_ok=True)
        save_embeddings(train_path, features_train, y_train)
        save_embeddings(val_path,   features_val,   y_val)

    train_dict = {
        "z1":        features_train["z1"],
        "z2":        features_train["z2"],
        "z1_tilde":  features_train["~z1"],
        "z2_tilde":  features_train["~z2"]
    }
    val_dict = {
        "z1":        features_val["z1"],
        "z2":        features_val["z2"],
        "z1_tilde":  features_val["~z1"],
        "z2_tilde":  features_val["~z2"]
    }


    # features_val = { k: v.mean(axis=-1) for k, v in features_val.items() if len(v.shape) == 3 else v }
    # train_dict = { k: v.mean(axis=-1) for k, v in train_dict.items() }
    # val_dict = { k: v.mean(axis=-1) for k, v in val_dict.items() }
    features_val = { k: (v.mean(axis=-1) if len(v.shape) == 3 else v) for k, v in features_val.items() }
    train_dict = { k: (v.mean(axis=-1) if len(v.shape) == 3 else v) for k, v in train_dict.items() }
    val_dict = { k: (v.mean(axis=-1) if len(v.shape) == 3 else v) for k, v in val_dict.items() }

    tsne_plot(
        features_val["z1"], features_val["z2"],
        features_val["~z1"], features_val["~z2"],
        y_val
    )

    print("\n=== Feature Space Comparisons (Validation) ===")
    comp = compare_feature_spaces(val_dict)
    print_feature_comparisons(comp)

    print("\n=== Linear Probing (Train→Val) ===")
    lp_results, probes = run_linear_probing(train_dict, val_dict, y_train, y_val)
    print_linear_probe_results(lp_results)

    print("\n=== Shared Correct Predictions ===")
    corr_res = compute_correct_intersections_from_preds(probes, y_val)
    print_correct_intersections(corr_res)



if __name__ == "__main__":
    main()
