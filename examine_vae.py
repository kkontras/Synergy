from utils.config import setup_logger
from posthoc.Helpers.Helper_Importer import Importer
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from openTSNE import TSNE
from colorama import Fore
from itertools import combinations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from scipy.spatial import procrustes
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


from posthoc.Helpers.Helper_Validator import Validator

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

def print_tsne(config_path, default_config_path, args):
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
    if getattr(args, "no_printing", False):
        args.printing = False

    importer.config.model.save_dir = importer.config.model.save_dir.format(m)

    try:
        importer.load_checkpoint()
    except Exception as e:
        print(f"❌ Could not load: {importer.config.model.save_dir}")
        return {}, {}


    dataloaders = importer.get_dataloaders()
    best_model= importer.get_model(return_model="best_model")


    validator = Validator(model=best_model, data_loader=dataloaders, config=importer.config, device="cuda:0")
    features, targets = validator.get_features(set="Validation")
    plot_tsne_grid(features["z1"], features["z2"], features["~z1"], features["~z1"], targets)
    # --- Usage example (place after t-SNE) ---
    feature_sets = {
        "z1": features["z1"],
        "z2": features["z2"],
        "z1_tilde": features["~z1"],
        "z2_tilde": features["~z2"]
    }
    results = compare_feature_spaces(feature_sets)
    print_comparison_results(results)

@dataclass
class FeatureComparisonResult:
    pair: str
    dist_corr: float
    cka: float
    cosine_sim: float
    cca_mean_corr: float
    procrustes_disp: float

def compute_pairwise_dist_corr(X, Y) -> float:
    return pearsonr(pdist(X), pdist(Y))[0]

def compute_cka(X, Y) -> float:
    Kx, Ky = linear_kernel(X), linear_kernel(Y)
    Kx -= Kx.mean(axis=0, keepdims=True)
    Ky -= Ky.mean(axis=0, keepdims=True)
    return np.sum(Kx * Ky) / (np.linalg.norm(Kx, 'fro') * np.linalg.norm(Ky, 'fro'))

def compute_mean_cosine_similarity(X, Y) -> float:
    return cosine_similarity(X, Y).mean()

def compute_cca_mean_corr(X, Y, n_components=10) -> float:
    n = min(n_components, X.shape[1], Y.shape[1])
    cca = CCA(n_components=n).fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    return np.mean([np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n)])

def compute_procrustes_disp(X, Y) -> float:
    _, _, disparity = procrustes(X, Y)
    return disparity

def compare_feature_spaces(features: dict) -> list[FeatureComparisonResult]:
    results = []
    for a, b in combinations(features.keys(), 2):
        Xa, Xb = features[a], features[b]
        result = FeatureComparisonResult(
            pair=f"{a} vs {b}",
            dist_corr=compute_pairwise_dist_corr(Xa, Xb),
            cka=compute_cka(Xa, Xb),
            cosine_sim=compute_mean_cosine_similarity(Xa, Xb),
            cca_mean_corr=compute_cca_mean_corr(Xa, Xb),
            procrustes_disp=compute_procrustes_disp(Xa, Xb)
        )
        results.append(result)
    return results

def print_comparison_results(results: list[FeatureComparisonResult]):
    header = f"{'Pair':<22} {'DistCorr':>9} {'CKA':>8} {'CosSim':>8} {'CCA':>8} {'Procrustes':>11}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r.pair:<22} {r.dist_corr:9.3f} {r.cka:8.3f} {r.cosine_sim:8.3f} {r.cca_mean_corr:8.3f} {r.procrustes_disp:11.2e}")
    print("-" * len(header) + "\n")



def plot_tsne_grid(
    z1: np.ndarray,
    z2: np.ndarray,
    z1_tilde: np.ndarray,
    z2_tilde: np.ndarray,
    y: np.ndarray,
    save_path: str or None = None,
    perplexity: float = 20.0,
    random_state: int = 54,
    separate_tnse: bool = False,
) -> None:
    """
    Plot 2×2 t-SNE grids for (z₁, z₂, ẑ₁, ẑ₂),
    where t-SNE is fit only on z₁ and z₂, and ẑ₁, ẑ₂ are projected
    into the same learned 2D manifolds.

    Parameters
    ----------
    z1, z2 : np.ndarray
        Encoder features of shape (N, D).
    z1_tilde, z2_tilde : np.ndarray
        Autoencoder reconstructions of shape (N, D).
    y : np.ndarray
        Target labels of shape (N,).
    save_path : str | None
        Optional file path to save the figure. If None, the plot is shown interactively.
    perplexity : float
        Perplexity parameter for t-SNE.
    random_state : int
        Random seed for reproducibility.
    """
    # ----- Setup -----
    print(Fore.CYAN + "🔹 Fitting t-SNE on z₁..." + Fore.RESET)
    tsne_z1_model = TSNE(
        n_components=2,
        perplexity=perplexity,
        initialization="pca",
        metric="euclidean",
        negative_gradient_method="fft",
        n_jobs=8,
        random_state=random_state,
    )
    embedding_z1 = tsne_z1_model.fit(z1)  # returns a TSNEEmbedding object
    z1_emb = np.asarray(embedding_z1)
    z1_tilde_emb = embedding_z1.transform(z1_tilde)

    if separate_tnse:
        tsne_tilde_z1_model = TSNE(
            n_components=2,
            perplexity=perplexity,
            initialization="pca",
            metric="euclidean",
            negative_gradient_method="fft",
            n_jobs=8,
            random_state=random_state,
        )
        embedding_tilde_z1 = tsne_tilde_z1_model.fit(z1_tilde)  # returns a TSNEEmbedding object
        z1_tilde_emb = np.asarray(embedding_tilde_z1)

    print(Fore.CYAN + "🔹 Fitting t-SNE on z₂..." + Fore.RESET)
    tsne_z2_model = TSNE(
        n_components=2,
        perplexity=perplexity,
        initialization="pca",
        metric="euclidean",
        negative_gradient_method="fft",
        n_jobs=8,
        random_state=random_state,
    )
    embedding_z2 = tsne_z2_model.fit(z2)  # returns a TSNEEmbedding object
    z2_emb = np.asarray(embedding_z2)
    z2_tilde_emb = embedding_z2.transform(z2_tilde)

    if separate_tnse:
        tsne_z2_tilede_model = TSNE(
            n_components=2,
            perplexity=perplexity,
            initialization="pca",
            metric="euclidean",
            negative_gradient_method="fft",
            n_jobs=8,
            random_state=random_state,
        )
        embedding_tilde_z2 = tsne_z2_tilede_model.fit(z2_tilde)  # returns a TSNEEmbedding object
        z2_tilde_emb = np.asarray(embedding_tilde_z2)


    # ----- Colormap -----
    okabe_ito_colors = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999"
    ]
    unique_labels = np.unique(y)
    color_list = (okabe_ito_colors * ((len(unique_labels) // len(okabe_ito_colors)) + 1))[:len(unique_labels)]
    cmap = ListedColormap(color_list)

    # ----- Plotting -----
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(right=0.88)
    axs = axs.ravel()
    titles = [
        "z₁ (Encoder 1)",
        "z₂ (Encoder 2)",
        "ẑ₁ (Autoencoder 1, projected in z₁ space)",
        "ẑ₂ (Autoencoder 2, projected in z₂ space)",
    ]
    embeddings = [z1_emb, z2_emb, z1_tilde_emb, z2_tilde_emb]

    for ax, emb, title in zip(axs, embeddings, titles):
        ax.scatter(
            emb[:, 0], emb[:, 1],
            c=y, cmap=cmap, s=10, alpha=0.95, linewidths=0
        )
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("white")

    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(
    #     handles, labels, loc="upper center",
    #     ncol=min(8, len(unique_labels)), frameon=False,
    #     fontsize=10, title="Task labels (y)"
    # )

    fig.suptitle(
        "t-SNE of Feature Representations (ẑ projected in z manifold)",
        fontsize=15, weight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])


    # ----- Save or Show -----
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(Fore.GREEN + f"Saved t-SNE plot to {save_path}" + Fore.RESET)
    else:
        plt.show()

    feature_sets = {
        "z1": z1_emb,
        "z2": z2_emb,
        "z1_tilde": z1_tilde_emb,
        "z2_tilde": z2_tilde_emb
    }




if __name__ == "__main__":
    args = get_args()
    args.fold = 0
    print_tsne(config_path=args.config, default_config_path=args.default_config, args=args)

    lp_results = run_linear_probing(feature_sets, y)
    print_linear_probe_results(lp_results)


