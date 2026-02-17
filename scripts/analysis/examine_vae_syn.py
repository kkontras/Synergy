
#!/usr/bin/env python3
"""
t-SNE visualization for original vs reconstructed representations.

What it does
------------
- Loads a trained model checkpoint via your project's Importer (same pattern as your eval utilities).
- Uses Validator.get_features(...) to fetch:
    z1, z2, ~z1, ~z2 (or z1_tilde/z2_tilde)
- Produces a 2×1 plot:
    Row 1: z1 vs z1_tilde (overlayed in same 2D space)
    Row 2: z2 vs z2_tilde (overlayed in same 2D space)
  Colors = labels, Marker shape = original vs reconstructed

t-SNE backend
-------------
- Prefers openTSNE (supports transform/projection).
- Falls back to scikit-learn TSNE by fitting on the concatenation [z; z_tilde].

Usage
-----
python tsne_recon_vs_orig.py --config <cfg.json> --default_config <default.json> --fold 0 \
    --set Validation --perplexity 30 --save_path tsne_fold0.png

Notes
-----
- This file assumes your repo provides: Importer, Validator, setup_logger.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from models.Synergy_Models_SVAE import *
# -------------------------
# Optional dependencies
# -------------------------
_HAS_OPENTSNE = False
try:
    from openTSNE import TSNE as OpenTSNE
    _HAS_OPENTSNE = True
except Exception:
    _HAS_OPENTSNE = False

try:
    from sklearn.manifold import TSNE as SkTSNE
except Exception as e:
    SkTSNE = None



def _format_with_fold(s: str, fold: int) -> str:
    """
    Format strings like '...fold{}.pth.tar' or '...{fold}...' with the fold value.
    Safe: if no placeholder exists, returns s unchanged.
    """
    if not isinstance(s, str):
        return s
    if "{fold}" in s:
        try:
            return s.format(fold=fold)
        except Exception:
            return s
    if "{}" in s:
        # most common pattern in this repo: '...fold{}.pth.tar'
        try:
            return s.format(fold)
        except Exception:
            # sometimes they want 'fold0' as the inserted token
            try:
                return s.format(f"fold{fold}")
            except Exception:
                return s
    return s


# -------------------------
# Project imports (your codebase)
# -------------------------
def _import_project_symbols():
    """
    Import project-specific utilities lazily so the script can be copied around.
    Edit these imports if your project uses different module paths.
    """
    from utils.config import setup_logger  # type: ignore
    from posthoc.Helpers.Helper_Validator import Validator      # type: ignore
    from posthoc.Helpers.Helper_Importer import Importer      # type: ignore
    return setup_logger, Importer, Validator


# -------------------------
# CLI
# -------------------------
def get_args():
    p = argparse.ArgumentParser("t-SNE: original vs reconstructed z1/z2")
    p.add_argument("--config", required=True, help="Path to config file")
    p.add_argument("--default_config", required=True, help="Path to default config file")
    p.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0 or cpu")
    p.add_argument("--fold", type=int, default=0, help="Data split fold")
    p.add_argument("--set", default="Validation", choices=["Train", "Validation", "Test"],
                   help="Dataset split to extract features from")
    p.add_argument("--perplexity", type=float, default=20.0, help="t-SNE perplexity")
    p.add_argument("--random_state", type=int, default=54, help="Random seed")
    p.add_argument("--n_jobs", type=int, default=8, help="Jobs for openTSNE (if used)")
    p.add_argument("--separate_tsne", action="store_true",
                   help="If set, fit separate t-SNE on tilde instead of projecting (openTSNE only).")
    p.add_argument("--max_points", type=int, default=5000,
                   help="Subsample to at most this many points (for speed). 0 disables subsampling.")
    p.add_argument("--save_path", default=None, help="If provided, saves the figure to this path")
    return p.parse_args()


# -------------------------
# Feature utilities
# -------------------------
def _as_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    # torch tensor
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _fetch_key(features: dict, *candidates: str):
    for k in candidates:
        if k in features:
            return features[k]
    return None


def _maybe_subsample(arrays, y, max_points: int, seed: int):
    if max_points is None or max_points <= 0:
        return arrays, y
    n = y.shape[0]
    if n <= max_points:
        return arrays, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    out_arrays = [a[idx] for a in arrays]
    return out_arrays, y[idx]


# -------------------------
# t-SNE core
# -------------------------
def _fit_embed_opentsne(X, X_tilde, perplexity, random_state, n_jobs, separate_tsne):
    # Fit on X, project X_tilde (unless separate_tsne=True)
    tsne = OpenTSNE(
        n_components=2,
        perplexity=perplexity,
        initialization="pca",
        metric="euclidean",
        negative_gradient_method="fft",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    emb = tsne.fit(X)
    X_emb = np.asarray(emb)
    if separate_tsne:
        tsne2 = OpenTSNE(
            n_components=2,
            perplexity=perplexity,
            initialization="pca",
            metric="euclidean",
            negative_gradient_method="fft",
            n_jobs=n_jobs,
            random_state=random_state,
        )
        X_tilde_emb = np.asarray(tsne2.fit(X_tilde))
    else:
        X_tilde_emb = emb.transform(X_tilde)
    return X_emb, X_tilde_emb


def _fit_embed_sklearn_concat(X, X_tilde, perplexity, random_state):
    if SkTSNE is None:
        raise RuntimeError("scikit-learn TSNE not available and openTSNE not available. Install one of them.")
    Z = np.concatenate([X, X_tilde], axis=0)
    tsne = SkTSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    Z_emb = tsne.fit_transform(Z)
    X_emb = Z_emb[: X.shape[0]]
    X_tilde_emb = Z_emb[X.shape[0] :]
    return X_emb, X_tilde_emb


def fit_embeddings(X, X_tilde, perplexity, random_state, n_jobs, separate_tsne):
    if _HAS_OPENTSNE:
        return _fit_embed_opentsne(X, X_tilde, perplexity, random_state, n_jobs, separate_tsne)
    return _fit_embed_sklearn_concat(X, X_tilde, perplexity, random_state)


# -------------------------
# Plotting
# -------------------------
def _okabe_ito_cmap(n_labels: int):
    okabe_ito = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999"
    ]
    colors = (okabe_ito * ((n_labels // len(okabe_ito)) + 1))[:n_labels]
    return ListedColormap(colors)


def plot_tsne_overlay(
    X_emb, X_tilde_emb, y,
    title: str,
    ax,
    cmap,
    s=10,
    alpha_orig=0.9,
    alpha_tilde=0.6,
):
    # original: circles
    sc1 = ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap=cmap, s=s, alpha=alpha_orig, linewidths=0, marker="o")
    # reconstructed: triangles
    ax.scatter(X_tilde_emb[:, 0], X_tilde_emb[:, 1], c=y, cmap=cmap, s=s, alpha=alpha_tilde, linewidths=0, marker="^")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    return sc1

def plot_tsne_4blocks(
    z1_emb, z1_tilde_emb,
    z2_emb, z2_tilde_emb,
    y,
    axs,
    cmap,
    s=10,
    alpha_orig=0.9,
    alpha_tilde=0.75,
    share_limits_per_col=True,
):
    """
    2x2 layout:
      [0,0] z1 original     [0,1] z2 original
      [1,0] z1 reconstructed [1,1] z2 reconstructed

    axs must be a (2,2) array of matplotlib axes (from plt.subplots(2,2)).
    """

    def _plot(ax, X, title, marker, alpha):
        sc = ax.scatter(
            X[:, 0], X[:, 1],
            c=y, cmap=cmap, s=s, alpha=alpha, linewidths=0, marker=marker
        )
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        return sc

    # Top row: originals
    sc = _plot(axs[0, 0], z1_emb, "z1 (original)", marker="o", alpha=alpha_orig)
    _plot(axs[0, 1], z2_emb, "z2 (original)", marker="o", alpha=alpha_orig)

    # Bottom row: reconstructed
    _plot(axs[1, 0], z1_tilde_emb, "z1~ (reconstructed)", marker="^", alpha=alpha_tilde)
    _plot(axs[1, 1], z2_tilde_emb, "z2~ (reconstructed)", marker="^", alpha=alpha_tilde)

    if share_limits_per_col:
        for col in (0, 1):
            x0 = axs[0, col].get_xlim(); x1 = axs[1, col].get_xlim()
            y0 = axs[0, col].get_ylim(); y1 = axs[1, col].get_ylim()
            xlim = (min(x0[0], x1[0]), max(x0[1], x1[1]))
            ylim = (min(y0[0], y1[0]), max(y0[1], y1[1]))
            axs[0, col].set_xlim(xlim); axs[1, col].set_xlim(xlim)
            axs[0, col].set_ylim(ylim); axs[1, col].set_ylim(ylim)

    return sc


def fit_on_train_project_val(X_tr, X_val, Xtilde_val, perplexity, random_state, n_jobs):
    if not _HAS_OPENTSNE:
        raise RuntimeError("Train-fit + Val-project requires openTSNE (sklearn TSNE cannot transform).")

    tsne = OpenTSNE(
        n_components=2,
        perplexity=perplexity,
        initialization="pca",
        metric="euclidean",
        negative_gradient_method="fft",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    emb_tr = tsne.fit(X_tr)                   # fit on Train
    X_val_emb = emb_tr.transform(X_val)       # project Val original
    Xtilde_val_emb = emb_tr.transform(Xtilde_val)  # project Val recon
    return np.asarray(X_val_emb), np.asarray(Xtilde_val_emb)


def main():
    args = get_args()
    setup_logger, Importer, Validator = _import_project_symbols()
    setup_logger()

    importer = Importer(
        config_name=args.config,
        default_files=args.default_config,
        device=args.device,
    )

    # set fold (if your config supports it)
    try:
        importer.config.dataset.data_split.fold = args.fold
    except Exception:
        pass

    # Format any fold placeholders in save_dir and encoder pretrained paths
    try:
        importer.config.model.save_dir = _format_with_fold(getattr(importer.config.model, "save_dir", ""), args.fold)
    except Exception:
        pass

    # importer.config.model.save_dir = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/SynIB_VAE_synergy_foldfold0_l1_lr0.0001_wd0.0001.pth.tar"

    # Some configs also put '{}' placeholders inside pretrainedEncoder.dir; format them too
    try:
        enc_list = getattr(importer.config.model, "encoders", None)
        if enc_list is not None:
            for enc in enc_list:
                pe = enc.get("pretrainedEncoder", None) if isinstance(enc, dict) else None
                if pe and pe.get("use", False) and isinstance(pe.get("dir", None), str):
                    pe["dir"] = _format_with_fold(pe["dir"], args.fold)
    except Exception:
        pass

    # try:
    #     importer.load_checkpoint()
    # except Exception as e:
    #     print(f"❌ Could not load checkpoint for config={args.config}. Error: {e}")
    #     raise

    dataloaders = importer.get_dataloaders()
    best_model = importer.get_model(return_model="untrained_model")

    validator = Validator(model=best_model, data_loader=dataloaders, config=importer.config, device=args.device)
    features_va, targets_va = validator.get_features(set=args.set)
    z1_va = _as_numpy(_fetch_key(features_va, "ez1"))
    z2_va = _as_numpy(_fetch_key(features_va, "ez2"))
    z1t_va = _as_numpy(_fetch_key(features_va, "~z1", "z1_tilde", "tz1", "z1_recon"))
    z2t_va = _as_numpy(_fetch_key(features_va, "~z2", "z2_tilde", "tz2", "z2_recon"))
    y_va = _as_numpy(_fetch_key(features_va, "ey")).astype(int)

    # y_va = _as_numpy(targets_va).astype(int)

    features_tr, targets_tr = validator.get_features(set="Train")
    z1_tr = _as_numpy(_fetch_key(features_tr, "ez1"))
    z2_tr = _as_numpy(_fetch_key(features_tr, "ez2"))
    z1t_tr = _as_numpy(_fetch_key(features_tr, "~z1", "z1_tilde", "tz1", "z1_recon"))
    z2t_tr = _as_numpy(_fetch_key(features_tr, "~z2", "z2_tilde", "tz2", "z2_recon"))
    # y_tr = _as_numpy(targets_tr).astype(int)

    # z, zt are (N, D) numpy or torch
    def stats(name, x):
        x = x.detach().cpu() if torch.is_tensor(x) else torch.tensor(x)
        print(name,
              "mean_norm", x.norm(dim=-1).mean().item(),
              "std_mean", x.std(dim=0).mean().item(),
              "std_all", x.std().item())

    stats("z1", z1_tr)
    stats("z1_tilde", z1t_tr)
    print("mse(z1~,z1)", F.mse_loss(torch.tensor(z1t_tr), torch.tensor(z1_tr)).item())
    stats("z2", z2_tr)
    stats("z2_tilde", z2t_tr)
    print("mse(z2~,z2)", F.mse_loss(torch.tensor(z2t_tr), torch.tensor(z2_tr)).item())

    print("🔹 Fit on Train, project Val for z1 ...")
    z1_emb, z1_tilde_emb = fit_on_train_project_val(
        z1_tr, z1_va, z1t_va, args.perplexity, args.random_state, args.n_jobs
    )

    print("🔹 Fit on Train, project Val for z2 ...")
    z2_emb, z2_tilde_emb = fit_on_train_project_val(
        z2_tr, z2_va, z2t_va, args.perplexity, args.random_state, args.n_jobs
    )

    # Plot
    unique_labels = np.unique(y_va)
    cmap = _okabe_ito_cmap(len(unique_labels))

    # fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    # plt.subplots_adjust(hspace=0.22)
    #
    # plot_tsne_overlay(
    #     z1_emb, z1_tilde_emb, y,
    #     title="z1 (original) vs z1~ (reconstructed) — same t-SNE space",
    #     ax=axs[0],
    #     cmap=cmap,
    # )
    # plot_tsne_overlay(
    #     z2_emb, z2_tilde_emb, y,
    #     title="z2 (original) vs z2~ (reconstructed) — same t-SNE space",
    #     ax=axs[1],
    #     cmap=cmap,
    # )
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plot_tsne_4blocks(z1_emb, z1_tilde_emb, z2_emb, z2_tilde_emb, y_va, axs, cmap=cmap)

    # Legend note
    fig.suptitle("t-SNE overlays: circles = original, triangles = reconstructed", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        fig.savefig(args.save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved figure to: {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
