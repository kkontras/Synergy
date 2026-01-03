import os
import sys
import json
import numpy as np
import pickle

# If your project relies on relative imports, you may need to set the working directory.
# The original script did:
# os.chdir('/users/sista/kkontras/Documents/Balance/')
# Keep it commented unless you really need it.
# os.chdir('/path/to/your/repo/root')

from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator

device = "cuda:0"

# -------------------------
# USER-PROVIDED CONFIGS
# -------------------------
DEFAULT_CONFIG_PATH = "configs/CREMA_D/default_config_cremad_res_syn.json"

# Pick ONE of these per run (you said you'll run twice: once audio, once video).
# CONFIG_PATH = "configs/CREMA_D/release/res/unimodal_audio.json"
# CHECKPOINT_TEMPLATE = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_audio_fold{}_lr0.001_wd0.0001.pth.tar"

CONFIG_PATH = "configs/CREMA_D/release/res/unimodal_video.json"
CHECKPOINT_TEMPLATE = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_video_fold{}_lr0.001_wd0.0001.pth.tar"

OUTPUT_JSON = "cremad_ceu_test_res.pkl"

FOLDS = [0, 1, 2]


def _to_jsonable(x):
    """Recursively convert common non-JSON types (np arrays, tensors, etc.) into JSONable python types."""
    # numpy
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64, np.float16)):
        return float(x)
    if isinstance(x, (np.int32, np.int64, np.int16, np.int8)):
        return int(x)

    # torch (optional; don’t import torch just for this)
    try:
        import torch  # noqa: F401

        if "torch" in sys.modules:
            import torch as _torch

            if isinstance(x, _torch.Tensor):
                return x.detach().cpu().tolist()
    except Exception:
        pass

    # dict / list / tuple
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # fallback (includes str, float, int, bool, None)
    return x


def _set_fold_on_config(importer: Importer, fold: int) -> None:
    """
    Mirror the *minimal* fold-related behavior from your original script:
      - set dataset fold fields
      - set seed from the same non-UCF list used before
      - format norm paths if present
      - format pretrained encoder dirs with fold if present
    """
    # Dataset fold
    if hasattr(importer.config, "dataset"):
        if hasattr(importer.config.dataset, "data_split") and hasattr(importer.config.dataset.data_split, "fold"):
            importer.config.dataset.data_split.fold = int(fold)
        if hasattr(importer.config.dataset, "fold"):
            importer.config.dataset.fold = int(fold)

        # Seeds used in the original for non-UCF configs: [109, 19, 337]
        seeds = [109, 19, 337]
        if hasattr(importer.config, "training_params") and hasattr(importer.config.training_params, "seed"):
            importer.config.training_params.seed = int(seeds[int(fold)])
            print("Seed:", importer.config.training_params.seed)

        # Optional dataset paths that were fold-formatted in the original
        try:
            if "norm_wav_path" in importer.config.dataset:
                importer.config.dataset.norm_wav_path = importer.config.dataset.norm_wav_path.format(fold)
        except Exception:
            pass

        try:
            if "norm_face_path" in importer.config.dataset:
                importer.config.dataset.norm_face_path = importer.config.dataset.norm_face_path.format(fold)
        except Exception:
            pass

    # Encoder pretrained dirs fold-formatting (same idea as original)
    if hasattr(importer.config, "model") and hasattr(importer.config.model, "encoders"):
        try:
            for i in range(len(importer.config.model.encoders)):
                enc = importer.config.model.encoders[i]
                if hasattr(enc, "pretrainedEncoder") and hasattr(enc.pretrainedEncoder, "dir"):
                    enc.pretrainedEncoder.dir = enc.pretrainedEncoder.dir.format(fold)
        except Exception:
            pass


def _set_checkpoint_path(importer: Importer, ckpt_path: str) -> None:
    """
    We set multiple likely fields so Importer.load_checkpoint() can find it,
    without changing any other config knobs.
    """
    # Common pattern: config.model.save_dir is used by the loader
    if hasattr(importer.config, "model") and hasattr(importer.config.model, "save_dir"):
        importer.config.model.save_dir = ckpt_path

    # Some codebases store it under a different attribute name; set if present
    if hasattr(importer.config, "model") and hasattr(importer.config.model, "checkpoint_path"):
        importer.config.model.checkpoint_path = ckpt_path

    if hasattr(importer.config, "model") and hasattr(importer.config.model, "ckpt_path"):
        importer.config.model.ckpt_path = ckpt_path


def run_fold(fold: int) -> dict:
    ckpt_path = CHECKPOINT_TEMPLATE.format(fold)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    importer = Importer(
        config_name=CONFIG_PATH,
        default_files=DEFAULT_CONFIG_PATH,
        device=device,
    )

    # Minimal fold logic (mirrors the relevant part of the original script)
    _set_fold_on_config(importer, fold)

    # Keep your testing batch size override (it was hard-set in the original)
    if hasattr(importer.config, "training_params") and hasattr(importer.config.training_params, "test_batch_size"):
        importer.config.training_params.test_batch_size = 6

    # Point Importer to the right checkpoint
    _set_checkpoint_path(importer, ckpt_path)

    # Load + build model exactly through Importer (same flow as original)
    importer.load_checkpoint()
    best_model = importer.get_model(return_model="best_model")
    data_loader = importer.get_dataloaders()

    validator = Validator(model=best_model, data_loader=data_loader, config=importer.config, device=device)
    test_results = validator.get_results(set="Test", print_results=True)

    # Pull out preds/targets in the structure you described (if present)
    preds = None
    targets = None
    try:
        preds = test_results["total_preds"]["combined"]
    except Exception:
        pass
    try:
        targets = test_results["total_preds_target"]
    except Exception:
        pass

    out = {
        "fold": fold,
        "checkpoint": ckpt_path,
        "test_results": test_results,
        "preds_combined": preds,
        "targets": targets,
    }
    return _to_jsonable(out)


def main():
    all_results = {
        "config_path": CONFIG_PATH,
        "default_config_path": DEFAULT_CONFIG_PATH,
        "checkpoint_template": CHECKPOINT_TEMPLATE,
        "folds": {},
    }

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "rb") as f:
            all_results = pickle.load(f)


    for fold in FOLDS:
        print(f"\n========== Running fold {fold} ==========")
        fold_res = run_fold(fold)
        reform_fold = int(fold+3)
        all_results["folds"][reform_fold] = fold_res

    with open(OUTPUT_JSON, "wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved JSON results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
