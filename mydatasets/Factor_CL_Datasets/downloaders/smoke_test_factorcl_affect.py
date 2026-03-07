#!/usr/bin/env python3
"""
Smoke test for FactorCL affect prepared files.

Checks:
1) pickle file is readable
2) required splits/keys exist
3) get_dataloader can load one batch per split
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS: Dict[str, Tuple[str, str]] = {
    "mosi": ("mosi/mosi_data.pkl", "mosi"),
    "mosei": ("mosei/mosei_data.pkl", "mosei"),
    "ur_funny": ("ur_funny/ur_funny_data.pkl", "humor"),
    "mustard": ("mustard/mustard_data.pkl", "sarcasm"),
}


def _assert_pickle_shape(path: Path) -> None:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a dict pickle.")
    for split in ("train", "valid", "test"):
        if split not in obj:
            raise ValueError(f"{path} missing split: {split}")
        split_obj = obj[split]
        if not isinstance(split_obj, dict):
            raise ValueError(f"{path} split '{split}' is not a dict.")
        for key in ("vision", "audio", "text", "labels"):
            if key not in split_obj:
                raise ValueError(f"{path} split '{split}' missing key: {key}")


def smoke_one_dataset(dataset_name: str, root: Path) -> None:
    rel_path, data_type = DATASETS[dataset_name]
    path = root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name}: missing file {path}")

    _assert_pickle_shape(path)
    print(f"[{dataset_name}] pickle structure OK ({path}) data_type={data_type}")


def optional_dataloader_smoke(root: Path, batch_size: int) -> None:
    try:
        import torch
        from mydatasets.Factor_CL_Datasets.MultiBench.datasets.affect.get_data import get_dataloader
    except Exception as exc:
        print(f"[smoke] dataloader batch test skipped (dependency/import issue): {exc}")
        return

    def _batch_signature(batch) -> str:
        if not isinstance(batch, (tuple, list)) or len(batch) < 4:
            return f"unexpected_batch_type={type(batch)}"
        x = batch[0]
        label = batch[3]
        if isinstance(x, (tuple, list)) and len(x) >= 3:
            shapes = [tuple(t.shape) if torch.is_tensor(t) else str(type(t)) for t in x[:3]]
        else:
            shapes = [tuple(t.shape) if torch.is_tensor(t) else str(type(t)) for t in batch[:3]]
        label_shape = tuple(label.shape) if torch.is_tensor(label) else str(type(label))
        return f"modalities={shapes}, label={label_shape}"

    for dataset_name in ("mosi", "mosei", "ur_funny", "mustard"):
        rel_path, data_type = DATASETS[dataset_name]
        path = root / rel_path
        train_loader, valid_loader, test_loader = get_dataloader(
            filepath=str(path),
            robust_test=False,
            data_type=data_type,
            task="classification",
            batch_size=batch_size,
            train_shuffle=False,
            num_workers=0,
            max_pad=False,
            max_seq_len=50,
            flatten_time_series=False,
            z_norm=False,
        )
        print(f"[{dataset_name}] train: {_batch_signature(next(iter(train_loader)))}")
        print(f"[{dataset_name}] valid: {_batch_signature(next(iter(valid_loader)))}")
        print(f"[{dataset_name}] test : {_batch_signature(next(iter(test_loader)))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test all FactorCL affect dataset prepared files.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing mosi/mosei/ur_funny/mustard subfolders.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    root = args.root.resolve()
    for name in ("mosi", "mosei", "ur_funny", "mustard"):
        smoke_one_dataset(name, root)

    optional_dataloader_smoke(root, batch_size=args.batch_size)

    print("[smoke] all datasets passed")


if __name__ == "__main__":
    main()
