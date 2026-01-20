# Token-cached ScienceQA loader (ONLY: input_ids, attention_mask, image_mask, hint_mask, label, id)

import os
import json
import random
import multiprocessing
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pynvml


# =========================
# Shard LRU cache (per worker process)
# =========================
class _ShardCache:
    def __init__(self, max_shards: int = 2):
        self.max = max_shards
        self.cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    def get(self, path: str) -> List[Dict[str, Any]]:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        items = torch.load(path, map_location="cpu")
        self.cache[path] = items
        self.cache.move_to_end(path)
        while len(self.cache) > self.max:
            self.cache.popitem(last=False)
        return items


_GLOBAL_SHARD_CACHE: Optional[_ShardCache] = None


def _get_cache(max_shards: int) -> _ShardCache:
    global _GLOBAL_SHARD_CACHE
    if _GLOBAL_SHARD_CACHE is None:
        _GLOBAL_SHARD_CACHE = _ShardCache(max_shards)
    return _GLOBAL_SHARD_CACHE


# =========================
# Dataset
# =========================
class ScienceQA_TokenCachedDataset(Dataset):
    """
    Loads cached shards produced by your cache script, but returns ONLY:
      - input_ids
      - attention_mask
      - masks["image"]
      - masks["hint"]
      - label
      - id
    """

    def __init__(self, cache_root: str, split: str, *, max_shards_in_mem: int = 2):
        super().__init__()
        self.split_dir = os.path.join(cache_root, split)
        assert os.path.isdir(self.split_dir), f"Missing split dir: {self.split_dir}"

        manifest_path = os.path.join(self.split_dir, "manifest.jsonl")
        assert os.path.isfile(manifest_path), f"Missing manifest: {manifest_path}"

        # Build global index: (shard_path, offset)
        self.index: List[Tuple[str, int]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                shard_path = os.path.join(self.split_dir, rec["shard"])
                n = int(rec["num_items"])
                for j in range(n):
                    self.index.append((shard_path, j))

        self.max_shards_in_mem = int(max_shards_in_mem)
        print(f"[TokenCachedScienceQA] split={split} N={len(self.index)}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_path, offset = self.index[idx]
        cache = _get_cache(self.max_shards_in_mem)
        shard_items = cache.get(shard_path)
        ex = shard_items[offset]

        masks = ex.get("masks", None)
        if masks is None or ("image" not in masks) or ("hint" not in masks):
            raise KeyError(
                "Cached example is missing masks['image'] or masks['hint'].\n"
                "Rebuild cache with masks enabled."
            )

        return {
            "id": ex["id"],
            "label": ex["label"].long(),
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "image_mask": masks["image"].bool(),
            "hint_mask": masks["hint"].bool(),
        }


# =========================
# Collate (LEFT pad)
# =========================
def _left_pad_1d(seqs: List[torch.Tensor], pad_val: int, dtype: torch.dtype) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_val, dtype=dtype)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        out[i, -L:] = s.to(dtype)
    return out


def _left_pad_bool(seqs: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs)
    out = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        out[i, -L:] = s.bool()
    return out


def scienceqa_token_cached_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [b["id"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    input_ids = _left_pad_1d([b["input_ids"] for b in batch], pad_val=0, dtype=torch.long)
    attention_mask = _left_pad_1d([b["attention_mask"] for b in batch], pad_val=0, dtype=torch.long)

    image_mask = _left_pad_bool([b["image_mask"] for b in batch])
    hint_mask = _left_pad_bool([b["hint_mask"] for b in batch])

    return {
        "ids": ids,
        "label": labels,
        "data": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_mask": image_mask,
            "hint_mask": hint_mask,
        },
    }


# =========================
# Dataloader wrapper
# =========================
class ScienceQA_TokenCachedDataloader:
    def __init__(self, config, *, max_shards_in_mem: int = 2):
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)

        g = torch.Generator()
        g.manual_seed(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        def get_physical_gpu_count():
            try:
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return int(count)
            except Exception:
                return max(1, torch.cuda.device_count())

        total_cpus = multiprocessing.cpu_count()
        num_gpus = max(1, get_physical_gpu_count())
        workers_per_gpu = max(1, (total_cpus - 1) // num_gpus)
        # workers_per_gpu = 0

        print(
            f"[TokenCachedScienceQA] GPUs: {torch.cuda.device_count()} (Phys: {num_gpus}) | "
            f"CPUs: {total_cpus} | Workers: {torch.cuda.device_count()}x{workers_per_gpu}="
            f"{torch.cuda.device_count() * workers_per_gpu}"
        )

        self.collate_fn = scienceqa_token_cached_collate

        def make_loader(split: str, shuffle: bool):
            ds = ScienceQA_TokenCachedDataset(
                cache_root=cache_root,
                split=split,
                max_shards_in_mem=max_shards_in_mem,
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=g if shuffle else None,
                worker_init_fn=seed_worker,
                collate_fn=self.collate_fn,
                num_workers=workers_per_gpu,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
            )

        self.train_loader = make_loader("train", shuffle=True)
        self.valid_loader = make_loader("validation", shuffle=False)
        self.test_loader = make_loader("test", shuffle=False)


# =========================
# Quick smoke test
# =========================
if __name__ == "__main__":
    import types

    config = types.SimpleNamespace()
    config.training_params = types.SimpleNamespace()
    config.dataset = types.SimpleNamespace()
    config.training_params.batch_size = 8

    config.dataset.cache_root = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA/cache_tokens2B"  # <-- set this

    loader = ScienceQA_TokenCachedDataloader(config)

    batch = next(iter(loader.train_loader))
    print("ids[0:3]:", batch["ids"][:3])
    print("label:", batch["label"].shape, batch["label"].dtype)
    print("input_ids:", batch["data"]["input_ids"].shape, batch["data"]["input_ids"].dtype)
    print("attention_mask:", batch["data"]["attention_mask"].shape, batch["data"]["attention_mask"].dtype)
    print("image_mask:", batch["data"]["image_mask"].shape, batch["data"]["image_mask"].dtype)
    print("hint_mask:", batch["data"]["hint_mask"].shape, batch["data"]["hint_mask"].dtype)
