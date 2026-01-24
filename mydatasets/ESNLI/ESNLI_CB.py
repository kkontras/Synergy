import os
import json
import random
import multiprocessing
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# Collate (unchanged)
# =========================

def _left_pad_1d(seqs: List[torch.Tensor], pad_val: int, dtype: torch.dtype) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs) if len(seqs) > 0 else 0
    out = torch.full((len(seqs), max_len), pad_val, dtype=dtype)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        if L > 0:
            out[i, -L:] = s.to(dtype)
    return out


def _left_pad_bool(seqs: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs) if len(seqs) > 0 else 0
    out = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        if L > 0:
            out[i, -L:] = s.bool()
    return out


def _pad_2d_by_rows(seqs: List[torch.Tensor], pad_val: float = 0.0):
    B = len(seqs)
    if B == 0:
        return torch.empty((0, 0, 0)), torch.empty((0, 0), dtype=torch.bool)

    D = 0
    ref_dtype = torch.float32
    for x in seqs:
        if torch.is_tensor(x) and x.numel() > 0 and x.dim() == 2:
            D = int(x.shape[1])
            ref_dtype = x.dtype
            break

    Nmax = 0
    for x in seqs:
        if torch.is_tensor(x) and x.dim() == 2:
            Nmax = max(Nmax, int(x.shape[0]))

    padded = torch.full((B, Nmax, D), float(pad_val), dtype=ref_dtype)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)

    for i, x in enumerate(seqs):
        if (not torch.is_tensor(x)) or x.numel() == 0:
            continue
        if x.dim() == 1:
            x = x.view(1, -1)
        if x.dim() != 2:
            x = x.view(1, -1)

        if int(x.shape[1]) > D:
            x = x[:, :D]
        elif int(x.shape[1]) < D:
            pad = torch.zeros((int(x.shape[0]), D - int(x.shape[1])), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        n = int(x.shape[0])
        padded[i, :n, :] = x
        mask[i, :n] = True

    return padded, mask


def _pad_deep_3d(seqs: List[torch.Tensor], deep_dim: int, pad_val: float = 0.0):
    B = len(seqs)
    if B == 0:
        return (
            torch.empty((0, 0, 0, deep_dim), dtype=torch.float32),
            torch.empty((0, 0, 0), dtype=torch.bool),
        )

    Tmax = 0
    Nmax = 0
    ref_dtype = torch.float32

    for x in seqs:
        if torch.is_tensor(x) and x.numel() > 0:
            if x.dim() == 4:
                x = x[0]
            if x.dim() == 3:
                Tmax = max(Tmax, int(x.shape[0]))
                Nmax = max(Nmax, int(x.shape[1]))
                ref_dtype = x.dtype

    padded = torch.full((B, Tmax, Nmax, deep_dim), float(pad_val), dtype=ref_dtype)
    mask = torch.zeros((B, Tmax, Nmax), dtype=torch.bool)

    for i, x in enumerate(seqs):
        if (not torch.is_tensor(x)) or x.numel() == 0:
            continue
        if x.dim() == 4:
            x = x[0]
        if x.dim() != 3:
            continue

        T, N, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        if D > deep_dim:
            x = x[..., :deep_dim]
        elif D < deep_dim:
            pad = torch.zeros((T, N, deep_dim - D), dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)

        padded[i, :T, :N, :] = x
        mask[i, :T, :N] = True

    return padded, mask


def esnli_memmap_collate(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    ids = [b.get("id", None) for b in batch]
    prompts = [b.get("prompt", "") for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    input_ids = _left_pad_1d([b["input_ids"] for b in batch], pad_val=int(pad_token_id), dtype=torch.long)
    attention_mask = _left_pad_1d([b["attention_mask"] for b in batch], pad_val=0, dtype=torch.long)
    position_ids = _left_pad_1d([b["position_ids"] for b in batch], pad_val=0, dtype=torch.long)

    image_mask = _left_pad_bool([b["image_mask"] for b in batch])
    text_mask = _left_pad_bool([b["text_mask"] for b in batch])

    data: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_mask": image_mask,
        "text_mask": text_mask,
    }

    if "image_grid_thw" in batch[0]:
        try:
            data["image_grid_thw"] = torch.stack(
                [b.get("image_grid_thw", torch.zeros(3, dtype=torch.long)) for b in batch],
                dim=0,
            )
        except Exception:
            pass

    if "pixel_values" in batch[0]:
        try:
            data["pixel_values"] = torch.stack([b["pixel_values"] for b in batch], dim=0)
        except Exception:
            pass

    if "vision_embeds" in batch[0]:
        vis_list = [b.get("vision_embeds", torch.empty((0, 0), dtype=torch.float32)) for b in batch]
        vis_pad, vis_mask = _pad_2d_by_rows(vis_list, pad_val=0.0)
        data["vision_embeds"] = vis_pad
        data["vision_mask"] = vis_mask

    if "deep_stack_viz" in batch[0]:
        deep_list = [b.get("deep_stack_viz", torch.empty((0, 0, 2048), dtype=torch.float32)) for b in batch]
        deep_dim = 2048
        for x in deep_list:
            if torch.is_tensor(x) and x.numel() > 0:
                deep_dim = int(x.shape[-1])
                break
        deep_pad, deep_mask = _pad_deep_3d(deep_list, deep_dim=deep_dim, pad_val=0.0)
        data["deep_stack_viz"] = deep_pad
        data["deep_mask"] = deep_mask

    return {"ids": ids, "prompts": prompts, "label": labels, "data": data}


# =========================
# Dataset (SINGLE-SHARD VERSION, same class name)
# =========================

def _as_1d_cpu_tensor(x, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if x is None:
        return torch.empty((0,), dtype=dtype or torch.long)
    if torch.is_tensor(x):
        t = x.detach().cpu().reshape(-1)
        return t.to(dtype) if dtype is not None else t
    try:
        t = torch.as_tensor(x).detach().cpu().reshape(-1)
        return t.to(dtype) if dtype is not None else t
    except Exception:
        return torch.empty((0,), dtype=dtype or torch.long)


def _as_bool_mask_1d(x, L: int) -> torch.Tensor:
    if x is None:
        return torch.zeros((L,), dtype=torch.bool)
    t = _as_1d_cpu_tensor(x, dtype=torch.bool)
    if t.numel() == L:
        return t
    if t.numel() == 0:
        return torch.zeros((L,), dtype=torch.bool)
    if t.numel() > L:
        return t[:L]
    out = torch.zeros((L,), dtype=torch.bool)
    out[-t.numel():] = t
    return out


def _as_position_ids_1d(x, L: int) -> torch.Tensor:
    if x is None:
        return torch.arange(L, dtype=torch.long) if L > 0 else torch.empty((0,), dtype=torch.long)
    t = _as_1d_cpu_tensor(x, dtype=torch.long)
    if t.numel() == L:
        return t
    if t.numel() == 0:
        return torch.arange(L, dtype=torch.long) if L > 0 else torch.empty((0,), dtype=torch.long)
    if t.numel() > L:
        return t[:L]
    out = torch.zeros((L,), dtype=torch.long)
    out[-t.numel():] = t
    return out


def _as_deep_3d(ex_deep: Any, deep_dim: int = 2048) -> torch.Tensor:
    if ex_deep is None:
        return torch.empty((0, 0, deep_dim), dtype=torch.float32)
    t = ex_deep.detach().cpu() if torch.is_tensor(ex_deep) else torch.as_tensor(ex_deep).detach().cpu()
    if t.numel() == 0:
        return torch.empty((0, 0, deep_dim), dtype=torch.float32)

    if t.dim() == 3:
        T, N, D = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
        if D != deep_dim:
            if D > deep_dim:
                t = t[..., :deep_dim]
            else:
                pad = torch.zeros((T, N, deep_dim - D), dtype=t.dtype)
                t = torch.cat([t, pad], dim=-1)
        return t.to(torch.float32)

    if t.dim() == 4:
        return _as_deep_3d(t[0], deep_dim=deep_dim)

    if t.dim() >= 2:
        D = int(t.shape[-1])
        flat = t.reshape(-1, D)
        if D != deep_dim:
            if D > deep_dim:
                flat = flat[:, :deep_dim]
            else:
                pad = torch.zeros((int(flat.shape[0]), deep_dim - D), dtype=flat.dtype)
                flat = torch.cat([flat, pad], dim=-1)
        return flat.view(1, int(flat.shape[0]), deep_dim).to(torch.float32)

    flat = t.reshape(-1)
    if flat.numel() < deep_dim:
        pad = torch.zeros((deep_dim - flat.numel(),), dtype=flat.dtype)
        flat = torch.cat([flat, pad], dim=0)
    flat = flat[:deep_dim]
    return flat.view(1, 1, deep_dim).to(torch.float32)


class ESNLI_MemmapDataset(Dataset):
    """
    SINGLE-SHARD dataset. Same name as your memmap dataset, but it reads ONE shard directly.

    How it picks the shard:
      - Reads {cache_root}/{split}/manifest.jsonl
      - Picks shard_index (int) if provided; else picks the first record (0) by default.
    """

    def __init__(
        self,
        cache_root: str,
        split: str,
        shard_index: int = 0,          # <-- change this to inspect other shards
        shard_path: Optional[str] = None,  # <-- or pass an explicit shard file path
        deep_dim: int = 2048,
        max_items: Optional[int] = None,   # <-- optionally truncate for faster iteration
    ):
        super().__init__()
        self.split_dir = os.path.join(cache_root, split)

        if shard_path is None:
            manifest_path = os.path.join(self.split_dir, "manifest.jsonl")
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError(f"Missing manifest.jsonl at {manifest_path}")

            recs = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
            if len(recs) == 0:
                raise RuntimeError(f"Empty manifest: {manifest_path}")

            shard_index = int(shard_index)
            if shard_index < 0 or shard_index >= len(recs):
                raise IndexError(f"shard_index {shard_index} out of range [0, {len(recs)-1}]")

            shard_path = os.path.join(self.split_dir, recs[shard_index]["shard"])

        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        self.shard_path = shard_path
        self.deep_dim = int(deep_dim)

        # load the one shard into memory (simplest + best for debugging)
        items = torch.load(self.shard_path, map_location="cpu")
        if not isinstance(items, (list, tuple)):
            raise TypeError(f"Expected shard to contain a list/tuple of examples, got {type(items)}")

        if max_items is not None:
            items = list(items)[: int(max_items)]
        self.items = items

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]
        if not isinstance(ex, dict):
            raise TypeError(f"Example[{idx}] is not a dict, got {type(ex)}")

        inp = _as_1d_cpu_tensor(ex.get("input_ids", None), dtype=torch.long)
        L = int(inp.numel())

        att = _as_1d_cpu_tensor(ex.get("attention_mask", None), dtype=torch.long)
        if att.numel() != L:
            att = torch.ones((L,), dtype=torch.long) if L > 0 else torch.empty((0,), dtype=torch.long)

        pos = _as_position_ids_1d(ex.get("position_ids", None), L)

        masks = ex.get("masks", {}) if isinstance(ex.get("masks", {}), dict) else {}
        imgm = _as_bool_mask_1d(masks.get("image", None), L)
        txtm = _as_bool_mask_1d(masks.get("text", None), L)

        lab = ex.get("label", 0)
        if isinstance(lab, torch.Tensor):
            lab = int(lab.detach().cpu().reshape(-1)[0].item()) if lab.numel() > 0 else 0
        try:
            lab = int(lab)
        except Exception:
            lab = 0

        out: Dict[str, Any] = {
            "id": ex.get("id", idx),
            "label": torch.tensor(lab, dtype=torch.long),
            "prompt": ex.get("prompt", ""),
            "input_ids": inp.long(),
            "attention_mask": att.long(),
            "position_ids": pos.long(),
            "image_mask": imgm.bool(),
            "text_mask": txtm.bool(),
        }

        # optional grid
        if ex.get("image_grid_thw", None) is not None:
            g = ex["image_grid_thw"]
            if isinstance(g, torch.Tensor):
                g = g.detach().cpu().reshape(-1)
            else:
                g = torch.as_tensor(g).detach().cpu().reshape(-1)
            tmp = torch.zeros((3,), dtype=torch.long)
            tmp[: min(3, int(g.numel()))] = g[: min(3, int(g.numel()))].long()
            out["image_grid_thw"] = tmp

        # optional vision
        ve = ex.get("vision_embeds", None)
        if ve is not None:
            ve_t = ve.detach().cpu() if torch.is_tensor(ve) else torch.as_tensor(ve).detach().cpu()
            if ve_t.numel() == 0:
                out["vision_embeds"] = torch.empty((0, 0), dtype=torch.float32)
            else:
                if ve_t.dim() == 1:
                    ve_t = ve_t.view(1, -1)
                elif ve_t.dim() != 2:
                    ve_t = ve_t.view(1, -1)
                out["vision_embeds"] = ve_t.to(torch.float32 if ve_t.dtype != torch.float32 else torch.float32)

        # optional deep
        dv = ex.get("deep_stack_viz", None)
        if dv is not None:
            out["deep_stack_viz"] = _as_deep_3d(dv, deep_dim=self.deep_dim)

        # optional pixel
        pv = ex.get("pixel_values", None)
        if pv is not None:
            pv_t = pv.detach().cpu() if torch.is_tensor(pv) else torch.as_tensor(pv).detach().cpu()
            out["pixel_values"] = pv_t.to(torch.float32)

        return out


# =========================
# Dataloader wrapper (SINGLE-SHARD VERSION, same class name)
# =========================

class ESNLI_MemmapDataloader:
    """
    SINGLE-SHARD loader. Same name as your memmap wrapper, but it does NOT build memmaps.
    It just loads one shard through ESNLI_MemmapDataset above.

    Configure via:
      config.dataset.cache_root
      config.training_params.batch_size
      optionally config.model.pad_token_id

    You can pass shard_index to pick which shard in manifest.jsonl to inspect.
    """

    def __init__(
        self,
        config,
        *,
        shard_index: int = 0,
        shard_path: Optional[str] = None,
        deep_dim: int = 2048,
        max_items: Optional[int] = None,
        num_workers: Optional[int] = 0,   # <-- default 0 for clean debugging
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ):
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)
        pad_token_id = int(getattr(getattr(config, "model", None), "pad_token_id", 0))

        g = torch.Generator()
        g.manual_seed(0)

        def seed_worker(worker_id: int):
            worker_seed = torch.initial_seed() % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        if num_workers is None:
            total_cpus = multiprocessing.cpu_count()
            num_workers = max(1, min(4, total_cpus // 8 if total_cpus >= 16 else 2))

        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            return esnli_memmap_collate(batch, pad_token_id=pad_token_id)

        self.collate_fn = collate_fn

        def make_loader(split: str, shuffle: bool):
            ds = ESNLI_MemmapDataset(
                cache_root=cache_root,
                split=split,
                shard_index=shard_index,
                shard_path=shard_path,
                deep_dim=deep_dim,
                max_items=max_items,
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=g if shuffle else None,
                worker_init_fn=seed_worker if int(num_workers) > 0 else None,
                collate_fn=self.collate_fn,
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory),
                prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
                persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
            )

        # Keep the same attributes as your original wrapper
        self.train_loader = make_loader("validation", shuffle=False) if os.path.isdir(os.path.join(cache_root, "validation")) else None
        self.valid_loader = make_loader("validation", shuffle=False) if os.path.isdir(os.path.join(cache_root, "validation")) else None
        self.test_loader = make_loader("test", shuffle=False) if os.path.isdir(os.path.join(cache_root, "test")) else None


# =========================
# Example usage
# =========================
# loader = ESNLI_MemmapDataloader(cfg, shard_index=0, num_workers=0, max_items=128)
# batch = next(iter(loader.valid_loader))
# print(batch["data"]["input_ids"].shape, batch["label"].shape)
# if "vision_embeds" in batch["data"]: print(batch["data"]["vision_embeds"].shape, batch["data"]["vision_mask"].shape)
# if "deep_stack_viz" in batch["data"]: print(batch["data"]["deep_stack_viz"].shape, batch["data"]["deep_mask"].shape)
