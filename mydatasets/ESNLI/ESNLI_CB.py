
import os
import json
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


def _stats(name: str, t: Any) -> str:
    if not torch.is_tensor(t):
        return f"{name:24s}: {type(t)}"
    if t.numel() == 0:
        return f"{name:24s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} numel=0"
    x = t.detach()
    xf = x.float()
    mn = xf.min().item()
    mx = xf.max().item()
    mean = xf.mean().item()
    std = xf.std(unbiased=False).item()
    nonzero = (x != 0).float().mean().item() * 100.0
    return (f"{name:24s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} nonzero={nonzero:.2f}%")
def _require_tensor(ex: dict, key: str) -> torch.Tensor:
    if key not in ex:
        raise KeyError(f"Missing key in shard item: {key}")
    t = ex[key]
    if not torch.is_tensor(t):
        raise TypeError(f"Key {key} must be a torch.Tensor, got {type(t)}")
    return t
def _check_shapes(ex: dict) -> None:
    input_ids = _require_tensor(ex, "input_ids")
    attention_mask = _require_tensor(ex, "attention_mask")
    position_ids = _require_tensor(ex, "position_ids")
    input_embeds = _require_tensor(ex, "input_embeds")
    visual_pos_masks = _require_tensor(ex, "visual_pos_masks")
    deep = _require_tensor(ex, "deepstack_visual_embeds")

    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids expected (1,L), got {tuple(input_ids.shape)}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(f"attention_mask expected {tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}")
    if input_embeds.dim() != 3 or input_embeds.shape[0] != 1 or input_embeds.shape[1] != input_ids.shape[1]:
        raise ValueError(f"input_embeds expected (1,L,2048), got {tuple(input_embeds.shape)} vs L={input_ids.shape[1]}")
    if input_embeds.shape[-1] != 2048:
        raise ValueError(f"input_embeds last dim must be 2048, got {input_embeds.shape[-1]}")
    if visual_pos_masks.shape != input_ids.shape:
        raise ValueError(f"visual_pos_masks expected {tuple(input_ids.shape)}, got {tuple(visual_pos_masks.shape)}")
    if position_ids.dim() != 3 or tuple(position_ids.shape[:2]) != (3, 1) or position_ids.shape[-1] != input_ids.shape[1]:
        raise ValueError(f"position_ids expected (3,1,L), got {tuple(position_ids.shape)} vs L={input_ids.shape[1]}")
    if deep.dim() != 3 or deep.shape[1:] != (64, 2048):
        raise ValueError(f"deepstack_visual_embeds expected (K,64,2048), got {tuple(deep.shape)}")

def _pad_1d(x: torch.Tensor, L: int, pad_value: int, padding_side: str) -> torch.Tensor:
    x = x.reshape(-1)
    n = int(x.numel())
    if n == L:
        return x
    if n > L:
        return x[-L:] if padding_side == "left" else x[:L]
    out = torch.full((L,), pad_value, dtype=x.dtype)
    if padding_side == "left":
        out[-n:] = x
    else:
        out[:n] = x
    return out
def _pad_2d_time(x: torch.Tensor, L: int, pad_value: float, padding_side: str) -> torch.Tensor:
    # x: (T,D)
    if x.dim() != 2:
        raise ValueError(f"_pad_2d_time expects 2D (T,D), got {tuple(x.shape)}")
    T, D = int(x.shape[0]), int(x.shape[1])
    if T == L:
        return x
    if T > L:
        return x[-L:, :] if padding_side == "left" else x[:L, :]
    out = torch.full((L, D), float(pad_value), dtype=x.dtype)
    if padding_side == "left":
        out[-T:, :] = x
    else:
        out[:T, :] = x
    return out
def _pad_pos_3_1_T(pos: torch.Tensor, L: int, padding_side: str) -> torch.Tensor:
    # pos: (3,1,T)
    if pos.dim() != 3 or int(pos.shape[0]) != 3 or int(pos.shape[1]) != 1:
        raise ValueError(f"position_ids expected (3,1,T), got {tuple(pos.shape)}")
    T = int(pos.shape[-1])
    if T == L:
        return pos
    if T > L:
        return pos[:, :, -L:] if padding_side == "left" else pos[:, :, :L]
    out = torch.zeros((3, 1, L), dtype=pos.dtype)
    if padding_side == "left":
        out[:, :, -T:] = pos
    else:
        out[:, :, :T] = pos
    return out
def _pad_deepstack_K_64_2048(x: torch.Tensor, Kmax: int, padding_side: str) -> torch.Tensor:
    # x: (K,64,2048) -> (Kmax,64,2048)
    if Kmax == 0:
        return torch.zeros((0, 64, 2048), dtype=x.dtype if torch.is_tensor(x) else torch.float16)

    if (not torch.is_tensor(x)) or x.numel() == 0:
        return torch.zeros((Kmax, 64, 2048), dtype=torch.float16)

    if x.dim() != 3 or x.shape[1:] != (64, 2048):
        raise ValueError(f"deepstack_visual_embeds expected (K,64,2048), got {tuple(x.shape)}")

    K = int(x.shape[0])
    if K == Kmax:
        return x
    if K > Kmax:
        return x[-Kmax:] if padding_side == "left" else x[:Kmax]

    out = torch.zeros((Kmax, 64, 2048), dtype=x.dtype)
    if padding_side == "left":
        out[-K:] = x
    else:
        out[:K] = x
    return out
def _pad_deepstack_stack64(x: torch.Tensor, Kmax: int, padding_side: str) -> torch.Tensor:
    """
    Stack the 64s across deepstack levels.

    Input:
      x: (K, 64, 2048)

    Output (padded/truncated to Kmax levels):
      (Kmax*64, 2048)

    Padding:
      - If x has fewer than Kmax levels -> pad with zeros (as extra 64-blocks)
      - If x has more than Kmax levels -> truncate levels (from left/right depending on padding_side)
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    # target rows after stacking 64s
    target_rows = int(Kmax) * 64

    # handle empty target
    if target_rows == 0:
        dtype = x.dtype if torch.is_tensor(x) else torch.float16
        return torch.zeros((0, 2048), dtype=dtype)

    # missing/empty deepstack -> zeros
    if (not torch.is_tensor(x)) or x.numel() == 0:
        return torch.zeros((target_rows, 2048), dtype=torch.float16)

    if x.dim() != 3 or tuple(x.shape[1:]) != (64, 2048):
        raise ValueError(f"deepstack_visual_embeds expected (K,64,2048), got {tuple(x.shape)}")

    K = int(x.shape[0])

    # truncate/pad in K first
    if K == Kmax:
        x2 = x
    elif K > Kmax:
        x2 = x[-Kmax:] if padding_side == "left" else x[:Kmax]
    else:
        pad_levels = torch.zeros((Kmax - K, 64, 2048), dtype=x.dtype)
        x2 = torch.cat([pad_levels, x], dim=0) if padding_side == "left" else torch.cat([x, pad_levels], dim=0)

    # now stack 64s: (Kmax,64,2048) -> (Kmax*64, 2048)
    return x2.reshape(Kmax * 64, 2048).contiguous()


class ESNLI_MemmapDataset(Dataset):
    def __init__(
        self,
        cache_root: str,
        split: str,
        shard_index: int = 0,
        shard_path: Optional[str] = None,
        max_items: Optional[int] = None,
    ):
        super().__init__()
        split_dir = os.path.join(cache_root, split)

        if shard_path is None:
            manifest_path = os.path.join(split_dir, "manifest.jsonl")
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError(f"Missing manifest.jsonl at {manifest_path}")
            recs = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
            if len(recs) == 0:
                raise RuntimeError(f"Empty manifest: {manifest_path}")
            shard_index = int(shard_index)
            if shard_index < 0 or shard_index >= len(recs):
                raise IndexError(f"shard_index {shard_index} out of range [0, {len(recs)-1}]")
            shard_path = os.path.join(split_dir, recs[shard_index]["shard"])

        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        items = torch.load(shard_path, map_location="cpu")
        if not isinstance(items, (list, tuple)):
            raise TypeError(f"Expected shard to contain list/tuple, got {type(items)}")

        if max_items is not None:
            items = list(items)[: int(max_items)]
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]
        if not isinstance(ex, dict):
            raise TypeError(f"Example[{idx}] is not a dict: {type(ex)}")

        lab = ex.get("label", 0)
        if torch.is_tensor(lab):
            lab = int(lab.detach().cpu().reshape(-1)[0].item()) if lab.numel() else 0
        try:
            lab = int(lab)
        except Exception:
            lab = 0

        out = {
            "id": ex.get("id", idx),
            "prompt": ex.get("prompt", ""),
            "label": torch.tensor(lab, dtype=torch.long),

            "input_ids": _require_tensor(ex, "input_ids").to(torch.long),
            "attention_mask": _require_tensor(ex, "attention_mask").to(torch.long),
            "position_ids": _require_tensor(ex, "position_ids").to(torch.long),
            "input_embeds": _require_tensor(ex, "input_embeds").to(torch.float16),
            "visual_pos_masks": _require_tensor(ex, "visual_pos_masks").to(torch.bool),
            "deepstack_visual_embeds": _require_tensor(ex, "deepstack_visual_embeds").to(torch.float16),
        }

        _check_shapes(out)
        return out

def make_collate_from_cache( *, padding_side: str = "right", pad_token_id: int = 0,):
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        ids_ = [b["id"] for b in batch]
        prompts_ = [b.get("prompt", "") for b in batch]
        labels = torch.stack([b["label"] for b in batch], dim=0)

        # squeeze cached (1,L)->(L) etc; collect lengths and K
        ids_list, attn_list, pos_list, emb_list, vmask_list, deep_list = [], [], [], [], [], []
        lengths, Ks = [], []

        for b in batch:
            ids = b["input_ids"]
            am = b["attention_mask"]
            pos = b["position_ids"]
            emb = b["input_embeds"]
            vm = b["visual_pos_masks"]
            ds = b["deepstack_visual_embeds"]

            # squeeze sample batch dim
            ids = ids[0] if (ids.dim() == 2 and ids.shape[0] == 1) else ids
            am  = am[0]  if (am.dim() == 2 and am.shape[0] == 1) else am
            emb = emb[0] if (emb.dim() == 3 and emb.shape[0] == 1) else emb  # (L,2048)
            vm  = vm[0]  if (vm.dim() == 2 and vm.shape[0] == 1) else vm      # (L,)
            # pos stays (3,1,L)

            L = int(ids.numel())
            K = int(ds.shape[0]) if (torch.is_tensor(ds) and ds.dim() == 3) else 0
            lengths.append(L)
            Ks.append(K)

            ids_list.append(ids)
            attn_list.append(am)
            pos_list.append(pos)
            emb_list.append(emb)
            vmask_list.append(vm)
            deep_list.append(ds)

        Lmax = max(lengths) if lengths else 0
        Kmax = max(Ks) if Ks else 0

        input_ids = torch.stack([_pad_1d(x, Lmax, pad_value=pad_token_id, padding_side=padding_side) for x in ids_list], dim=0)
        attention_mask = torch.stack([_pad_1d(x, Lmax, pad_value=0, padding_side=padding_side) for x in attn_list], dim=0)

        # (B,3,1,Lmax)
        position_ids = torch.stack([_pad_pos_3_1_T(p, Lmax, padding_side=padding_side) for p in pos_list], dim=0)

        # (B,Lmax,2048)
        input_embeds = torch.stack([_pad_2d_time(e, Lmax, pad_value=0.0, padding_side=padding_side) for e in emb_list], dim=0)

        # (B,Lmax) bool
        visual_pos_masks = torch.stack([_pad_1d(v.to(torch.bool), Lmax, pad_value=0, padding_side=padding_side).to(torch.bool) for v in vmask_list], dim=0)

        # (B,Kmax,64,2048)
        deepstack_visual_embeds = torch.stack(
            [_pad_deepstack_stack64(d, Kmax, padding_side=padding_side) for d in deep_list],
            dim=0,
        )  # (B, Kmax*64, 2048)

        return {
            "ids": ids_,
            "prompts": prompts_,
            "label": labels,
            "data": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "input_embeds": input_embeds,
                "visual_pos_masks": visual_pos_masks,
                "deepstack_visual_embeds": deepstack_visual_embeds,
            },
        }

    return collate

class ESNLI_MemmapDataloader:
    def __init__(
        self,
        config,
        *,
        split: str = "validation",
        shard_index: int = 0,
        shard_path: Optional[str] = None,
        max_items: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        padding_side: str = "right",
        pad_token_id: int = 0,
        print_first_batch_stats: bool = True,
    ):
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)

        ds = ESNLI_MemmapDataset(
            cache_root=cache_root,
            split=split,
            shard_index=shard_index,
            shard_path=shard_path,
            max_items=max_items,
        )

        collate_fn = make_collate_from_cache(
            padding_side=padding_side,
            pad_token_id=pad_token_id,
        )

        self.loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=collate_fn,
        )

        self.train_loader = self.loader
        self.valid_loader = self.loader
        self.test_loader = self.loader

        if print_first_batch_stats:
            batch = next(iter(self.loader))
            ex = batch["data"]
            print("\n[cache-reader] first batch loaded (padded from cache):")
            for k in ["input_ids", "attention_mask", "position_ids", "input_embeds", "visual_pos_masks", "deepstack_visual_embeds"]:
                print(_stats(k, ex[k]))

