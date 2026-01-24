"""
ScienceQA memmap loader (low-RAM, worker-friendly) UPDATED:
Returns:
  - input_ids, attention_mask, position_ids
  - image_mask, hint_mask
  - vision_embeds (+ vision_mask in collate)
  - deep_stack_viz (+ deep_mask in collate)
  - label, id, prompt

Assumes cached shards store list[dict] with keys (best-effort, no strict checks):
  - id, label, prompt
  - input_ids, attention_mask, position_ids
  - masks: {"image": ..., "hint": ...}  (if hint missing, falls back to masks["text"])
  - vision_embeds (optional; 1D or 2D)
  - deep_stack_viz (optional; expected per-example [T, Nimg, 2048] or [1,T,Nimg,2048])

Memmap cache per split:
  meta.json
  offsets.npy, lengths.npy, labels.npy
  ids.jsonl, prompts.jsonl
  input_ids.bin, attention.bin, position_ids.bin
  image_mask.bin, hint_mask.bin
  vision_offsets.npy, vision_lengths.npy, vision_dim.npy, vision.bin          (if present)
  deep_offsets.npy, deep_tlens.npy, deep_nlens.npy, deep_dim.npy, deep.bin    (if present)
"""

import os
import json
import time
import random
import multiprocessing
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# Utilities
# =========================

def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _try_acquire_lock(lock_path: str, wait_seconds: int = 600) -> bool:
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"pid={os.getpid()} time={time.time()}\n")
            return True
        except FileExistsError:
            if time.time() - start > wait_seconds:
                return False
            time.sleep(0.25)


def _release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_1d_cpu_tensor(x, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Best-effort 1D CPU tensor. Safe for non-contiguous (reshape)."""
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


def _as_vision_2d(ex_vision: Any) -> torch.Tensor:
    """
    Best-effort coerce vision_embeds to [Nimg, D] on CPU.
    None -> [0,0]
    """
    if ex_vision is None:
        return torch.empty((0, 0), dtype=torch.float16)

    t = ex_vision.detach().cpu() if torch.is_tensor(ex_vision) else torch.as_tensor(ex_vision).detach().cpu()
    if t.numel() == 0:
        return torch.empty((0, 0), dtype=torch.float16)

    if t.dim() == 1:
        return t.view(1, -1).to(torch.float16)
    if t.dim() != 2:
        t = t.reshape(1, -1)
    return t.to(torch.float16)


def _as_deep_3d(ex_deep: Any, deep_dim: int = 2048) -> torch.Tensor:
    """
    Best-effort coerce deep_stack_viz to [T, Nimg, D] (CPU float16).
    None -> [0,0,D]
    """
    if ex_deep is None:
        return torch.empty((0, 0, deep_dim), dtype=torch.float16)

    t = ex_deep.detach().cpu() if torch.is_tensor(ex_deep) else torch.as_tensor(ex_deep).detach().cpu()
    if t.numel() == 0:
        return torch.empty((0, 0, deep_dim), dtype=torch.float16)

    if t.dim() == 4:
        # drop batch dim: [1,T,N,D] -> [T,N,D]
        t = t[0]
    if t.dim() == 3:
        T, N, D = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
        if D > deep_dim:
            t = t[..., :deep_dim]
        elif D < deep_dim:
            pad = torch.zeros((T, N, deep_dim - D), dtype=t.dtype)
            t = torch.cat([t, pad], dim=-1)
        return t.to(torch.float16)

    # fallback: interpret as [1, M, D]
    if t.dim() >= 2:
        D = int(t.shape[-1])
        flat = t.reshape(-1, D)
        if D > deep_dim:
            flat = flat[:, :deep_dim]
        elif D < deep_dim:
            pad = torch.zeros((int(flat.shape[0]), deep_dim - D), dtype=flat.dtype)
            flat = torch.cat([flat, pad], dim=-1)
        return flat.view(1, int(flat.shape[0]), deep_dim).to(torch.float16)

    # 1D vector
    flat = t.reshape(-1)
    if flat.numel() < deep_dim:
        pad = torch.zeros((deep_dim - flat.numel(),), dtype=flat.dtype)
        flat = torch.cat([flat, pad], dim=0)
    flat = flat[:deep_dim]
    return flat.view(1, 1, deep_dim).to(torch.float16)


# =========================
# Memmap builder
# =========================

def build_memmap_from_token_shards(
    *,
    split_dir: str,
    out_dir: str,
    input_ids_dtype: str = "int32",
    vision_dtype: str = "float16",
    deep_dim: int = 2048,
) -> None:
    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    _ensure_dir(out_dir)

    meta_path = os.path.join(out_dir, "meta.json")
    done_flag = os.path.join(out_dir, ".DONE")
    lock_path = os.path.join(out_dir, ".LOCK")

    if os.path.isfile(done_flag) and os.path.isfile(meta_path):
        return

    if not _try_acquire_lock(lock_path, wait_seconds=600):
        raise RuntimeError(f"Timed out waiting for lock: {lock_path}")

    try:
        if os.path.isfile(done_flag) and os.path.isfile(meta_path):
            return

        # -------- Read manifest --------
        shard_recs: List[Dict[str, Any]] = []
        N = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                shard_path = os.path.join(split_dir, rec["shard"])
                n_items = int(rec.get("n", rec.get("num_items", 0)))
                shard_recs.append({"path": shard_path, "n": n_items})
                N += n_items

        if N <= 0:
            raise RuntimeError(f"No items found in manifest: {manifest_path}")

        if input_ids_dtype not in ("int32", "int64"):
            input_ids_dtype = "int32"
        if vision_dtype not in ("float16", "float32"):
            vision_dtype = "float16"

        input_ids_np_dtype = np.int32 if input_ids_dtype == "int32" else np.int64
        vision_np_dtype = np.float16 if vision_dtype == "float16" else np.float32

        # -------- Pass 1 totals --------
        total_tokens = 0

        has_vision = False
        vision_dim: Optional[int] = None
        total_vision_elems = 0

        has_deep = False
        total_deep_elems = 0

        for r in shard_recs:
            items = torch.load(r["path"], map_location="cpu")
            for ex in items:
                inp = _as_1d_cpu_tensor(ex.get("input_ids", None), dtype=torch.long)
                L = int(inp.numel())
                total_tokens += L

                ve = ex.get("vision_embeds", None)
                if ve is not None:
                    v2 = _as_vision_2d(ve)
                    if v2.numel() > 0:
                        has_vision = True
                        nimg, D = int(v2.shape[0]), int(v2.shape[1])
                        if vision_dim is None and D > 0:
                            vision_dim = D
                        D_use = int(vision_dim) if (vision_dim is not None and vision_dim > 0) else D
                        total_vision_elems += int(nimg * D_use)

                dv = ex.get("deep_stack_viz", None)
                if dv is not None:
                    d3 = _as_deep_3d(dv, deep_dim=deep_dim)
                    if d3.numel() > 0:
                        has_deep = True
                        T = int(d3.shape[0])
                        Nimg = int(d3.shape[1])
                        total_deep_elems += int(T * Nimg * deep_dim)

        if has_vision and (vision_dim is None or vision_dim <= 0):
            vision_dim = 1
        D_use = int(vision_dim) if has_vision else 0

        # -------- Index arrays --------
        offsets = np.zeros((N,), dtype=np.int64)
        lengths = np.zeros((N,), dtype=np.int32)
        labels = np.zeros((N,), dtype=np.int64)

        vision_offsets = np.zeros((N,), dtype=np.int64) if has_vision else None
        vision_lengths = np.zeros((N,), dtype=np.int32) if has_vision else None

        deep_offsets = np.zeros((N,), dtype=np.int64) if has_deep else None
        deep_tlens = np.zeros((N,), dtype=np.int32) if has_deep else None
        deep_nlens = np.zeros((N,), dtype=np.int32) if has_deep else None

        # ids + prompts
        ids_path = os.path.join(out_dir, "ids.jsonl")
        prompts_path = os.path.join(out_dir, "prompts.jsonl")
        ids_f = open(ids_path + ".tmp", "w", encoding="utf-8")
        prompts_f = open(prompts_path + ".tmp", "w", encoding="utf-8")

        # -------- Token memmaps --------
        input_ids_mm = np.memmap(os.path.join(out_dir, "input_ids.bin"), mode="w+", dtype=input_ids_np_dtype, shape=(total_tokens,))
        attn_mm = np.memmap(os.path.join(out_dir, "attention.bin"), mode="w+", dtype=np.uint8, shape=(total_tokens,))
        pos_mm = np.memmap(os.path.join(out_dir, "position_ids.bin"), mode="w+", dtype=np.int32, shape=(total_tokens,))
        image_mm = np.memmap(os.path.join(out_dir, "image_mask.bin"), mode="w+", dtype=np.uint8, shape=(total_tokens,))
        hint_mm = np.memmap(os.path.join(out_dir, "hint_mask.bin"), mode="w+", dtype=np.uint8, shape=(total_tokens,))

        # -------- Vision memmap --------
        vision_mm = None
        if has_vision:
            vision_mm = np.memmap(os.path.join(out_dir, "vision.bin"), mode="w+", dtype=vision_np_dtype, shape=(max(int(total_vision_elems), 1),))

        # -------- Deep memmap --------
        deep_mm = None
        if has_deep:
            deep_mm = np.memmap(os.path.join(out_dir, "deep.bin"), mode="w+", dtype=np.float16, shape=(max(int(total_deep_elems), 1),))

        # -------- Pass 2 fill --------
        token_cursor = 0
        vision_cursor = 0
        deep_cursor = 0
        i = 0

        for r in shard_recs:
            items = torch.load(r["path"], map_location="cpu")
            for ex in items:
                inp = _as_1d_cpu_tensor(ex.get("input_ids", None), dtype=torch.long)
                L = int(inp.numel())

                att = _as_1d_cpu_tensor(ex.get("attention_mask", None), dtype=torch.long)
                if att.numel() != L:
                    att = torch.ones((L,), dtype=torch.long) if L > 0 else torch.empty((0,), dtype=torch.long)

                pos = _as_position_ids_1d(ex.get("position_ids", None), L)

                masks = ex.get("masks", {}) if isinstance(ex.get("masks", {}), dict) else {}
                imgm = _as_bool_mask_1d(masks.get("image", None), L)
                hsrc = masks.get("hint", None)
                if hsrc is None:
                    hsrc = masks.get("text", None)
                hntm = _as_bool_mask_1d(hsrc, L)

                offsets[i] = token_cursor
                lengths[i] = L

                lab = ex.get("label", 0)
                if isinstance(lab, torch.Tensor):
                    lab = int(lab.detach().cpu().reshape(-1)[0].item()) if lab.numel() > 0 else 0
                try:
                    labels[i] = int(lab)
                except Exception:
                    labels[i] = 0

                ids_f.write(json.dumps(ex.get("id", i)) + "\n")
                prompts_f.write(json.dumps(ex.get("prompt", "")) + "\n")

                if L > 0:
                    input_ids_mm[token_cursor:token_cursor + L] = inp.numpy().astype(input_ids_np_dtype, copy=False)
                    attn_mm[token_cursor:token_cursor + L] = att.numpy().astype(np.uint8, copy=False)
                    pos_mm[token_cursor:token_cursor + L] = pos.numpy().astype(np.int32, copy=False)
                    image_mm[token_cursor:token_cursor + L] = imgm.numpy().astype(np.uint8, copy=False)
                    hint_mm[token_cursor:token_cursor + L] = hntm.numpy().astype(np.uint8, copy=False)

                # vision
                if has_vision and vision_offsets is not None and vision_lengths is not None and vision_mm is not None:
                    v2 = _as_vision_2d(ex.get("vision_embeds", None))  # [N,D] float16
                    nimg = int(v2.shape[0]) if (v2.dim() == 2 and v2.numel() > 0) else 0

                    vision_offsets[i] = vision_cursor
                    vision_lengths[i] = nimg

                    if nimg > 0:
                        D_obs = int(v2.shape[1])
                        if D_obs > D_use:
                            v_use = v2[:, :D_use]
                        elif D_obs < D_use:
                            pad = torch.zeros((nimg, D_use - D_obs), dtype=v2.dtype)
                            v_use = torch.cat([v2, pad], dim=1)
                        else:
                            v_use = v2

                        flat = v_use.numpy().astype(vision_np_dtype, copy=False).reshape(-1)
                        n_elems = int(flat.size)
                        vision_mm[vision_cursor:vision_cursor + n_elems] = flat
                        vision_cursor += n_elems

                # deep
                if has_deep and deep_offsets is not None and deep_tlens is not None and deep_nlens is not None and deep_mm is not None:
                    d3 = _as_deep_3d(ex.get("deep_stack_viz", None), deep_dim=deep_dim)  # [T,N,D]
                    T = int(d3.shape[0])
                    Nimg = int(d3.shape[1])

                    deep_offsets[i] = deep_cursor
                    deep_tlens[i] = T
                    deep_nlens[i] = Nimg

                    if T > 0 and Nimg > 0:
                        flat = d3.numpy().astype(np.float16, copy=False).reshape(-1)
                        n_elems = int(flat.size)
                        deep_mm[deep_cursor:deep_cursor + n_elems] = flat
                        deep_cursor += n_elems

                token_cursor += L
                i += 1

        # flush + finalize
        input_ids_mm.flush(); attn_mm.flush(); pos_mm.flush()
        image_mm.flush(); hint_mm.flush()
        if vision_mm is not None: vision_mm.flush()
        if deep_mm is not None: deep_mm.flush()

        ids_f.close(); prompts_f.close()
        os.replace(ids_path + ".tmp", ids_path)
        os.replace(prompts_path + ".tmp", prompts_path)

        np.save(os.path.join(out_dir, "offsets.npy"), offsets)
        np.save(os.path.join(out_dir, "lengths.npy"), lengths)
        np.save(os.path.join(out_dir, "labels.npy"), labels)

        if has_vision and vision_offsets is not None:
            np.save(os.path.join(out_dir, "vision_offsets.npy"), vision_offsets)
            np.save(os.path.join(out_dir, "vision_lengths.npy"), vision_lengths)
            np.save(os.path.join(out_dir, "vision_dim.npy"), np.array([int(D_use)], dtype=np.int32))

        if has_deep and deep_offsets is not None:
            np.save(os.path.join(out_dir, "deep_offsets.npy"), deep_offsets)
            np.save(os.path.join(out_dir, "deep_tlens.npy"), deep_tlens)
            np.save(os.path.join(out_dir, "deep_nlens.npy"), deep_nlens)
            np.save(os.path.join(out_dir, "deep_dim.npy"), np.array([int(deep_dim)], dtype=np.int32))

        meta = {
            "version": 3,
            "N": int(N),
            "total_tokens": int(total_tokens),
            "input_ids_dtype": input_ids_dtype,
            "has_position_ids": True,
            "has_prompts": True,
            "has_vision_embeds": bool(has_vision),
            "vision_dtype": vision_dtype if has_vision else None,
            "vision_dim": int(D_use) if has_vision else None,
            "has_deep_stack_viz": bool(has_deep),
            "deep_dim": int(deep_dim) if has_deep else None,
            "paths": {
                "offsets": "offsets.npy",
                "lengths": "lengths.npy",
                "labels": "labels.npy",
                "ids": "ids.jsonl",
                "prompts": "prompts.jsonl",
                "input_ids": "input_ids.bin",
                "attention": "attention.bin",
                "position_ids": "position_ids.bin",
                "image_mask": "image_mask.bin",
                "hint_mask": "hint_mask.bin",
                "vision_offsets": "vision_offsets.npy" if has_vision else None,
                "vision_lengths": "vision_lengths.npy" if has_vision else None,
                "vision_dim": "vision_dim.npy" if has_vision else None,
                "vision": "vision.bin" if has_vision else None,
                "deep_offsets": "deep_offsets.npy" if has_deep else None,
                "deep_tlens": "deep_tlens.npy" if has_deep else None,
                "deep_nlens": "deep_nlens.npy" if has_deep else None,
                "deep_dim": "deep_dim.npy" if has_deep else None,
                "deep": "deep.bin" if has_deep else None,
            },
        }
        _atomic_write_json(meta_path, meta)

        with open(done_flag, "w", encoding="utf-8") as f:
            f.write("ok\n")

        print(f"[ScienceQA Memmap] Built: {out_dir} | N={N} | tokens={total_tokens} | vision={has_vision} | deep={has_deep}")

    finally:
        _release_lock(lock_path)


# =========================
# Dataset (memmap)
# =========================

class ScienceQA_MemmapDataset(Dataset):
    def __init__(self, cache_root: str, split: str):
        super().__init__()
        self.mem_dir = os.path.join(cache_root, split, "_memmap")
        meta_path = os.path.join(self.mem_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.N = int(meta.get("N", 0))
        self.total_tokens = int(meta.get("total_tokens", 0))
        self.input_ids_dtype = meta.get("input_ids_dtype", "int32")

        self.has_vision = bool(meta.get("has_vision_embeds", False))
        self.vision_dtype = meta.get("vision_dtype", "float16")
        self.D = int(meta.get("vision_dim", 0) or 0)

        self.has_deep = bool(meta.get("has_deep_stack_viz", False))
        self.deep_dim = int(meta.get("deep_dim", 2048) or 2048)

        self.offsets = np.load(os.path.join(self.mem_dir, "offsets.npy"))
        self.lengths = np.load(os.path.join(self.mem_dir, "lengths.npy"))
        self.labels = np.load(os.path.join(self.mem_dir, "labels.npy"))

        self.ids = [json.loads(l) for l in open(os.path.join(self.mem_dir, "ids.jsonl"), "r", encoding="utf-8")]
        pr_path = os.path.join(self.mem_dir, "prompts.jsonl")
        self.prompts = [json.loads(l) for l in open(pr_path, "r", encoding="utf-8")] if os.path.isfile(pr_path) else [""] * self.N

        input_dtype = np.int32 if self.input_ids_dtype == "int32" else np.int64
        self.input_ids_mm = np.memmap(os.path.join(self.mem_dir, "input_ids.bin"), mode="r", dtype=input_dtype, shape=(self.total_tokens,))
        self.attn_mm = np.memmap(os.path.join(self.mem_dir, "attention.bin"), mode="r", dtype=np.uint8, shape=(self.total_tokens,))
        self.pos_mm = np.memmap(os.path.join(self.mem_dir, "position_ids.bin"), mode="r", dtype=np.int32, shape=(self.total_tokens,))
        self.image_mm = np.memmap(os.path.join(self.mem_dir, "image_mask.bin"), mode="r", dtype=np.uint8, shape=(self.total_tokens,))
        self.hint_mm = np.memmap(os.path.join(self.mem_dir, "hint_mask.bin"), mode="r", dtype=np.uint8, shape=(self.total_tokens,))

        # vision
        self.vision_offsets = None
        self.vision_lengths = None
        self.vision_mm = None
        if self.has_vision and os.path.isfile(os.path.join(self.mem_dir, "vision_offsets.npy")):
            self.vision_offsets = np.load(os.path.join(self.mem_dir, "vision_offsets.npy"))
            self.vision_lengths = np.load(os.path.join(self.mem_dir, "vision_lengths.npy"))
            vd = np.load(os.path.join(self.mem_dir, "vision_dim.npy"))
            self.D = int(vd[0]) if vd.size else self.D
            v_dtype = np.float16 if self.vision_dtype == "float16" else np.float32
            total_vision_elems = int((self.vision_lengths.astype(np.int64) * max(self.D, 1)).sum())
            self.vision_mm = np.memmap(os.path.join(self.mem_dir, "vision.bin"), mode="r", dtype=v_dtype, shape=(max(total_vision_elems, 1),))

        # deep
        self.deep_offsets = None
        self.deep_tlens = None
        self.deep_nlens = None
        self.deep_mm = None
        if self.has_deep and os.path.isfile(os.path.join(self.mem_dir, "deep_offsets.npy")):
            self.deep_offsets = np.load(os.path.join(self.mem_dir, "deep_offsets.npy"))
            self.deep_tlens = np.load(os.path.join(self.mem_dir, "deep_tlens.npy"))
            self.deep_nlens = np.load(os.path.join(self.mem_dir, "deep_nlens.npy"))
            dd = np.load(os.path.join(self.mem_dir, "deep_dim.npy"))
            self.deep_dim = int(dd[0]) if dd.size else self.deep_dim
            total_deep_elems = int((self.deep_tlens.astype(np.int64) * self.deep_nlens.astype(np.int64) * self.deep_dim).sum())
            self.deep_mm = np.memmap(os.path.join(self.mem_dir, "deep.bin"), mode="r", dtype=np.float16, shape=(max(total_deep_elems, 1),))

        print(f"[ScienceQA MemmapDataset] split={split} N={self.N} mem_dir={self.mem_dir}")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = int(self.offsets[idx])
        L = int(self.lengths[idx])

        out: Dict[str, Any] = {
            "id": self.ids[idx],
            "prompt": self.prompts[idx] if idx < len(self.prompts) else "",
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "input_ids": torch.from_numpy(np.array(self.input_ids_mm[off:off + L], copy=True)).long(),
            "attention_mask": torch.from_numpy(np.array(self.attn_mm[off:off + L], copy=True)).long(),
            "position_ids": torch.from_numpy(np.array(self.pos_mm[off:off + L], copy=True)).long(),
            "image_mask": torch.from_numpy(np.array(self.image_mm[off:off + L], copy=True)).bool(),
            "hint_mask": torch.from_numpy(np.array(self.hint_mm[off:off + L], copy=True)).bool(),
        }

        if self.vision_mm is not None and self.vision_offsets is not None and self.vision_lengths is not None and self.D > 0:
            voff = int(self.vision_offsets[idx])
            nimg = int(self.vision_lengths[idx])
            if nimg <= 0:
                out["vision_embeds"] = torch.empty((0, self.D), dtype=torch.float16 if self.vision_dtype == "float16" else torch.float32)
            else:
                start = voff
                end = voff + nimg * self.D
                flat = np.array(self.vision_mm[start:end], copy=True)
                out["vision_embeds"] = torch.from_numpy(flat).view(nimg, self.D)
            print(out["vision_embeds"].shape)
            out["vision_len"] = torch.tensor(nimg, dtype=torch.long)

        if self.deep_mm is not None and self.deep_offsets is not None and self.deep_tlens is not None and self.deep_nlens is not None:
            doff = int(self.deep_offsets[idx])
            T = int(self.deep_tlens[idx])
            Nimg = int(self.deep_nlens[idx])
            if T <= 0 or Nimg <= 0:
                out["deep_stack_viz"] = torch.empty((0, 0, self.deep_dim), dtype=torch.float16)
            else:
                n_elems = T * Nimg * self.deep_dim
                flat = np.array(self.deep_mm[doff:doff + n_elems], copy=True)
                out["deep_stack_viz"] = torch.from_numpy(flat).view(T, Nimg, self.deep_dim)
            out["deep_t"] = torch.tensor(T, dtype=torch.long)
            out["deep_nimg"] = torch.tensor(Nimg, dtype=torch.long)

        return out


# =========================
# Collate (LEFT pad tokens; pad vision + deep)
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


def _pad_2d_by_rows(seqs: List[torch.Tensor], pad_val: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(seqs)
    if B == 0:
        return torch.empty((0, 0, 0)), torch.empty((0, 0), dtype=torch.bool)

    D = 0
    ref_dtype = torch.float16
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
            x = x.reshape(1, -1)

        if int(x.shape[1]) > D:
            x = x[:, :D]
        elif int(x.shape[1]) < D:
            pad = torch.zeros((int(x.shape[0]), D - int(x.shape[1])), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        n = int(x.shape[0])
        padded[i, :n, :] = x
        mask[i, :n] = True

    return padded, mask


def _pad_deep_3d(
    seqs: List[torch.Tensor],
    deep_dim: int,
    pad_val: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads [T_i, N_i, D] -> [B, Tmax, Nmax, D] and mask [B, Tmax, Nmax]
    """
    B = len(seqs)
    if B == 0:
        return (
            torch.empty((0, 0, 0, deep_dim), dtype=torch.float16),
            torch.empty((0, 0, 0), dtype=torch.bool),
        )

    Tmax = 0
    Nmax = 0
    ref_dtype = torch.float16
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


def scienceqa_memmap_collate(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    ids = [b.get("id", None) for b in batch]
    prompts = [b.get("prompt", "") for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    input_ids = _left_pad_1d([b["input_ids"] for b in batch], pad_val=int(pad_token_id), dtype=torch.long)
    attention_mask = _left_pad_1d([b["attention_mask"] for b in batch], pad_val=0, dtype=torch.long)
    position_ids = _left_pad_1d([b["position_ids"] for b in batch], pad_val=0, dtype=torch.long)

    image_mask = _left_pad_bool([b["image_mask"] for b in batch])
    hint_mask = _left_pad_bool([b["hint_mask"] for b in batch])

    data: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_mask": image_mask,
        "hint_mask": hint_mask,
    }

    if "vision_embeds" in batch[0]:
        vis_list = [b.get("vision_embeds", torch.empty((0, 0), dtype=torch.float16)) for b in batch]
        vis_pad, vis_mask = _pad_2d_by_rows(vis_list, pad_val=0.0)
        data["vision_embeds"] = vis_pad
        data["vision_mask"] = vis_mask

    if "deep_stack_viz" in batch[0]:
        deep_list = [b.get("deep_stack_viz", torch.empty((0, 0, 2048), dtype=torch.float16)) for b in batch]
        deep_pad, deep_mask = _pad_deep_3d(deep_list, deep_dim=int(deep_list[0].shape[-1]) if (torch.is_tensor(deep_list[0]) and deep_list[0].numel() > 0) else 2048)
        data["deep_stack_viz"] = deep_pad
        data["deep_mask"] = deep_mask

    return {"ids": ids, "prompts": prompts, "label": labels, "data": data}


# =========================
# Dataloader wrapper
# =========================

class ScienceQA_MemmapDataloader:
    def __init__(
        self,
        config,
        *,
        input_ids_dtype: str = "int32",
        vision_dtype: str = "float16",
        deep_dim: int = 2048,
        num_workers: Optional[int] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        force_rebuild_memmap: bool = False,
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
            num_workers = max(2, min(8, total_cpus // 8 if total_cpus >= 16 else 4))
            num_workers = 0

        print(
            f"[ScienceQA MemmapLoader] CPUs={multiprocessing.cpu_count()} | num_workers={num_workers} | "
            f"prefetch_factor={prefetch_factor} | persistent_workers={persistent_workers} | pin_memory={pin_memory} | "
            f"vision_dtype={vision_dtype} | deep_dim={deep_dim} | force_rebuild_memmap={force_rebuild_memmap}"
        )

        for split in ("validation", "test"):
            split_dir = os.path.join(cache_root, split)
            if not os.path.isdir(split_dir):
                continue
            mem_dir = os.path.join(split_dir, "_memmap")
            meta_path = os.path.join(mem_dir, "meta.json")
            done_flag = os.path.join(mem_dir, ".DONE")

            if force_rebuild_memmap and os.path.isdir(mem_dir):
                for p in (done_flag, meta_path):
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

            if not (os.path.isfile(meta_path) and os.path.isfile(done_flag)):
                print(f"[ScienceQA MemmapLoader] Building memmap cache for split={split} ...")
                build_memmap_from_token_shards(
                    split_dir=split_dir,
                    out_dir=mem_dir,
                    input_ids_dtype=input_ids_dtype,
                    vision_dtype=vision_dtype,
                    deep_dim=deep_dim,
                )

        self.collate_fn = lambda batch: scienceqa_memmap_collate(batch, pad_token_id=pad_token_id)

        def make_loader(split: str, shuffle: bool):
            ds = ScienceQA_MemmapDataset(cache_root=cache_root, split=split)
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=g if shuffle else None,
                worker_init_fn=seed_worker,
                collate_fn=self.collate_fn,
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory),
                prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
                persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
            )

        self.train_loader = make_loader("validation", shuffle=True) if os.path.isdir(os.path.join(cache_root, "validation")) else None
        self.valid_loader = make_loader("validation", shuffle=False) if os.path.isdir(os.path.join(cache_root, "validation")) else None
        self.test_loader = make_loader("test", shuffle=False) if os.path.isdir(os.path.join(cache_root, "test")) else None
