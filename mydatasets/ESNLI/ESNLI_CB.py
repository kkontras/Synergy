"""
ESNLI-VE Qwen3-VL codebook memmap loader (low-RAM, worker-friendly) + token masks
UPDATED to match your cached item keys:

Per-example keys now expected (best-effort, NO strict format checks):
  - id
  - label
  - prompt
  - input_ids
  - attention_mask
  - position_ids
  - deep_stack_viz
  - masks: {"image": ..., "text": ...}
  - pixel_values (optional, if you enabled it)
  - image_grid_thw (optional)
  - vision_embeds (optional)

Memmap cache:
  <cache_root>/<split>/_memmap/
    meta.json
    offsets.npy         int64 [N]
    lengths.npy         int32 [N]
    labels.npy          int64 [N]
    ids.jsonl           N lines JSON
    prompts.jsonl       N lines JSON (string or any JSON-serializable)
    input_ids.bin       int32/int64 concatenated tokens
    attention.bin       uint8 concatenated 0/1
    position_ids.bin    int32 concatenated
    deep_stack_viz.bin  float16 concatenated
    image_mask.bin      uint8 concatenated 0/1
    text_mask.bin       uint8 concatenated 0/1
    image_grid_thw.npy  int32 [N, 3] (if present)
    vision_offsets.npy  int64 [N]
    vision_lengths.npy  int32 [N]
    vision_dim.npy      int32 scalar
    vision.bin          float16/float32 concatenated
    pixel_values.bin    float16 [N, C, H, W] (optional)

Returned per item:
  - id
  - label
  - prompt
  - input_ids
  - attention_mask
  - position_ids
  - deep_stack_viz
  - image_mask (bool)
  - text_mask  (bool)
  - image_grid_thw (optional)
  - vision_embeds + vision_mask (optional)
  - pixel_values (optional)

Usage:
  from esnli_memmap_loader_updated import ESNLI_MemmapDataloader
"""

import os
import json
import time
import random
import multiprocessing
from typing import Dict, Any, List, Optional, Tuple

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



def _as_bool_1d_cpu_tensor(x, L: int) -> torch.Tensor:
    """Best-effort bool mask length L. Missing -> zeros. Wrong length -> pad/crop."""
    if x is None:
        return torch.zeros((L,), dtype=torch.bool)
    t = _as_1d_cpu_tensor(x, dtype=torch.bool)
    if t.numel() == L:
        return t
    if t.numel() == 0:
        return torch.zeros((L,), dtype=torch.bool)
    if t.numel() > L:
        return t[:L]
    # pad left with zeros to match L (aligns with left-pad collation)
    out = torch.zeros((L,), dtype=torch.bool)
    out[-t.numel():] = t
    return out


def _as_int_1d_cpu_tensor(x, L: int, dtype: torch.dtype) -> torch.Tensor:
    """Best-effort int tensor length L. Missing -> range(L) for position_ids else zeros."""
    if x is None:
        return torch.empty((0,), dtype=dtype)
    t = _as_1d_cpu_tensor(x, dtype=dtype)
    if t.numel() == L:
        return t
    if t.numel() == 0:
        return torch.zeros((L,), dtype=dtype)
    if t.numel() > L:
        return t[:L]
    out = torch.zeros((L,), dtype=dtype)
    out[-t.numel():] = t
    return out


def _as_float_1d_cpu_tensor(x, L: int, dtype: torch.dtype) -> torch.Tensor:
    """Best-effort float tensor length L. Missing -> zeros. Wrong length -> pad/crop."""
    if x is None:
        return torch.zeros((L,), dtype=dtype)
    t = _as_1d_cpu_tensor(x, dtype=dtype)
    if t.numel() == L:
        return t
    if t.numel() == 0:
        return torch.zeros((L,), dtype=dtype)
    if t.numel() > L:
        return t[:L]
    out = torch.zeros((L,), dtype=dtype)
    out[-t.numel():] = t
    return out


# =========================
# Memmap builder
# =========================

def build_memmap_from_token_shards(
    *,
    split_dir: str,
    out_dir: str,
    input_ids_dtype: str = "int32",
    vision_dtype: str = "float16",
    store_pixel_values: bool = False,
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

        input_np_dtype = np.int32 if input_ids_dtype == "int32" else np.int64
        vision_np_dtype = np.float16 if vision_dtype == "float16" else np.float32

        # -------- Pass 1: compute totals + discover shapes (best-effort) --------
        total_tokens = 0
        has_grid = False
        has_vision = False
        has_pixel = False

        vision_dim: Optional[int] = None
        total_vision_elems = 0

        pixel_shape: Optional[Tuple[int, int, int]] = None  # (C,H,W)

        for r in shard_recs:
            items = torch.load(r["path"], map_location="cpu")
            for ex in items:
                inp = _as_1d_cpu_tensor(ex.get("input_ids", None), dtype=torch.long)
                L = int(inp.numel())
                total_tokens += L

                # image_grid_thw presence
                if ex.get("image_grid_thw", None) is not None:
                    has_grid = True

                # vision_embeds presence
                ve = ex.get("vision_embeds", None)
                if ve is not None:
                    ve_t = _as_1d_cpu_tensor(ve) if (not torch.is_tensor(ve)) else ve.detach().cpu()
                    if ve_t.numel() > 0:
                        has_vision = True
                        if ve_t.dim() == 1:
                            # treat as single vector
                            D = int(ve_t.numel())
                            nimg = 1
                        else:
                            # best-effort: if not 2D, flatten and treat as one vector
                            if ve_t.dim() != 2:
                                ve_t = ve_t.view(1, -1)
                            nimg, D = int(ve_t.shape[0]), int(ve_t.shape[1])
                        if vision_dim is None and D > 0:
                            vision_dim = D
                        # accumulate elems using chosen/observed D (if unknown yet, use observed)
                        D_use = int(vision_dim) if (vision_dim is not None and vision_dim > 0) else D
                        total_vision_elems += int(nimg * D_use)

                # pixel_values presence (optional)
                if store_pixel_values:
                    pv = ex.get("pixel_values", None)
                    if pv is not None:
                        pv_t = pv.detach().cpu() if torch.is_tensor(pv) else torch.as_tensor(pv).detach().cpu()
                        # accept [C,H,W] or [H,W,C] or [H, W*3] etc; best-effort: just flatten if weird
                        if pv_t.dim() == 3:
                            C, H, W = int(pv_t.shape[0]), int(pv_t.shape[1]), int(pv_t.shape[2])
                            if pixel_shape is None:
                                pixel_shape = (C, H, W)
                            has_pixel = True

        if has_vision and (vision_dim is None or vision_dim <= 0):
            # allow empty vision: set D=1 to keep files consistent
            vision_dim = 1

        # -------- Allocate index arrays --------
        offsets = np.zeros((N,), dtype=np.int64)
        lengths = np.zeros((N,), dtype=np.int32)
        labels = np.zeros((N,), dtype=np.int64)

        grid_thw = np.zeros((N, 3), dtype=np.int32) if has_grid else None

        vision_offsets = np.zeros((N,), dtype=np.int64) if has_vision else None
        vision_lengths = np.zeros((N,), dtype=np.int32) if has_vision else None

        # ids + prompts files
        ids_path = os.path.join(out_dir, "ids.jsonl")
        prompts_path = os.path.join(out_dir, "prompts.jsonl")
        ids_f = open(ids_path + ".tmp", "w", encoding="utf-8")
        prompts_f = open(prompts_path + ".tmp", "w", encoding="utf-8")

        # -------- Allocate memmaps --------
        input_ids_bin = os.path.join(out_dir, "input_ids.bin")
        attn_bin = os.path.join(out_dir, "attention.bin")
        pos_bin = os.path.join(out_dir, "position_ids.bin")
        deep_bin = os.path.join(out_dir, "deep_stack_viz.bin")
        image_mask_bin = os.path.join(out_dir, "image_mask.bin")
        text_mask_bin = os.path.join(out_dir, "text_mask.bin")

        input_ids_mm = np.memmap(input_ids_bin, mode="w+", dtype=input_np_dtype, shape=(total_tokens,))
        attn_mm = np.memmap(attn_bin, mode="w+", dtype=np.uint8, shape=(total_tokens,))
        pos_mm = np.memmap(pos_bin, mode="w+", dtype=np.int32, shape=(total_tokens,))
        deep_mm = np.memmap(deep_bin, mode="w+", dtype=np.float16, shape=(total_tokens,))
        image_mask_mm = np.memmap(image_mask_bin, mode="w+", dtype=np.uint8, shape=(total_tokens,))
        text_mask_mm = np.memmap(text_mask_bin, mode="w+", dtype=np.uint8, shape=(total_tokens,))

        vision_mm = None
        if has_vision:
            vision_bin = os.path.join(out_dir, "vision.bin")
            vision_mm = np.memmap(vision_bin, mode="w+", dtype=vision_np_dtype, shape=(max(int(total_vision_elems), 1),))

        pixel_mm = None
        if store_pixel_values and has_pixel and (pixel_shape is not None):
            C, H, W = pixel_shape
            pixel_bin = os.path.join(out_dir, "pixel_values.bin")
            pixel_mm = np.memmap(pixel_bin, mode="w+", dtype=np.float16, shape=(N, C, H, W))

        # -------- Pass 2: fill --------
        token_cursor = 0
        vision_cursor = 0
        i = 0

        D_use = int(vision_dim) if has_vision else 0

        for r in shard_recs:
            items = torch.load(r["path"], map_location="cpu")
            for ex in items:
                inp = _as_1d_cpu_tensor(ex.get("input_ids", None), dtype=torch.long)
                L = int(inp.numel())

                att = _as_1d_cpu_tensor(ex.get("attention_mask", None), dtype=torch.long)
                if int(att.numel()) != L:
                    # best-effort fallback: 1s for tokens, 0 for empty
                    att = torch.ones((L,), dtype=torch.long) if L > 0 else torch.empty((0,), dtype=torch.long)

                pos = _as_1d_cpu_tensor(ex.get("position_ids", None), dtype=torch.long)
                if int(pos.numel()) != L:
                    # fallback: simple 0..L-1
                    pos = torch.arange(L, dtype=torch.long) if L > 0 else torch.empty((0,), dtype=torch.long)

                deep = _as_1d_cpu_tensor(ex.get("deep_stack_viz", None), dtype=torch.float16)
                if int(deep.numel()) != L:
                    deep = torch.zeros((L,), dtype=torch.float16) if L > 0 else torch.empty((0,), dtype=torch.float16)

                masks = ex.get("masks", {}) if isinstance(ex.get("masks", {}), dict) else {}
                imgm = _as_bool_1d_cpu_tensor(masks.get("image", None), L)
                txtm = _as_bool_1d_cpu_tensor(masks.get("text", None), L)

                offsets[i] = token_cursor
                lengths[i] = L

                lab = ex.get("label", 0)
                if isinstance(lab, torch.Tensor):
                    lab = lab.detach().cpu().view(-1)
                    lab = int(lab[0].item()) if lab.numel() > 0 else 0
                try:
                    labels[i] = int(lab)
                except Exception:
                    labels[i] = 0

                # id + prompt
                ids_f.write(json.dumps(ex.get("id", i)) + "\n")
                prompts_f.write(json.dumps(ex.get("prompt", "")) + "\n")

                # write token-aligned arrays
                if L > 0:
                    input_ids_mm[token_cursor:token_cursor + L] = inp.detach().cpu().numpy().astype(input_np_dtype, copy=False)
                    attn_mm[token_cursor:token_cursor + L] = att.detach().cpu().numpy().astype(np.uint8, copy=False)
                    pos_mm[token_cursor:token_cursor + L] = pos.detach().cpu().numpy().astype(np.int32, copy=False)
                    deep_mm[token_cursor:token_cursor + L] = deep.detach().cpu().numpy().astype(np.float16, copy=False)
                    image_mask_mm[token_cursor:token_cursor + L] = imgm.detach().cpu().numpy().astype(np.uint8, copy=False)
                    text_mask_mm[token_cursor:token_cursor + L] = txtm.detach().cpu().numpy().astype(np.uint8, copy=False)

                # grid
                if has_grid and grid_thw is not None:
                    g = ex.get("image_grid_thw", None)
                    if g is None:
                        grid_thw[i, :] = 0
                    else:
                        try:
                            if isinstance(g, torch.Tensor):
                                g = g.detach().cpu().view(-1).tolist()
                            g = list(g)
                            if len(g) >= 3:
                                grid_thw[i, :] = np.array(g[:3], dtype=np.int32)
                            else:
                                tmp = [0, 0, 0]
                                for k in range(len(g)):
                                    tmp[k] = int(g[k])
                                grid_thw[i, :] = np.array(tmp, dtype=np.int32)
                        except Exception:
                            grid_thw[i, :] = 0

                # vision embeds
                if has_vision and vision_offsets is not None and vision_lengths is not None and vision_mm is not None:
                    ve = ex.get("vision_embeds", None)
                    if ve is None:
                        vision_offsets[i] = vision_cursor
                        vision_lengths[i] = 0
                    else:
                        ve_t = ve.detach().cpu() if torch.is_tensor(ve) else torch.as_tensor(ve).detach().cpu()
                        if ve_t.numel() == 0:
                            vision_offsets[i] = vision_cursor
                            vision_lengths[i] = 0
                        else:
                            if ve_t.dim() == 1:
                                ve_t = ve_t.view(1, -1)
                            elif ve_t.dim() != 2:
                                ve_t = ve_t.view(1, -1)

                            nimg = int(ve_t.shape[0])
                            D_obs = int(ve_t.shape[1])

                            # best-effort: adjust to D_use
                            if D_obs == D_use:
                                ve_use = ve_t
                            elif D_obs > D_use:
                                ve_use = ve_t[:, :D_use]
                            else:
                                pad = torch.zeros((nimg, D_use - D_obs), dtype=ve_t.dtype)
                                ve_use = torch.cat([ve_t, pad], dim=1)

                            vision_offsets[i] = vision_cursor
                            vision_lengths[i] = nimg

                            flat = ve_use.detach().cpu().numpy().astype(vision_np_dtype, copy=False).reshape(-1)
                            n_elems = int(flat.size)
                            vision_mm[vision_cursor:vision_cursor + n_elems] = flat
                            vision_cursor += n_elems

                # pixel_values (optional)
                if pixel_mm is not None:
                    pv = ex.get("pixel_values", None)
                    if pv is None:
                        pixel_mm[i, ...] = 0
                    else:
                        try:
                            arr = pv.detach().cpu().numpy() if torch.is_tensor(pv) else np.asarray(pv)
                            arr = arr.astype(np.float16, copy=False)
                            C, H, W = pixel_mm.shape[1], pixel_mm.shape[2], pixel_mm.shape[3]
                            # best-effort: coerce to [C,H,W]
                            if arr.ndim == 3:
                                if arr.shape[0] == C:
                                    use = arr
                                elif arr.shape[-1] == C:
                                    use = np.transpose(arr, (2, 0, 1))
                                else:
                                    use = np.zeros((C, H, W), dtype=np.float16)
                            else:
                                use = np.zeros((C, H, W), dtype=np.float16)

                            # crop/pad spatial
                            use2 = np.zeros((C, H, W), dtype=np.float16)
                            cmin = min(C, use.shape[0])
                            hmin = min(H, use.shape[1])
                            wmin = min(W, use.shape[2])
                            use2[:cmin, :hmin, :wmin] = use[:cmin, :hmin, :wmin]
                            pixel_mm[i, ...] = use2
                        except Exception:
                            pixel_mm[i, ...] = 0

                token_cursor += L
                i += 1

        # Flush
        input_ids_mm.flush()
        attn_mm.flush()
        pos_mm.flush()
        deep_mm.flush()
        image_mask_mm.flush()
        text_mask_mm.flush()
        if vision_mm is not None:
            vision_mm.flush()
        if pixel_mm is not None:
            pixel_mm.flush()

        ids_f.close()
        prompts_f.close()
        os.replace(ids_path + ".tmp", ids_path)
        os.replace(prompts_path + ".tmp", prompts_path)

        # Save index arrays
        np.save(os.path.join(out_dir, "offsets.npy"), offsets)
        np.save(os.path.join(out_dir, "lengths.npy"), lengths)
        np.save(os.path.join(out_dir, "labels.npy"), labels)

        if has_grid and grid_thw is not None:
            np.save(os.path.join(out_dir, "image_grid_thw.npy"), grid_thw)

        if has_vision and vision_offsets is not None and vision_lengths is not None:
            np.save(os.path.join(out_dir, "vision_offsets.npy"), vision_offsets)
            np.save(os.path.join(out_dir, "vision_lengths.npy"), vision_lengths)
            np.save(os.path.join(out_dir, "vision_dim.npy"), np.array([int(D_use)], dtype=np.int32))

        meta = {
            "version": 3,
            "N": int(N),
            "total_tokens": int(total_tokens),
            "input_ids_dtype": input_ids_dtype,
            "has_image_grid_thw": bool(has_grid),
            "has_vision_embeds": bool(has_vision),
            "vision_dtype": vision_dtype if has_vision else None,
            "vision_dim": int(D_use) if has_vision else None,
            "store_pixel_values": bool(store_pixel_values and (pixel_mm is not None)),
            "pixel_shape": list(pixel_shape) if (pixel_mm is not None and pixel_shape is not None) else None,
            "has_token_masks": True,
            "has_position_ids": True,
            "has_deep_stack_viz": True,
            "has_prompts": True,
            "paths": {
                "offsets": "offsets.npy",
                "lengths": "lengths.npy",
                "labels": "labels.npy",
                "ids": "ids.jsonl",
                "prompts": "prompts.jsonl",
                "input_ids": "input_ids.bin",
                "attention": "attention.bin",
                "position_ids": "position_ids.bin",
                "deep_stack_viz": "deep_stack_viz.bin",
                "image_mask": "image_mask.bin",
                "text_mask": "text_mask.bin",
                "image_grid_thw": "image_grid_thw.npy" if has_grid else None,
                "vision_offsets": "vision_offsets.npy" if has_vision else None,
                "vision_lengths": "vision_lengths.npy" if has_vision else None,
                "vision_dim": "vision_dim.npy" if has_vision else None,
                "vision": "vision.bin" if has_vision else None,
                "pixel_values": "pixel_values.bin" if (pixel_mm is not None) else None,
            },
        }
        _atomic_write_json(os.path.join(out_dir, "meta.json"), meta)

        with open(done_flag, "w", encoding="utf-8") as f:
            f.write("ok\n")

        print(
            f"[ESNLI Memmap] Built: {out_dir} | N={N} | total_tokens={total_tokens} | "
            f"masks=True | position_ids=True | deep_stack_viz=True | vision={has_vision} | "
            f"pixel_values={bool(pixel_mm is not None)}"
        )

    finally:
        _release_lock(lock_path)


# =========================
# Dataset (memmap)
# =========================

class ESNLI_MemmapDataset(Dataset):
    """
    Loads ESNLI codebook examples from memmap.
    Returns per item:
      - id
      - label
      - prompt
      - input_ids (1D long)
      - attention_mask (1D long)
      - position_ids (1D long)
      - deep_stack_viz (1D float16)
      - image_mask (bool 1D)
      - text_mask (bool 1D)
      - image_grid_thw (long[3]) if present
      - vision_embeds (float [Nimg, D]) if present
      - pixel_values (float16 [C,H,W]) if stored
    """

    def __init__(self, cache_root: str, split: str):
        super().__init__()
        self.split_dir = os.path.join(cache_root, split)
        self.mem_dir = os.path.join(self.split_dir, "_memmap")
        meta_path = os.path.join(self.mem_dir, "meta.json")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.N = int(meta.get("N", 0))
        self.total_tokens = int(meta.get("total_tokens", 0))
        self.input_ids_dtype = meta.get("input_ids_dtype", "int32")

        self.has_grid = bool(meta.get("has_image_grid_thw", False))
        self.has_vision = bool(meta.get("has_vision_embeds", False))
        self.vision_dtype = meta.get("vision_dtype", "float16")
        self.vision_dim = meta.get("vision_dim", None)

        self.store_pixel_values = bool(meta.get("store_pixel_values", False))
        self.pixel_shape = meta.get("pixel_shape", None)

        # small index arrays in RAM
        self.offsets = np.load(os.path.join(self.mem_dir, "offsets.npy"))
        self.lengths = np.load(os.path.join(self.mem_dir, "lengths.npy"))
        self.labels = np.load(os.path.join(self.mem_dir, "labels.npy"))

        # ids + prompts in RAM
        self.ids: List[Any] = []
        with open(os.path.join(self.mem_dir, "ids.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                self.ids.append(json.loads(line))

        self.prompts: List[Any] = []
        prompts_path = os.path.join(self.mem_dir, "prompts.jsonl")
        if os.path.isfile(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.prompts.append(json.loads(line))
        else:
            self.prompts = [""] * self.N

        # memmaps (token-aligned)
        input_dtype = np.int32 if self.input_ids_dtype == "int32" else np.int64

        self.input_ids_mm = np.memmap(
            os.path.join(self.mem_dir, "input_ids.bin"),
            mode="r",
            dtype=input_dtype,
            shape=(self.total_tokens,),
        )
        self.attn_mm = np.memmap(
            os.path.join(self.mem_dir, "attention.bin"),
            mode="r",
            dtype=np.uint8,
            shape=(self.total_tokens,),
        )
        self.pos_mm = np.memmap(
            os.path.join(self.mem_dir, "position_ids.bin"),
            mode="r",
            dtype=np.int32,
            shape=(self.total_tokens,),
        )
        self.deep_mm = np.memmap(
            os.path.join(self.mem_dir, "deep_stack_viz.bin"),
            mode="r",
            dtype=np.float16,
            shape=(self.total_tokens,),
        )
        self.image_mask_mm = np.memmap(
            os.path.join(self.mem_dir, "image_mask.bin"),
            mode="r",
            dtype=np.uint8,
            shape=(self.total_tokens,),
        )
        self.text_mask_mm = np.memmap(
            os.path.join(self.mem_dir, "text_mask.bin"),
            mode="r",
            dtype=np.uint8,
            shape=(self.total_tokens,),
        )

        # optional grid
        self.grid_thw = None
        if self.has_grid and os.path.isfile(os.path.join(self.mem_dir, "image_grid_thw.npy")):
            self.grid_thw = np.load(os.path.join(self.mem_dir, "image_grid_thw.npy"))

        # optional vision
        self.vision_offsets = None
        self.vision_lengths = None
        self.D = 0
        self.vision_mm = None
        if self.has_vision and os.path.isfile(os.path.join(self.mem_dir, "vision_offsets.npy")):
            self.vision_offsets = np.load(os.path.join(self.mem_dir, "vision_offsets.npy"))
            self.vision_lengths = np.load(os.path.join(self.mem_dir, "vision_lengths.npy"))
            vd = np.load(os.path.join(self.mem_dir, "vision_dim.npy"))
            self.D = int(vd[0]) if vd.size > 0 else int(self.vision_dim or 0)

            v_dtype = np.float16 if self.vision_dtype == "float16" else np.float32
            total_vision_elems = int((self.vision_lengths.astype(np.int64) * max(self.D, 1)).sum())
            total_vision_elems = max(total_vision_elems, 1)

            self.vision_mm = np.memmap(
                os.path.join(self.mem_dir, "vision.bin"),
                mode="r",
                dtype=v_dtype,
                shape=(total_vision_elems,),
            )

        # optional pixel
        self.pixel_mm = None
        if self.store_pixel_values and self.pixel_shape is not None:
            C, H, W = self.pixel_shape
            pv_path = os.path.join(self.mem_dir, "pixel_values.bin")
            if os.path.isfile(pv_path):
                self.pixel_mm = np.memmap(
                    pv_path,
                    mode="r",
                    dtype=np.float16,
                    shape=(self.N, int(C), int(H), int(W)),
                )

        print(f"[ESNLI MemmapDataset] split={split} N={self.N} mem_dir={self.mem_dir}")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = int(self.offsets[idx])
        L = int(self.lengths[idx])

        input_ids = torch.from_numpy(np.array(self.input_ids_mm[off:off + L], copy=True)).to(torch.long)
        attention_mask = torch.from_numpy(np.array(self.attn_mm[off:off + L], copy=True)).to(torch.long)
        position_ids = torch.from_numpy(np.array(self.pos_mm[off:off + L], copy=True)).to(torch.long)
        deep_stack_viz = torch.from_numpy(np.array(self.deep_mm[off:off + L], copy=True)).to(torch.float16)

        image_mask = torch.from_numpy(np.array(self.image_mask_mm[off:off + L], copy=True)).to(torch.bool)
        text_mask = torch.from_numpy(np.array(self.text_mask_mm[off:off + L], copy=True)).to(torch.bool)

        out: Dict[str, Any] = {
            "id": self.ids[idx],
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "prompt": self.prompts[idx] if idx < len(self.prompts) else "",
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "deep_stack_viz": deep_stack_viz,
            "image_mask": image_mask,
            "text_mask": text_mask,
        }

        if self.grid_thw is not None:
            out["image_grid_thw"] = torch.tensor(self.grid_thw[idx], dtype=torch.long)

        if self.vision_mm is not None and self.vision_offsets is not None and self.vision_lengths is not None and self.D > 0:
            voff = int(self.vision_offsets[idx])
            nimg = int(self.vision_lengths[idx])
            if nimg <= 0:
                vision = torch.empty((0, self.D), dtype=torch.float16 if self.vision_dtype == "float16" else torch.float32)
            else:
                start = voff
                end = voff + nimg * self.D
                flat = np.array(self.vision_mm[start:end], copy=True)
                vision = torch.from_numpy(flat).view(nimg, self.D)
            out["vision_embeds"] = vision
            out["vision_len"] = torch.tensor(int(nimg), dtype=torch.long)

        if self.pixel_mm is not None:
            out["pixel_values"] = torch.from_numpy(np.array(self.pixel_mm[idx], copy=True))  # float16

        print(out.keys())
        return out


# =========================
# Collate (LEFT pad text; pad masks + aligned arrays)
# =========================

def _left_pad_1d(seqs: List[torch.Tensor], pad_val: int, dtype: torch.dtype) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs) if len(seqs) > 0 else 0
    out = torch.full((len(seqs), max_len), pad_val, dtype=dtype)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        if L > 0:
            out[i, -L:] = s.to(dtype)
    return out


def _left_pad_float(seqs: List[torch.Tensor], pad_val: float, dtype: torch.dtype) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs) if len(seqs) > 0 else 0
    out = torch.full((len(seqs), max_len), float(pad_val), dtype=dtype)
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
    """
    Pads a list of [Ni, D] tensors to [B, Nmax, D].
    Returns (padded, mask) where mask is [B, Nmax] bool (True for real rows).
    """
    B = len(seqs)
    D = int(seqs[0].shape[1]) if (B > 0 and seqs[0].dim() == 2) else 0
    Nmax = max(int(x.shape[0]) for x in seqs) if B > 0 else 0

    if B == 0:
        return torch.empty((0, 0, 0)), torch.empty((0, 0), dtype=torch.bool)

    padded = torch.full((B, Nmax, D), float(pad_val), dtype=seqs[0].dtype)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)

    for i, x in enumerate(seqs):
        n = int(x.shape[0])
        if n > 0:
            padded[i, :n, :] = x
            mask[i, :n] = True

    return padded, mask

def _as_1d_cpu_tensor(x, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Best-effort: turn x into 1D CPU tensor. Missing/None -> empty."""
    if x is None:
        return torch.empty((0,), dtype=dtype or torch.long)
    if torch.is_tensor(x):
        # reshape handles non-contiguous tensors safely
        t = x.detach().cpu().reshape(-1)
        return t.to(dtype) if dtype is not None else t
    try:
        t = torch.as_tensor(x).detach().cpu().reshape(-1)
        return t.to(dtype) if dtype is not None else t
    except Exception:
        return torch.empty((0,), dtype=dtype or torch.long)


def esnli_memmap_collate(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    ids = [b.get("id", None) for b in batch]
    prompts = [b.get("prompt", "") for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    input_ids = _left_pad_1d([b["input_ids"] for b in batch], pad_val=int(pad_token_id), dtype=torch.long)
    attention_mask = _left_pad_1d([b["attention_mask"] for b in batch], pad_val=0, dtype=torch.long)
    position_ids = _left_pad_1d([b["position_ids"] for b in batch], pad_val=0, dtype=torch.long)
    deep_stack_viz = _left_pad_float([b["deep_stack_viz"] for b in batch], pad_val=0.0, dtype=torch.float16)

    image_mask = _left_pad_bool([b["image_mask"] for b in batch])
    text_mask = _left_pad_bool([b["text_mask"] for b in batch])

    data: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "deep_stack_viz": deep_stack_viz,
        "image_mask": image_mask,
        "text_mask": text_mask,
    }

    if "image_grid_thw" in batch[0]:
        try:
            data["image_grid_thw"] = torch.stack([b.get("image_grid_thw", torch.zeros(3, dtype=torch.long)) for b in batch], dim=0)
        except Exception:
            pass

    if "pixel_values" in batch[0]:
        try:
            data["pixel_values"] = torch.stack([b["pixel_values"] for b in batch], dim=0)
        except Exception:
            pass

    if "vision_embeds" in batch[0]:
        vis_list = [b.get("vision_embeds", torch.empty((0, 0), dtype=torch.float16)) for b in batch]
        non_empty = [v for v in vis_list if (torch.is_tensor(v) and v.numel() > 0)]
        if len(non_empty) == 0:
            data["vision_embeds"] = torch.empty((len(batch), 0, 0), dtype=torch.float16)
            data["vision_mask"] = torch.empty((len(batch), 0), dtype=torch.bool)
        else:
            D = int(non_empty[0].shape[1]) if non_empty[0].dim() == 2 else 0
            fixed = []
            for v in vis_list:
                if not torch.is_tensor(v) or v.numel() == 0:
                    fixed.append(torch.empty((0, D), dtype=non_empty[0].dtype))
                else:
                    if v.dim() == 1:
                        v = v.view(1, -1)
                    if v.dim() != 2:
                        v = v.view(1, -1)
                    # best-effort: adjust D
                    if int(v.shape[1]) > D:
                        v = v[:, :D]
                    elif int(v.shape[1]) < D:
                        pad = torch.zeros((int(v.shape[0]), D - int(v.shape[1])), dtype=v.dtype)
                        v = torch.cat([v, pad], dim=1)
                    fixed.append(v)
            vis_pad, vis_mask = _pad_2d_by_rows(fixed, pad_val=0.0)
            data["vision_embeds"] = vis_pad
            data["vision_mask"] = vis_mask

    return {"ids": ids, "prompts": prompts, "label": labels, "data": data}


# =========================
# Dataloader wrapper
# =========================

class ESNLI_MemmapDataloader:
    """
    Builds memmap caches if needed (one-time), then exposes train/validation/test loaders.

    Expects config to have:
      config.dataset.cache_root
      config.training_params.batch_size
      optionally config.model.pad_token_id (else default 0)
    """

    def __init__(
        self,
        config,
        *,
        input_ids_dtype: str = "int32",
        vision_dtype: str = "float16",
        store_pixel_values: bool = False,
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

        print(
            f"[ESNLI MemmapLoader] CPUs={multiprocessing.cpu_count()} | num_workers={num_workers} | "
            f"prefetch_factor={prefetch_factor} | persistent_workers={persistent_workers} | pin_memory={pin_memory} | "
            f"store_pixel_values={store_pixel_values} | force_rebuild_memmap={force_rebuild_memmap}"
        )

        # Build memmap BEFORE DataLoader workers
        for split in ("validation", "test"):
            split_dir = os.path.join(cache_root, split)
            mem_dir = os.path.join(split_dir, "_memmap")
            meta_path = os.path.join(mem_dir, "meta.json")
            done_flag = os.path.join(mem_dir, ".DONE")

            if force_rebuild_memmap and os.path.isdir(mem_dir):
                for p in [done_flag, meta_path]:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

            if os.path.isdir(split_dir) and (not (os.path.isfile(meta_path) and os.path.isfile(done_flag))):
                print(f"[ESNLI MemmapLoader] Building memmap cache for split={split} ...")
                build_memmap_from_token_shards(
                    split_dir=split_dir,
                    out_dir=mem_dir,
                    input_ids_dtype=input_ids_dtype,
                    vision_dtype=vision_dtype,
                    store_pixel_values=store_pixel_values,
                )

        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            return esnli_memmap_collate(batch, pad_token_id=pad_token_id)

        self.collate_fn = collate_fn

        def make_loader(split: str, shuffle: bool):
            ds = ESNLI_MemmapDataset(cache_root=cache_root, split=split)
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

        # Standard split mapping
        self.train_loader = make_loader("validation", shuffle=True) if os.path.isdir(os.path.join(cache_root, "validation")) else None
        self.valid_loader = make_loader("validation", shuffle=False) if os.path.isdir(os.path.join(cache_root, "validation")) else None
        self.test_loader = make_loader("test", shuffle=False) if os.path.isdir(os.path.join(cache_root, "test")) else None
