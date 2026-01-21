#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESNLI-VE cache builder for Qwen/Qwen3-VL-2B-Instruct
- NO <CLS> in cached tokens (you append later)
- YES vision embeddings cached (vision tower output)

Defaults are set to YOUR paths:
  data_root         = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI"
  flickr_images_dir = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI"
  output_dir        = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_qwen3_vl_2b_nocls_vis"

Run:
  python esnli_cache_qwen3vl_2b_nocls_vis.py --split train

If you OOM:
  python esnli_cache_qwen3vl_2b_nocls_vis.py --split train --batch_size 2 --shard_size 500

"""

import os
import json
import glob
import zipfile
import random
import argparse
import logging
import urllib.request
import multiprocessing
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration  # <-- FIX


# -----------------------------
# Defaults (YOUR PATHS)
# -----------------------------
DEFAULT_DATA_ROOT = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI"
DEFAULT_FLICKR_DIR = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATA_ROOT, "cache_qwen3_vl_2b_nocls_vis")


# -----------------------------
# Labels
# -----------------------------
LABEL2IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


# -----------------------------
# Download / repo handling
# -----------------------------
def _download_url(url: str, dst_path: str, logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0:
        logger.info(f"[download] already exists: {dst_path}")
        return
    logger.info(f"[download] {url} -> {dst_path}")
    urllib.request.urlretrieve(url, dst_path)


def ensure_esnli_repo(cache_root: str, source: str, logger: logging.Logger) -> str:
    """
    Downloads and extracts a repo zip into cache_root if needed.
    Returns extracted folder path.
    """
    if source == "evil":
        zip_url = "https://github.com/multimodal-ai-lab/e-ViL/archive/refs/heads/main.zip"
        zip_name = "e-ViL-main.zip"
        extracted_folder_name = "e-ViL-main"
    elif source == "virginie":
        zip_url = "https://github.com/virginie-do/e-SNLI-VE/archive/refs/heads/master.zip"
        zip_name = "e-SNLI-VE-master.zip"
        extracted_folder_name = "e-SNLI-VE-master"
    else:
        raise ValueError(f"Unknown source={source!r}. Use 'evil' or 'virginie'.")

    zip_path = os.path.join(cache_root, zip_name)
    extract_root = os.path.join(cache_root, extracted_folder_name)

    if os.path.isdir(extract_root) and len(os.listdir(extract_root)) > 0:
        logger.info(f"[GitHub] Found extracted repo at: {extract_root}")
        return extract_root

    _download_url(zip_url, zip_path, logger)
    logger.info(f"[GitHub] Extracting: {zip_path} -> {cache_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_root)

    if not os.path.isdir(extract_root):
        raise RuntimeError(f"Could not find extracted repo folder at: {extract_root}")
    return extract_root


def find_esnli_ve_split_files(repo_root: str) -> Dict[str, str]:
    """
    Best-effort search for train/dev/test files in the extracted repo.
    """
    candidates = glob.glob(os.path.join(repo_root, "**", "*.*"), recursive=True)
    split_paths: Dict[str, str] = {}

    for split in ["train", "dev", "test"]:
        best = []
        for p in candidates:
            fn = os.path.basename(p).lower()
            if split in fn and ("snli" in fn) and fn.endswith((".csv", ".tsv", ".jsonl", ".json")):
                best.append(p)
        if best:
            best.sort(
                key=lambda x: (
                    0 if x.lower().endswith(".csv") else 1 if x.lower().endswith(".tsv") else 2,
                    len(x),
                )
            )
            split_paths[split] = best[0]

    missing = [s for s in ["train", "dev", "test"] if s not in split_paths]
    if missing:
        raise RuntimeError(f"Could not find split files for {missing} inside {repo_root}. Found: {split_paths}")
    return split_paths


def load_split_rows(cache_root: str, split: str, source: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    repo_root = ensure_esnli_repo(cache_root=cache_root, source=source, logger=logger)
    split_files = find_esnli_ve_split_files(repo_root)
    path = split_files[split]
    logger.info(f"[e-SNLI-VE] Loading split={split} from: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        rows = df.to_dict(orient="records")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
        rows = df.to_dict(orient="records")
    elif ext == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext == ".json":
        obj = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            rows = obj.get("data", obj.get("examples", []))
        else:
            raise ValueError(f"Unexpected JSON type in {path}: {type(obj)}")
    else:
        raise ValueError(f"Unsupported split file extension: {ext}")

    logger.info(f"[e-SNLI-VE] Loaded {len(rows)} rows")
    return rows


# -----------------------------
# Small utils
# -----------------------------
def pick_first(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def label_to_index(gold: Any) -> int:
    g = str(gold).strip().lower()
    if g not in LABEL2IDX:
        raise ValueError(f"Bad label: {gold}")
    return LABEL2IDX[g]


def find_image_path(fid: str, flickr_images_dir: str) -> Optional[str]:
    """
    fid can be '1000092795' or '1000092795.jpg'
    Searches recursively as fallback.
    """
    if fid is None:
        return None
    fid = str(fid).strip()
    if not fid:
        return None

    if fid.lower().endswith(".jpg"):
        cands = [os.path.join(flickr_images_dir, fid)]
    else:
        cands = [os.path.join(flickr_images_dir, fid + ".jpg"), os.path.join(flickr_images_dir, fid)]

    for p in cands:
        if os.path.isfile(p):
            return p

    base = os.path.basename(fid.replace(".jpg", "")) + ".jpg"
    g = glob.glob(os.path.join(flickr_images_dir, "**", base), recursive=True)
    if g:
        return g[0]
    return None


def tensor_image_to_pil(img_t: torch.Tensor) -> Image.Image:
    """
    img_t: [3,H,W] float in [0,1]
    """
    img_np = (img_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# -----------------------------
# Dataset + Dataloader
# -----------------------------
class ESNLIVE_Dataset(Dataset):
    """
    Minimal ESNLI-VE dataset returning hypothesis + image + label.
    """

    def __init__(
        self,
        data_root: str,
        flickr_images_dir: str,
        split: str = "train",
        source: str = "evil",
        image_size: int = 224,
        max_samples: Optional[int] = None,
        seed: int = 0,
        drop_invalid_labels: bool = True,
    ):
        super().__init__()
        self.logger = logging.getLogger("ESNLIVE_Dataset")
        self.data_root = data_root
        self.flickr_images_dir = flickr_images_dir
        self.split = split

        repo_split = "dev" if split in ["val", "valid", "validation"] else split
        self.rows = load_split_rows(cache_root=data_root, split=repo_split, source=source, logger=self.logger)

        keep: List[int] = []
        for i, ex in enumerate(self.rows):
            gold = pick_first(ex, ["gold_label", "Gold_Label", "label"], default=None)
            if gold is None:
                if drop_invalid_labels:
                    continue
                gold = "neutral"
            try:
                _ = label_to_index(gold)
            except Exception:
                if drop_invalid_labels:
                    continue
            keep.append(i)

        if max_samples is not None and max_samples > 0 and len(keep) > max_samples:
            rnd = random.Random(seed)
            rnd.shuffle(keep)
            keep = keep[:max_samples]

        self.keep = keep

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.logger.info(f"split={split} kept {len(self.keep)} / {len(self.rows)}")

    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[self.keep[idx]]

        hypothesis = str(pick_first(ex, ["sentence2", "hypothesis", "Sentence2"], default="")).strip()
        gold = pick_first(ex, ["gold_label", "Gold_Label", "label"], default="neutral")
        label = torch.tensor(label_to_index(gold), dtype=torch.long)

        fid = str(pick_first(ex, ["Flikr30kID", "Flickr30kID", "flickr30k_id", "image_id"], default="")).strip()
        img_path = find_image_path(fid, self.flickr_images_dir)
        if img_path is None:
            raise FileNotFoundError(f"Could not find image for id={fid!r} under {self.flickr_images_dir}")

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            image = self.tf(im)

        return {"id": f"{self.split}_{self.keep[idx]}", "text": hypothesis, "image": image, "label": label}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": [b["id"] for b in batch],
        "text": [b["text"] for b in batch],
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
    }


def make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(0)

    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=g,
    )


# -----------------------------
# Prompt (NO CLS)
# -----------------------------
def build_prompt_no_cls(hypothesis: str, label_options: List[str]) -> str:
    hyp = (hypothesis or "").strip()
    main_text = f"Hypothesis:\n{hyp}".strip()
    instr_text = "Choose one label: " + ", ".join(label_options) + "."
    return "<image>\n" + main_text + "\n\n" + instr_text


# -----------------------------
# Vision embedding extraction
# -----------------------------
def extract_vision_embeds(model: Qwen3VLForConditionalGeneration, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> Any:
    """
    Returns vision features. Shape depends on model internals.
    """
    # This matches your training code path: backbone.model.visual(...)
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual(pixel_values, grid_thw=image_grid_thw)
    if hasattr(model, "visual"):
        return model.visual(pixel_values, grid_thw=image_grid_thw)
    raise AttributeError("No visual module found on model (expected model.model.visual or model.visual).")


# -----------------------------
# Sharding
# -----------------------------
def flush_shard(items: List[Dict[str, Any]], split_out: str, shard_idx: int, manifest_path: str) -> None:
    os.makedirs(split_out, exist_ok=True)
    shard_file = os.path.join(split_out, f"shard_{shard_idx:05d}.pt")
    torch.save(items, shard_file)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"shard": os.path.basename(shard_file), "n": len(items)}) + "\n")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--flickr_images_dir", type=str, default=DEFAULT_FLICKR_DIR)
    ap.add_argument("--split", type=str, default="train", choices=["train", "dev", "test", "validation"])
    ap.add_argument("--source", type=str, default="evil", choices=["evil", "virginie"])

    ap.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_samples", type=int, default=-1)

    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)   # default smaller because we store vision_embeds
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=1000)

    ap.add_argument("--label_options", type=str, default="entailment,neutral,contradiction")

    # vision caching knobs
    ap.add_argument("--cache_vision_embeds", type=int, default=1)  # ON by default
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--fp16", type=int, default=0)
    ap.add_argument("--bf16", type=int, default=0)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("ESNLI_Codebook_NoCLS_VIS")

    label_options = [x.strip() for x in args.label_options.split(",") if x.strip()]

    # Dataset / loader
    ds = ESNLIVE_Dataset(
        data_root=args.data_root,
        flickr_images_dir=args.flickr_images_dir,
        split=args.split,
        source=args.source,
        image_size=args.image_size,
        max_samples=None if args.max_samples < 0 else int(args.max_samples),
    )
    dl = make_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Processor
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    # Model (needed for vision embeddings)
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map={"": args.device} if args.device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    model.eval()

    # Output
    split_out = os.path.join(args.output_dir, args.split)
    os.makedirs(split_out, exist_ok=True)
    manifest_path = os.path.join(split_out, "manifest.jsonl")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    items: List[Dict[str, Any]] = []
    shard_idx = 0

    logger.info(f"Writing cache to: {split_out}")
    logger.info(f"Model: {args.model_name} | dtype={dtype} | device={args.device}")
    logger.info(f"cache_vision_embeds={args.cache_vision_embeds} | batch_size={args.batch_size} | shard_size={args.shard_size}")

    from tqdm import tqdm

    # Compute vision embeddings per *batch*, then store per-sample.
    for batch in tqdm(dl, desc=f"[cache] {args.split}"):
        texts: List[str] = batch["text"]
        images_t: torch.Tensor = batch["image"]  # [B,3,H,W]
        labels: torch.Tensor = batch["label"]    # [B]
        ids: List[str] = batch["id"]

        prompts = [build_prompt_no_cls(t, label_options) for t in texts]
        pil_images = [tensor_image_to_pil(images_t[i]) for i in range(images_t.size(0))]

        enc = processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

        input_ids = enc["input_ids"].detach().cpu()
        attention_mask = enc["attention_mask"].detach().cpu()

        pixel_values = enc.get("pixel_values", None)
        image_grid_thw = enc.get("image_grid_thw", None)

        vision_embeds_cpu = None
        vision_error = None

        if args.cache_vision_embeds:
            if pixel_values is None or image_grid_thw is None:
                vision_error = "Processor did not return pixel_values/image_grid_thw; cannot compute vision embeddings."
            else:
                try:
                    with torch.no_grad():
                        pv = pixel_values.to(model.device, dtype=dtype, non_blocking=True)
                        gthw = image_grid_thw.to(model.device, non_blocking=True)
                        vis = extract_vision_embeds(model, pv, gthw)

                    if torch.is_tensor(vis):
                        vision_embeds_cpu = [vis[i].detach().cpu() for i in range(vis.size(0))]
                    elif isinstance(vis, (list, tuple)):
                        B = pixel_values.size(0)
                        per_sample = []
                        for i in range(B):
                            per_sample.append([v[i].detach().cpu() if torch.is_tensor(v) else v for v in vis])
                        vision_embeds_cpu = per_sample
                    elif isinstance(vis, dict):
                        B = pixel_values.size(0)
                        per_sample = []
                        for i in range(B):
                            per_sample.append({k: (v[i].detach().cpu() if torch.is_tensor(v) else v) for k, v in vis.items()})
                        vision_embeds_cpu = per_sample
                    else:
                        vision_error = f"Unrecognized vision output type: {type(vis)}"
                except Exception as e:
                    vision_error = str(e)

        B = len(ids)
        for i in range(B):
            item: Dict[str, Any] = {
                "id": ids[i],
                "label": labels[i].detach().cpu(),
                "prompt": prompts[i],
                "input_ids": input_ids[i].contiguous(),
                "attention_mask": attention_mask[i].contiguous(),
            }

            if pixel_values is not None:
                item["pixel_values"] = pixel_values[i].detach().cpu()
            if image_grid_thw is not None:
                item["image_grid_thw"] = image_grid_thw[i].detach().cpu()

            if args.cache_vision_embeds:
                if vision_embeds_cpu is not None:
                    item["vision_embeds"] = vision_embeds_cpu[i]
                if vision_error is not None:
                    item["vision_embeds_error"] = vision_error

            items.append(item)

            if len(items) >= args.shard_size:
                flush_shard(items, split_out, shard_idx, manifest_path)
                items = []
                shard_idx += 1

    if items:
        flush_shard(items, split_out, shard_idx, manifest_path)

    meta = {
        "model_name": args.model_name,
        "split": args.split,
        "data_root": args.data_root,
        "flickr_images_dir": args.flickr_images_dir,
        "max_length": args.max_length,
        "image_size": args.image_size,
        "label_options": label_options,
        "cache_vision_embeds": bool(args.cache_vision_embeds),
        "dtype": str(dtype),
        "notes": [
            "Cache does NOT include <CLS>. Append <CLS> at training time.",
            "Uses Qwen3VLForConditionalGeneration (AutoModelForCausalLM won't work for Qwen3-VL in your env).",
            "vision_embeds come from model.model.visual(pixel_values, grid_thw=image_grid_thw).",
        ],
    }
    with open(os.path.join(split_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[done] Wrote cache to: {split_out}")


# -----------------------------
# Training-time helper: append <CLS> later
# -----------------------------
def add_cls_on_the_fly(
    proc: Dict[str, torch.Tensor],
    tokenizer,
    max_length: int,
    cls_token_str: str = "<CLS>",
) -> Dict[str, torch.Tensor]:
    """
    Call this right before model.forward() on a BATCH dict:
      proc = {"input_ids": (B,T), "attention_mask": (B,T), ...}
    """
    cls_id = tokenizer.convert_tokens_to_ids(cls_token_str)
    if cls_id is None or cls_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [cls_token_str]})
        cls_id = tokenizer.convert_tokens_to_ids(cls_token_str)

    input_ids = proc["input_ids"]
    attn = proc["attention_mask"]
    B, _T = input_ids.shape

    cls_col = torch.full((B, 1), int(cls_id), device=input_ids.device, dtype=input_ids.dtype)
    one_col = torch.ones((B, 1), device=attn.device, dtype=attn.dtype)

    input_ids2 = torch.cat([input_ids, cls_col], dim=1)
    attn2 = torch.cat([attn, one_col], dim=1)

    if input_ids2.size(1) > max_length:
        input_ids2 = input_ids2[:, -max_length:]
        attn2 = attn2[:, -max_length:]

    proc2 = dict(proc)
    proc2["input_ids"] = input_ids2
    proc2["attention_mask"] = attn2
    return proc2


if __name__ == "__main__":
    main()
