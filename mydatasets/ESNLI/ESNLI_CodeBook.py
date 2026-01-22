#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESNLI-VE cache builder for Qwen/Qwen3-VL-2B-Instruct

- NO <CLS> in cached tokens (append later)
- YES vision embeddings cached
- YES token-level masks cached:
    item["masks"]["image"] : uint8 [L]
    item["masks"]["text"]  : uint8 [L]

This file is COMPATIBLE with your memmap loader.
"""

import os
import json
import glob
import zipfile
import random
import argparse
import logging
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration


# =============================
# Defaults
# =============================
DEFAULT_DATA_ROOT = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI"
DEFAULT_FLICKR_DIR = DEFAULT_DATA_ROOT
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATA_ROOT, "cache_qwen3_vl_2b_nocls_vis")

LABEL2IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


# =============================
# Utils
# =============================
def label_to_index(x: Any) -> int:
    x = str(x).strip().lower()
    if x not in LABEL2IDX:
        raise ValueError(x)
    return LABEL2IDX[x]


def pick_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def tensor_image_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# =============================
# Token-mask helpers
# =============================
def _get_tokenizer(processor):
    if hasattr(processor, "tokenizer"):
        return processor.tokenizer
    if hasattr(processor, "processor") and hasattr(processor.processor, "tokenizer"):
        return processor.processor.tokenizer
    return None


def _infer_image_token_ids(tokenizer) -> List[int]:
    ids = []
    if tokenizer is None:
        return ids

    for name in [
        "image_token_id", "image_start_token_id", "image_end_token_id",
        "vision_start_token_id", "vision_end_token_id",
        "im_start_id", "im_end_id",
    ]:
        v = getattr(tokenizer, name, None)
        if isinstance(v, int) and v >= 0:
            ids.append(v)

    for s in ["<image>", "<im_start>", "<im_end>", "<vision>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if tid != tokenizer.unk_token_id:
                ids.append(tid)
        except Exception:
            pass

    return sorted(set(ids))


def build_token_masks(input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      processor) -> Dict[str, torch.Tensor]:
    """
    input_ids, attention_mask: [B,T]
    returns bool masks [B,T]
    """
    tok = _get_tokenizer(processor)
    img_ids = _infer_image_token_ids(tok)

    att = attention_mask.bool()
    img = torch.zeros_like(input_ids, dtype=torch.bool)

    if img_ids:
        img_ids_t = torch.tensor(img_ids, device=input_ids.device)
        img = torch.isin(input_ids, img_ids_t)

    img = img & att
    txt = att & (~img)
    return {"image": img, "text": txt}


# =============================
# Dataset
# =============================
class ESNLIVE_Dataset(Dataset):
    def __init__(self, root, flickr_dir, split):
        self.root = root
        self.flickr = flickr_dir
        self.split = split

        csv = glob.glob(os.path.join(root, "**", f"*{split}*.csv"), recursive=True)[0]
        df = pd.read_csv(csv)
        self.rows = df.to_dict("records")

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        hyp = pick_first(r, ["sentence2", "hypothesis"])
        label = torch.tensor(label_to_index(pick_first(r, ["gold_label"])))
        fid = str(pick_first(r, ["Flickr30kID", "image_id"]))

        img_path = glob.glob(os.path.join(self.flickr, "**", fid + ".jpg"), recursive=True)[0]
        img = self.tf(Image.open(img_path).convert("RGB"))

        return {
            "id": f"{self.split}_{idx}",
            "text": hyp,
            "image": img,
            "label": label,
        }


def collate(b):
    return {
        "id": [x["id"] for x in b],
        "text": [x["text"] for x in b],
        "image": torch.stack([x["image"] for x in b]),
        "label": torch.stack([x["label"] for x in b]),
    }


# =============================
# Vision embeddings
# =============================
def extract_vision_embeds(model, pixel_values, grid_thw):
    if hasattr(model.model, "visual"):
        return model.model.visual(pixel_values, grid_thw=grid_thw)
    raise RuntimeError("No visual backbone")


# =============================
# Sharding
# =============================
def flush_shard(items, out_dir, shard_idx, manifest):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shard_{shard_idx:05d}.pt")
    torch.save(items, path)
    with open(manifest, "a") as f:
        f.write(json.dumps({"shard": os.path.basename(path), "num_items": len(items)}) + "\n")


# =============================
# Main
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    ds = ESNLIVE_Dataset(DEFAULT_DATA_ROOT, DEFAULT_FLICKR_DIR, args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)

    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL_NAME,
        device_map={"": args.device},
        trust_remote_code=True,
    ).eval()

    out_split = os.path.join(DEFAULT_OUTPUT_DIR, args.split)
    os.makedirs(out_split, exist_ok=True)
    manifest = os.path.join(out_split, "manifest.jsonl")
    if os.path.exists(manifest):
        os.remove(manifest)

    items, shard_idx = [], 0

    for batch in dl:
        enc = processor(
            text=[f"<image>\nHypothesis:\n{t}" for t in batch["text"]],
            images=[tensor_image_to_pil(i) for i in batch["image"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = enc["input_ids"].cpu()
        attn = enc["attention_mask"].cpu()
        masks = build_token_masks(input_ids, attn, processor)

        pv = enc["pixel_values"].to(args.device)
        gthw = enc["image_grid_thw"].to(args.device)
        with torch.no_grad():
            vis = extract_vision_embeds(model, pv, gthw)
        vis = vis.cpu()

        for i in range(len(batch["id"])):
            L = int(attn[i].sum())
            item = {
                "id": batch["id"][i],
                "label": batch["label"][i],
                "input_ids": input_ids[i, :L],
                "attention_mask": attn[i, :L],
                "masks": {
                    "image": masks["image"][i, :L].to(torch.uint8),
                    "text": masks["text"][i, :L].to(torch.uint8),
                },
                "vision_embeds": vis[i],
            }
            items.append(item)

            if len(items) >= args.shard_size:
                flush_shard(items, out_split, shard_idx, manifest)
                items, shard_idx = [], shard_idx + 1

    if items:
        flush_shard(items, out_split, shard_idx, manifest)


if __name__ == "__main__":
    main()
