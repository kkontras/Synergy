#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESNLI-VE cache builder for Qwen/Qwen3-VL-2B-Instruct
- NO <CLS> in cached tokens (append later)
- YES vision embeddings cached
- YES token-level masks cached:
    item["masks"]["image"] : uint8 [L]
    item["masks"]["text"]  : uint8 [L]   (attention & ~image)

This version DOES NOT assume local CSV names.
It downloads/uses the ESNLI-VE repo zip and auto-finds train/dev/test files.

Usage:
python esnli_cache_qwen3vl_2b_nocls_vis_WITH_MASKS.py \
  --data_root /scratch/kkontras/ESNLI \
  --flickr_images_dir /scratch/kkontras/ESNLI/flickr30k-images \
  --model_name Qwen/Qwen3-VL-2B-Instruct \
  --output_dir /scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis \
  --split validation \
  --source evil
"""

import os
import json
import glob
import zipfile
import random
import argparse
import logging
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import einops
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration


# -----------------------------
# Labels
# -----------------------------
LABEL2IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


# -----------------------------
# Download / repo handling (ROBUST)
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
    Best-effort search for train/dev/test files in extracted repo.
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
    img_np = (img_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# -----------------------------
# Token-mask helpers
# -----------------------------
def _get_tokenizer_from_processor(processor):
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        return processor.tokenizer
    if hasattr(processor, "processor") and hasattr(processor.processor, "tokenizer"):
        return processor.processor.tokenizer
    return None


def _infer_image_token_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    if tokenizer is None:
        return ids

    attr_names = [
        "image_token_id",
        "image_start_token_id",
        "image_end_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "im_start_id",
        "im_end_id",
    ]
    for name in attr_names:
        v = getattr(tokenizer, name, None)
        if isinstance(v, int) and v >= 0:
            ids.append(int(v))

    cand_strs = ["<image>", "<img>", "<vision>", "<im_start>", "<im_end>", "<vision_start>", "<vision_end>"]
    for s in cand_strs:
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -999):
                ids.append(int(tid))
        except Exception:
            pass

    return sorted(set(ids))


def build_image_text_token_masks(enc_cpu: Dict[str, torch.Tensor], processor) -> Dict[str, torch.Tensor]:
    """
    Returns bool masks (CPU):
      masks["image"] : [B,T]
      masks["text"]  : [B,T]  (attention & ~image)

    If processor provides an image mask, we use it.
    Otherwise infer from tokenizer image token ids.
    """
    input_ids = enc_cpu["input_ids"]
    attention_mask = enc_cpu.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    candidate_keys = ["image_mask", "image_token_mask", "vision_token_mask", "media_token_mask"]
    for k in candidate_keys:
        m = enc_cpu.get(k, None)
        if torch.is_tensor(m) and m.shape == input_ids.shape:
            img_mask = m.to(torch.bool)
            att_bool = attention_mask.to(torch.bool)
            img_mask = img_mask & att_bool
            txt_mask = att_bool & (~img_mask)
            return {"image": img_mask, "text": txt_mask}

    tok = _get_tokenizer_from_processor(processor)
    img_token_ids = _infer_image_token_ids(tok)
    if len(img_token_ids) > 0:
        img_ids = torch.tensor(img_token_ids, dtype=input_ids.dtype, device=input_ids.device)
        img_mask = torch.isin(input_ids, img_ids)
        att_bool = attention_mask.to(torch.bool)
        img_mask = img_mask & att_bool
        txt_mask = att_bool & (~img_mask)
        return {"image": img_mask, "text": txt_mask}

    att_bool = attention_mask.to(torch.bool)
    img_mask = torch.zeros_like(input_ids, dtype=torch.bool) & att_bool
    txt_mask = att_bool & (~img_mask)
    return {"image": img_mask, "text": txt_mask}


# -----------------------------
# Dataset + Dataloader
# -----------------------------
class ESNLIVE_Dataset(Dataset):
    """
    ESNLI-VE dataset returning hypothesis + image + label.
    Loads split rows from repo zip, not by local CSV naming.
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
        if repo_split not in ["train", "dev", "test"]:
            raise ValueError(f"split must be train/dev/test/validation, got {split}")

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
        self.tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
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
def build_prompt_no_cls(
        hypothesis: Sequence[str],
        label_options: List[str],
) -> List[str]:
    instr_text = """\
    You are given an image and a hypothesis about the image.
    Decide whether the hypothesis is supported by the image.

    Choose EXACTLY ONE label: entailment, neutral, or contradiction.

    Definitions:
    - entailment:
      The hypothesis is clearly true given what is visible in the image.

    - contradiction:
      The hypothesis is clearly false given the image.
      This includes cases where the hypothesis describes an action, state, or situation
      that is incompatible with what is visible in the image.

    - neutral:
      The image does not provide enough information to decide.
      The hypothesis could be true or false, and nothing visible contradicts it.

    Important rules:
    - "The image does not show the hypothesis" is NOT enough to choose neutral.
    - If the hypothesis claims something that is NOT happening in the image
      (e.g., walking vs sitting, outdoors vs clearly indoors), choose contradiction.
    - Use neutral ONLY when the image neither supports NOR contradicts the hypothesis.

    Answer format:
    Label: one word only (entailment / neutral / contradiction)
    Explanation: free text

    <CLS>
    """

    return [
        f"Hypothesis:\n{str(h).strip()}\n\n{instr_text}"
        for h in hypothesis
    ]



# -----------------------------
# Vision embedding extraction
# -----------------------------
def extract_vision_embeds(model, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Returns a Tensor of vision token embeddings (B, N, D).
    Handles common Qwen3-VL variants where visual() returns:
      - Tensor
      - Tuple(Tensor, ...)
      - Dict with known keys
    """
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        out = model.model.visual(pixel_values, grid_thw=image_grid_thw)
    elif hasattr(model, "visual"):
        out = model.visual(pixel_values, grid_thw=image_grid_thw)
    else:
        raise AttributeError("No visual module found on model (expected model.model.visual or model.visual).")

    # Case 1: direct tensor
    if torch.is_tensor(out):
        return out

    # Case 2: tuple/list -> pick first tensor element
    if isinstance(out, (tuple, list)):
        for z in out:
            if torch.is_tensor(z):
                return z
        raise TypeError(f"visual() returned tuple/list but no tensor found. Types: {[type(z) for z in out]}")

    # Case 3: dict -> try common keys
    if isinstance(out, dict):
        for k in ["last_hidden_state", "hidden_states", "vision_embeds", "embeds", "features", "x"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        # fallback: first tensor value
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise TypeError(f"visual() returned dict but no tensor values found. Keys: {list(out.keys())}")

    raise TypeError(f"Unrecognized visual() output type: {type(out)}")

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

    ap.add_argument("--data_root", required=True)
    ap.add_argument("--flickr_images_dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--split", type=str, default="train", choices=["train", "dev", "test", "validation"])
    ap.add_argument("--source", type=str, default="evil", choices=["evil", "virginie"])

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_samples", type=int, default=-1)

    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=1000)

    ap.add_argument("--label_options", type=str, default="entailment,neutral,contradiction")

    ap.add_argument("--cache_vision_embeds", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--fp16", type=int, default=0)
    ap.add_argument("--bf16", type=int, default=0)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("ESNLI_Codebook_NoCLS_VIS_MASKS")

    label_options = [x.strip() for x in args.label_options.split(",") if x.strip()]

    ds = ESNLIVE_Dataset(
        data_root=args.data_root,
        flickr_images_dir=args.flickr_images_dir,
        split=args.split,
        source=args.source,
        image_size=args.image_size,
        max_samples=None if args.max_samples < 0 else int(args.max_samples),
    )
    dl = make_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

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

    for batch in tqdm(dl, desc=f"[cache] {args.split}"):
        texts: List[str] = batch["text"]
        images_t: torch.Tensor = batch["image"]  # [B,3,H,W]
        labels: torch.Tensor = batch["label"]    # [B]
        ids: List[str] = batch["id"]
        batch_size = images_t.shape[0]
        label_options = "entailment, neutral, contradiction"

        texts = build_prompt_no_cls(hypothesis=texts, label_options=label_options)

        messages_batch = [
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": t},
            ]}]
            for t in texts
        ]
        prompts = [
            processor.apply_chat_template( m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_batch
        ]

        pil_images = [tensor_image_to_pil(images_t[i]) for i in range(images_t.size(0))]

        enc = processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_ids_batch = enc["input_ids"].detach().cpu()
        attention_batch = enc["attention_mask"].detach().cpu()

        # Build token masks on CPU
        enc_cpu_for_masks: Dict[str, torch.Tensor] = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_batch,
        }
        for kk in ["image_mask", "image_token_mask", "vision_token_mask", "media_token_mask"]:
            if kk in enc and torch.is_tensor(enc[kk]):
                enc_cpu_for_masks[kk] = enc[kk].detach().cpu()

        masks_batch = build_image_text_token_masks(enc_cpu_for_masks, processor)
        image_mask_batch = masks_batch["image"]  # bool [B,T]
        text_mask_batch = masks_batch["text"]    # bool [B,T]
        pixel_values = enc.get("pixel_values", None)
        image_grid_thw = enc.get("image_grid_thw", None)

        try:
            with torch.no_grad():
                pv = pixel_values.to(model.device, dtype=dtype, non_blocking=True)
                gthw = image_grid_thw.to(model.device, non_blocking=True)
                vis = extract_vision_embeds(model, pv, gthw)
                vis = einops.rearrange(vis, "(b i) c -> b i c", b=batch_size)

            vision_embeds_cpu = [vis[i].detach().cpu() for i in range(vis.size(0))]
        except Exception as e:
            raise Exception(e)

        B = len(ids)
        for i in range(B):
            # trim to real length (assumes attention_mask marks real tokens)
            L = int(attention_batch[i].sum().item())
            if L <= 0:
                L = 1

            input_ids_i = input_ids_batch[i, :L].contiguous()
            attention_i = attention_batch[i, :L].contiguous()

            image_mask_i = image_mask_batch[i, :L].to(torch.uint8).contiguous()
            text_mask_i = text_mask_batch[i, :L].to(torch.uint8).contiguous()

            item: Dict[str, Any] = {
                "id": ids[i],
                "label": labels[i].detach().cpu(),
                "prompt": prompts[i],
                "input_ids": input_ids_i,
                "attention_mask": attention_i,
                "masks": {
                    "image": image_mask_i,
                    "text": text_mask_i,
                },
            }

            item["pixel_values"] = pixel_values[i].detach().cpu()
            item["image_grid_thw"] = image_grid_thw[i].detach().cpu()
            item["vision_embeds"] = vision_embeds_cpu[i]

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
            "vision_embeds come from model.model.visual(pixel_values, grid_thw=image_grid_thw).",
            "Token-level masks saved in item['masks']: {'image':..., 'text':...} aligned with input_ids.",
            "Split mapping: validation -> dev",
        ],
    }
    with open(os.path.join(split_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[done] Wrote cache to: {split_out}")

import math, textwrap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def save_image_prompt_label_grid(
    pil_images,
    prompts,
    labels,
    ids=None,
    label_options=None,
    outpath="sanity_batch.png",
    max_items=16,
    ncols=4,
    wrap=60,
    dpi=200,
    title="Sanity check: image + prompt + label",
):
    """
    Save a grid showing PIL images with (id, label, optional label name) and prompt text.
    """
    B = min(len(pil_images), len(prompts), len(labels), max_items)
    nrows = math.ceil(B / ncols)

    fig = plt.figure(figsize=(ncols * 4.2, nrows * 4.8), dpi=dpi)
    fig.suptitle(title, fontsize=14)

    for i in range(B):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(pil_images[i])
        ax.axis("off")

        # label -> int
        y = labels[i]
        try:
            y = int(y)  # works for torch scalar / numpy scalar / python int
        except Exception:
            pass

        y_name = None
        if label_options is not None:
            try:
                y_name = label_options[y]
            except Exception:
                y_name = None

        header = []
        if ids is not None and i < len(ids):
            header.append(f"id: {ids[i]}")
        header.append(f"y: {y}" + (f" ({y_name})" if y_name is not None else ""))

        prompt_wrapped = "\n".join(textwrap.wrap(prompts[i], width=wrap))
        ax.set_title("\n".join(header) + "\n" + prompt_wrapped, fontsize=8, loc="left")

    # hide unused slots
    for j in range(B + 1, nrows * ncols + 1):
        ax = fig.add_subplot(nrows, ncols, j)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    main()
