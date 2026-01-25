# esnli_cache_qwen3vl_b1_proof_dump_and_verify.py
# Batch-size=1 ONLY. "Proofed" against padding/collation mistakes by:
#   (1) Saving EXACT tensors you will pass to language_model (no extra trimming beyond attention keep)
#   (2) Immediately re-loading the just-written shard and printing shapes + stats
#
# Saves per item:
#   input_ids            : (1, L) int64
#   attention_mask       : (1, L) int64
#   position_ids         : (3, 1, L) int64
#   input_embeds         : (1, L, 2048) float16
#   visual_pos_masks     : (1, L) bool
#   deepstack_visual_embeds : (K, 64, 2048) float32 (K levels; e.g. 3)
# plus id/label/prompt
#
# IMPORTANT: this script enforces batch_size=1 and will error otherwise.

import os
import json
import glob
import zipfile
import random
import argparse
import logging
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


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
# Dataset + Dataloader
# -----------------------------
class ESNLIVE_Dataset(Dataset):
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
        return 10

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
    # NOTE: this collate can support B>1 but we enforce B=1 later.
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
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
        generator=g,
    )


# -----------------------------
# Prompt
# -----------------------------
def build_prompt_cls(hypothesis: Sequence[str]) -> List[str]:
    instr_text = """
You are given an image and a hypothesis about the image.
Decide whether the hypothesis is supported by the image.

Choose EXACTLY ONE label: entailment, neutral, or contradiction.

Answer format:
Label: one word only (entailment / neutral / contradiction)
Explanation: free text

<CLS>
"""
    return [f"Hypothesis:\n{str(h).strip()}\n\n{instr_text}" for h in hypothesis]


# -----------------------------
# Shard write + immediate verify
# -----------------------------
def flush_shard(items: List[Dict[str, Any]], split_out: str, shard_idx: int, manifest_path: str) -> str:
    os.makedirs(split_out, exist_ok=True)
    shard_file = os.path.join(split_out, f"shard_{shard_idx:05d}.pt")
    torch.save(items, shard_file)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"shard": os.path.basename(shard_file), "n": len(items)}) + "\n")
    return shard_file


def _tensor_stats_line(name: str, t: Any) -> str:
    if not torch.is_tensor(t):
        return f"{name:22s}: {type(t)}"
    if t.numel() == 0:
        return f"{name:22s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} numel=0"
    x = t.detach()
    # compute in float for stable stats
    xf = x.float() if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else x.float()
    mn = xf.min().item()
    mx = xf.max().item()
    mean = xf.mean().item()
    std = xf.std(unbiased=False).item()
    nonzero = (x != 0).float().mean().item() * 100.0
    return (f"{name:22s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} nonzero={nonzero:.2f}%")


def verify_shard(shard_path: str, n_show: int = 1) -> None:
    print(f"\n[verify] loading shard: {shard_path}")
    items = torch.load(shard_path, map_location="cpu")
    if not isinstance(items, (list, tuple)):
        raise TypeError(f"Shard is not list/tuple: {type(items)}")
    print(f"[verify] items: {len(items)}")

    n_show = min(int(n_show), len(items))
    for i in range(n_show):
        ex = items[i]
        print(f"\n[verify] example {i} id={ex.get('id', None)}")
        keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "input_embeds",
            "visual_pos_masks",
            "deepstack_visual_embeds",
        ]
        for k in keys:
            print(_tensor_stats_line(k, ex.get(k, None)))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", required=True)
    ap.add_argument("--flickr_images_dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--split", type=str, default="validation", choices=["train", "dev", "test", "validation"])
    ap.add_argument("--source", type=str, default="evil", choices=["evil", "virginie"])

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_samples", type=int, default=-1)

    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--shard_size", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])

    ap.add_argument("--verify_every_flush", type=int, default=1)   # verify right after each flush
    ap.add_argument("--verify_n_show", type=int, default=1)        # print first N examples per shard

    args = ap.parse_args()

    if int(args.batch_size) != 1:
        raise ValueError("This debug script is batch_size=1 ONLY. Set --batch_size 1.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("ESNLI_Cache_Qwen3VL_B1_VERIFY")

    ds = ESNLIVE_Dataset(
        data_root=args.data_root,
        flickr_images_dir=args.flickr_images_dir,
        split=args.split,
        source=args.source,
        image_size=args.image_size,
        max_samples=None if args.max_samples < 0 else int(args.max_samples),
    )
    dl = make_loader(ds, batch_size=1, num_workers=int(args.num_workers), shuffle=False)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    if args.dtype == "float16":
        model_dtype = torch.float16
    elif args.dtype == "bfloat16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
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
    logger.info(f"Model: {args.model_name} | dtype={model_dtype} | device={args.device}")
    logger.info(f"batch_size=1 | shard_size={args.shard_size} | max_length={args.max_length}")
    logger.info("This script saves exactly what you later feed to language_model (after attention keep).")

    from tqdm import tqdm

    for batch in tqdm(dl, desc=f"[cache] {args.split}"):
        # enforce B=1
        hyp: List[str] = batch["text"]
        images_t: torch.Tensor = batch["image"]  # [1,3,H,W]
        labels: torch.Tensor = batch["label"]    # [1]
        ids: List[str] = batch["id"]
        if images_t.shape[0] != 1:
            raise RuntimeError(f"Expected batch size 1 but got {images_t.shape[0]}")
        if len(hyp) != 1 or len(ids) != 1:
            raise RuntimeError("Expected single element lists for text/id")

        text = build_prompt_cls(hypothesis=hyp)[0]

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        pil_image = tensor_image_to_pil(images_t[0])

        enc = processor(
            text=[prompt],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
        )

        input_ids = enc["input_ids"]            # (1,T)
        attention_mask = enc["attention_mask"]  # (1,T)
        if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
            raise RuntimeError("processor did not return tensor input_ids/attention_mask")

        pixel_values = enc.get("pixel_values", None)
        image_grid_thw = enc.get("image_grid_thw", None)
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError("Processor did not return pixel_values / image_grid_thw; check processor/model version.")

        # move ids/mask to CPU for trimming, but embeddings computed on model.device
        input_ids_cpu = input_ids.detach().cpu()
        attention_cpu = attention_mask.detach().cpu()

        # build token embeddings then scatter image features
        token_embeds = model.model.get_input_embeddings()(input_ids.to(model.device))  # (1,T,2048)

        with torch.no_grad():
            pv = pixel_values.to(model.device, dtype=pixel_values.dtype, non_blocking=True)
            gthw = image_grid_thw.to(model.device, non_blocking=True)

            image_embeds_list, deep_stack_viz_list = model.get_image_features(pv, gthw)

            image_embeds_cat = torch.cat(image_embeds_list, dim=0).to(token_embeds.device, token_embeds.dtype)

            placeholder_mask, _ = model.model.get_placeholder_mask(
                input_ids.to(model.device),
                inputs_embeds=token_embeds,
                image_features=image_embeds_cat,
            )
            placeholder_mask_2d = placeholder_mask[..., 0] if placeholder_mask.dim() == 3 else placeholder_mask  # (1,T)

            # scatter
            token_embeds = token_embeds.masked_scatter(placeholder_mask, image_embeds_cat)

            # force fp16 for caching (your requirement)
            token_embeds = token_embeds.to(torch.float16)

            # rope indices
            position_ids, _ = model.model.get_rope_index(
                input_ids.to(model.device),
                gthw,
                None,
                attention_mask=attention_mask.to(model.device),
            )
            position_ids_cpu = position_ids.detach().cpu()  # typically (1,3,1,T) or (3,1,T) depending impl

        # normalize position_ids to (3,1,T) for this single sample
        # Common case from your working print: (3,1,T)
        # Some impls return leading batch: (1,3,1,T)
        pos = position_ids_cpu
        if pos.dim() == 4 and pos.shape[0] == 1:
            pos = pos[0]  # -> (3,1,T)
        if not (pos.dim() == 3 and int(pos.shape[0]) == 3 and int(pos.shape[1]) == 1):
            raise RuntimeError(f"Unexpected position_ids shape after normalize: {tuple(pos.shape)}")

        # deepstack: keep all levels -> stack (K,64,2048)
        deep_levels = []
        if isinstance(deep_stack_viz_list, (list, tuple)):
            for t in deep_stack_viz_list:
                if torch.is_tensor(t):
                    deep_levels.append(t.detach().cpu())
        if len(deep_levels) > 0:
            deepstack_visual_embeds = torch.cat([t.unsqueeze(0) for t in deep_levels], dim=0)  # (K,64,2048)
        else:
            deepstack_visual_embeds = torch.empty((0, 0, 0), dtype=torch.float32)

        # attention keep (padding-proof): keep only tokens with attention==1
        keep = attention_cpu[0].bool()  # (T,)
        if keep.sum().item() <= 0:
            raise RuntimeError("attention_mask had no True tokens?")

        input_ids_keep = input_ids_cpu[0][keep].contiguous().unsqueeze(0)        # (1,L)
        attention_keep = attention_cpu[0][keep].contiguous().unsqueeze(0)        # (1,L)
        input_embeds_keep = token_embeds.detach().cpu()[0][keep].contiguous().unsqueeze(0)  # (1,L,2048) fp16

        # visual token positions in final embeds
        visual_pos_masks = placeholder_mask_2d.detach().cpu()[0][keep].contiguous().unsqueeze(0).bool()  # (1,L)

        # position ids (3,1,T) -> (3,1,L)
        pos_keep = pos[:, :, keep].contiguous()

        # build item (THIS IS EXACTLY WHAT YOU FEED LATER)
        item = {
            "id": ids[0],
            "label": labels[0].detach().cpu(),
            "prompt": prompt,

            "input_ids": input_ids_keep,                 # (1,L)
            "attention_mask": attention_keep,            # (1,L)
            "position_ids": pos_keep,                    # (3,1,L)
            "input_embeds": input_embeds_keep,           # (1,L,2048) fp16
            "visual_pos_masks": visual_pos_masks,        # (1,L) bool
            "deepstack_visual_embeds": deepstack_visual_embeds,  # (K,64,2048)
        }

        items.append(item)

        # flush
        if len(items) >= int(args.shard_size):
            shard_path = flush_shard(items, split_out, shard_idx, manifest_path)
            if int(args.verify_every_flush) == 1:
                verify_shard(shard_path, n_show=int(args.verify_n_show))
            items = []
            shard_idx += 1

    # flush final
    if items:
        shard_path = flush_shard(items, split_out, shard_idx, manifest_path)
        if int(args.verify_every_flush) == 1:
            verify_shard(shard_path, n_show=int(args.verify_n_show))

    # meta
    meta = {
        "model_name": args.model_name,
        "split": args.split,
        "max_length": int(args.max_length),
        "image_size": int(args.image_size),
        "dtype": str(model_dtype),
        "notes": [
            "BATCH SIZE 1 ONLY debug writer.",
            "Saved tensors are post-attention_keep (no padding).",
            "input_embeds are after placeholder scatter and cast to float16.",
            "position_ids normalized to (3,1,T) then trimmed to (3,1,L).",
            "deepstack_visual_embeds stacks all deep levels to (K,64,2048).",
            "After each shard write, script reloads shard and prints shapes/stats.",
        ],
    }
    with open(os.path.join(split_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[done] wrote cache to: {split_out}")


if __name__ == "__main__":
    main()
