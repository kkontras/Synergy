"""
LongVALE SFT Dataset + Dataloader for the Synergy framework.

Loads pre-extracted visual (100, 768) and audio (52, 768) features from .npy
files alongside the longvale-sft-it-25k.json instruction-tuning annotations.
Flattens multi-turn conversations into individual (question, answer) pairs and
tokenizes them at dataset init time.

Expected on-disk layout (after zip extraction):
  {data_root}/features_training/video_features_7240/{video_id}.npy
  {data_root}/features_training/audio_features_7240/{video_id}.npy

Config keys read:
  config.dataset.data_root         — LongVALE root dir
  config.dataset.ann_file          — annotation filename (default: longvale-sft-it-25k.json)
  config.dataset.val_ratio         — fraction of video IDs held out for val (default: 0.1)
  config.model.args.model_name     — HF model name (for tokenizer)
  config.model.args.max_text_len   — max tokens per (Q+A) sequence (default: 512)
  config.model.save_base_dir       — used as HF cache_dir fallback
  config.training_params.batch_size
  config.training_params.test_batch_size
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_temporal_tokens(text: str, token_map: Dict[str, float]) -> str:
    """Replace <sN> / <eN> placeholders with their timestamp strings."""
    for tok, sec in token_map.items():
        text = text.replace(tok, f"{sec:.1f}s")
    return text


def _flatten_conversation(sample: dict) -> List[Dict]:
    """Flatten a multi-turn conversation entry into (question, answer) pairs."""
    video_id = sample["id"]
    token_map = sample.get("meta", {}).get("token", {})
    convs = sample["conversations"]
    pairs: List[Dict] = []
    for i in range(0, len(convs) - 1, 2):
        h = convs[i]
        g = convs[i + 1]
        if h.get("from") != "human" or g.get("from") != "gpt":
            continue
        q = h["value"].replace("<video>\n", "").replace("<video>", "").strip()
        a = g["value"].strip()
        q = _replace_temporal_tokens(q, token_map)
        a = _replace_temporal_tokens(a, token_map)
        pairs.append({"video_id": video_id, "question": q, "answer": a})
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LongVALE_SFT_Dataset(Dataset):
    """
    PyTorch Dataset for LongVALE SFT instruction-tuning.

    Returns per sample:
      video_feats   — (100, 768) float32
      audio_feats   — (52, 768) float32
      input_ids     — (T,) long   — Q + A tokens
      label_ids     — (T,) long   — -100 for Q tokens, A token ids for answer
      label         — scalar 0    — dummy (Synergy framework compatibility)
    """

    VIS_SUBDIR = "video_features_7240"
    AUD_SUBDIR = "audio_features_7240"
    N_VIS = 100
    N_AUD = 52
    VIS_DIM = 768
    AUD_DIM = 768

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer,
        ann_file: str = "longvale-sft-it-25k.json",
        max_text_len: int = 512,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.tokenizer = tokenizer
        self.max_text_len = int(max_text_len)

        # ---- load & flatten annotations ----
        ann_path = os.path.join(data_root, ann_file)
        with open(ann_path, "r", encoding="utf-8") as f:
            raw: List[dict] = json.load(f)

        all_pairs: List[dict] = []
        for sample in raw:
            all_pairs.extend(_flatten_conversation(sample))

        # ---- train / val split by video ID ----
        all_ids = sorted(set(p["video_id"] for p in all_pairs))
        n_val = max(1, int(val_ratio * len(all_ids)))
        val_ids = set(all_ids[-n_val:])

        if split == "train":
            self.pairs = [p for p in all_pairs if p["video_id"] not in val_ids]
        else:
            # "val", "validation", "test" all use the held-out set
            self.pairs = [p for p in all_pairs if p["video_id"] in val_ids]

        # Feature directories (training zip contents)
        feat_base = os.path.join(data_root, "features_training")
        self.vis_dir = os.path.join(feat_base, self.VIS_SUBDIR)
        self.aud_dir = os.path.join(feat_base, self.AUD_SUBDIR)

        # ---- pre-tokenise everything once ----
        pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
        self.tokenized: List[Dict] = [self._tokenize(p) for p in self.pairs]

        print(
            f"[LongVALE_SFT_Dataset] split={split}  N={len(self.pairs)}"
            f"  vis_dir={self.vis_dir}"
        )

    # ------------------------------------------------------------------
    def _tokenize(self, pair: dict) -> dict:
        tok = self.tokenizer
        half = self.max_text_len // 2

        q_text = f"User: {pair['question']}\nAssistant:"
        a_text = f" {pair['answer']}"

        q_ids = tok.encode(
            q_text,
            add_special_tokens=True,
            truncation=True,
            max_length=half,
        )
        a_ids = tok.encode(
            a_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_len - len(q_ids),
        )
        # Ensure EOS at end of answer
        eos = tok.eos_token_id
        if eos is not None and (not a_ids or a_ids[-1] != eos):
            a_ids = a_ids + [eos]

        input_ids = q_ids + a_ids
        label_ids = [-100] * len(q_ids) + a_ids

        return {
            "video_id": pair["video_id"],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int) -> dict:
        tok_item = self.tokenized[idx]
        video_id = tok_item["video_id"]

        vis_path = os.path.join(self.vis_dir, f"{video_id}.npy")
        aud_path = os.path.join(self.aud_dir, f"{video_id}.npy")

        try:
            vis_feat = torch.from_numpy(
                np.load(vis_path).astype(np.float32)
            )  # (100, 768)
        except Exception:
            vis_feat = torch.zeros(self.N_VIS, self.VIS_DIM, dtype=torch.float32)

        try:
            aud_feat = torch.from_numpy(
                np.load(aud_path).astype(np.float32)
            )  # (52, 768)
        except Exception:
            aud_feat = torch.zeros(self.N_AUD, self.AUD_DIM, dtype=torch.float32)

        return {
            "video_feats": vis_feat,
            "audio_feats": aud_feat,
            "input_ids": tok_item["input_ids"],
            "label_ids": tok_item["label_ids"],
            "label": torch.tensor(0, dtype=torch.long),  # dummy
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def longvale_sft_collate(
    batch: List[dict],
    pad_token_id: int = 0,
) -> dict:
    """Right-pad text sequences; stack fixed-size feature tensors."""
    B = len(batch)
    max_len = max(int(b["input_ids"].shape[0]) for b in batch)

    video_feats = torch.stack([b["video_feats"] for b in batch])  # (B,100,768)
    audio_feats = torch.stack([b["audio_feats"] for b in batch])  # (B, 52,768)
    labels_cls = torch.stack([b["label"] for b in batch])          # (B,) dummy

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    label_ids = torch.full((B, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        L = int(b["input_ids"].shape[0])
        input_ids[i, :L] = b["input_ids"]
        label_ids[i, :L] = b["label_ids"]
        attention_mask[i, :L] = 1

    return {
        "label": labels_cls,
        "data": {
            "video_feats": video_feats,
            "audio_feats": audio_feats,
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": attention_mask,
        },
    }


# ---------------------------------------------------------------------------
# Dataloader wrapper (Synergy framework interface)
# ---------------------------------------------------------------------------

class LongVALE_SFT_Dataloader:
    """
    Drop-in dataloader class for the Synergy framework.
    Provides .train_loader, .valid_loader, .test_loader.
    """

    def __init__(self, config, **kwargs):
        data_root: str = config.dataset.data_root
        ann_file: str = getattr(config.dataset, "ann_file", "longvale-sft-it-25k.json")
        val_ratio: float = float(getattr(config.dataset, "val_ratio", 0.1))
        max_text_len: int = int(getattr(getattr(config, "model", None), "args", None) and
                                getattr(config.model.args, "max_text_len", 512) or 512)

        model_name: str = getattr(
            getattr(getattr(config, "model", None), "args", None),
            "model_name",
            "Qwen/Qwen2.5-Omni-3B",
        )
        hf_cache: Optional[str] = getattr(
            getattr(config, "model", None), "save_base_dir", None
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_size = int(config.training_params.batch_size)
        test_batch_size = int(config.training_params.test_batch_size)
        num_workers = int(getattr(config.training_params, "data_loader_workers", 4))

        train_ds = LongVALE_SFT_Dataset(
            data_root, "train", tokenizer,
            ann_file=ann_file, max_text_len=max_text_len, val_ratio=val_ratio,
        )
        val_ds = LongVALE_SFT_Dataset(
            data_root, "val", tokenizer,
            ann_file=ann_file, max_text_len=max_text_len, val_ratio=val_ratio,
        )
        test_ds = LongVALE_SFT_Dataset(
            data_root, "test", tokenizer,
            ann_file=ann_file, max_text_len=max_text_len, val_ratio=val_ratio,
        )

        pad_id = int(tokenizer.pad_token_id)
        collate = lambda b: longvale_sft_collate(b, pad_token_id=pad_id)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False, collate_fn=collate,
        )
        self.valid_loader = DataLoader(
            val_ds, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False, collate_fn=collate,
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False, collate_fn=collate,
        )
