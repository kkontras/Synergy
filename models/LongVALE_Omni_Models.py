"""
LongVALE Omni Model Classes for the Synergy framework.

Architecture
------------
  - Backbone: Qwen2.5-Omni-3B (thinker only; talker discarded after loading).
  - Modality projectors:
      video_proj  : Linear(768 → d_model)   projects (B, 100, 768) video features
      audio_proj  : Linear(768 → d_model)   projects (B,  52, 768) audio features
  - Forward:
      1. Project features → prefix token embeddings
      2. Embed text tokens via thinker.model.embed_tokens
      3. Concatenate [video_emb | audio_emb | text_emb]  → inputs_embeds
      4. Build matching attention_mask and labels (−100 for prefix + question)
      5. thinker(inputs_embeds=…, attention_mask=…, labels=…) → lm_loss
  - Return: {"preds": {}, "losses": {"lm_loss": lm_loss}, "features": {}}

Variants
--------
  LongVALE_Omni_Combined   — video + audio
  LongVALE_Omni_VideoOnly  — video only (audio zeroed)
  LongVALE_Omni_AudioOnly  — audio only (video zeroed)
  LongVALE_Omni_RMask      — random modality masking during training

Config keys consumed from args
-------------------------------
  model_name        (str)  HF checkpoint name, default "Qwen/Qwen2.5-Omni-3B"
  hf_cache          (str)  HF cache_dir
  save_base_dir     (str)  fallback HF cache_dir
  lora_config       (dict) optional LoRA settings (use_lora, lora_r, …)
  clip_grad         (bool) gradient clipping flag (handled externally)
  rmask_p           (float) per-sample masking probability for RMask (default 0.15)
  multi_loss        (dict) must contain {"multi_supervised_w": {}} for generative mode
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class LongVALE_Omni_Combined(nn.Module):
    """
    Qwen2.5-Omni thinker fine-tuned with LoRA on LongVALE features.
    Video + audio prefix tokens are concatenated before the text tokens.
    """

    N_VIS = 100   # fixed number of visual feature tokens
    N_AUD = 52    # fixed number of audio feature tokens
    FEAT_DIM = 768

    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        self.args = args

        model_name: str = getattr(args, "model_name", "Qwen/Qwen2.5-Omni-3B")
        hf_cache: Optional[str] = (
            getattr(args, "hf_cache", None) or getattr(args, "save_base_dir", None)
        )

        # ------------------------------------------------------------------
        # Load Qwen2.5-Omni; keep only thinker to save memory
        # ------------------------------------------------------------------
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration
            _cls = Qwen2_5OmniForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "transformers >= 4.51 with Qwen2_5OmniForConditionalGeneration is required"
            ) from exc

        full_model = _cls.from_pretrained(
            model_name,
            cache_dir=hf_cache,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        self.thinker = full_model.thinker
        # Free talker weights
        if hasattr(full_model, "talker"):
            del full_model.talker
        del full_model

        # d_model from thinker config
        t_cfg = self.thinker.config
        if hasattr(t_cfg, "text_config"):
            self.d_model: int = int(t_cfg.text_config.hidden_size)
        else:
            self.d_model: int = int(t_cfg.hidden_size)

        # ------------------------------------------------------------------
        # Modality projectors (trained from scratch)
        # ------------------------------------------------------------------
        self.video_proj = nn.Linear(self.FEAT_DIM, self.d_model)
        self.audio_proj = nn.Linear(self.FEAT_DIM, self.d_model)

        # ------------------------------------------------------------------
        # Apply LoRA to the thinker language model
        # ------------------------------------------------------------------
        self._apply_lora()

        # ------------------------------------------------------------------
        # Freeze backbone; keep projectors + LoRA trainable
        # ------------------------------------------------------------------
        self._setup_trainables()

    # ------------------------------------------------------------------
    def _apply_lora(self):
        cfg = getattr(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return
        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA but is not installed")

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
        )
        self.thinker = get_peft_model(self.thinker, lora_cfg)
        try:
            self.thinker.print_trainable_parameters()
        except Exception:
            pass

    def _setup_trainables(self):
        # Freeze thinker
        for p in self.thinker.parameters():
            p.requires_grad = False
        # Re-enable LoRA adapters
        for n, p in self.thinker.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        # Projectors always trainable
        for p in self.video_proj.parameters():
            p.requires_grad = True
        for p in self.audio_proj.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    def _get_embed_tokens(self):
        """Navigate PEFT wrapping to find embed_tokens."""
        thinker = self.thinker
        # Possible attribute paths
        for attr in ("model", "base_model"):
            inner = getattr(thinker, attr, None)
            if inner is None:
                continue
            if hasattr(inner, "embed_tokens"):
                return inner.embed_tokens
            for attr2 in ("model", "base_model"):
                inner2 = getattr(inner, attr2, None)
                if inner2 is not None and hasattr(inner2, "embed_tokens"):
                    return inner2.embed_tokens
        raise AttributeError("Cannot locate embed_tokens in thinker")

    # ------------------------------------------------------------------
    def _build_prefix(
        self,
        video_feats: torch.Tensor,  # (B, N_VIS, 768)
        audio_feats: torch.Tensor,  # (B, N_AUD, 768)
    ):
        """Project and cast feature tokens to the model dtype."""
        dtype = next(self.thinker.parameters()).dtype
        video_emb = self.video_proj(video_feats.to(torch.float32)).to(dtype)  # (B, N_VIS, d)
        audio_emb = self.audio_proj(audio_feats.to(torch.float32)).to(dtype)  # (B, N_AUD, d)
        return video_emb, audio_emb

    # ------------------------------------------------------------------
    def forward(
        self,
        data: Dict[str, Any],
        label=None,
        return_features: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        video_feats: torch.Tensor = data["video_feats"]   # (B, 100, 768)
        audio_feats: torch.Tensor = data["audio_feats"]   # (B,  52, 768)
        input_ids: torch.Tensor   = data["input_ids"]     # (B, T)
        label_ids: torch.Tensor   = data["label_ids"]     # (B, T) -100 for Q
        attn_mask: torch.Tensor   = data["attention_mask"]# (B, T)

        B, T = input_ids.shape
        device = input_ids.device

        video_emb, audio_emb = self._build_prefix(video_feats, audio_feats)
        n_prefix = video_emb.shape[1] + audio_emb.shape[1]  # 100 + 52 = 152

        # Text embeddings
        embed_fn = self._get_embed_tokens()
        dtype = next(self.thinker.parameters()).dtype
        text_emb = embed_fn(input_ids).to(dtype)  # (B, T, d_model)

        # Full embeddings: [video | audio | text]
        inputs_embeds = torch.cat([video_emb, audio_emb, text_emb], dim=1)  # (B, 152+T, d)

        # Attention mask: 1 for all prefix + real text tokens, 0 for text padding
        prefix_mask = torch.ones(B, n_prefix, dtype=attn_mask.dtype, device=device)
        full_attn_mask = torch.cat([prefix_mask, attn_mask], dim=1)  # (B, 152+T)

        # Labels: -100 for prefix, real labels for text
        prefix_lbl = torch.full((B, n_prefix), -100, dtype=label_ids.dtype, device=device)
        full_labels = torch.cat([prefix_lbl, label_ids], dim=1)  # (B, 152+T)

        outputs = self.thinker(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn_mask,
            labels=full_labels,
            use_cache=False,
        )

        lm_loss = outputs.loss

        return {
            "preds": {},
            "losses": {"lm_loss": lm_loss},
            "features": {},
        }


# ---------------------------------------------------------------------------
# VideoOnly — audio zeroed at forward
# ---------------------------------------------------------------------------

class LongVALE_Omni_VideoOnly(LongVALE_Omni_Combined):
    """Uses video features only; audio is set to zero."""

    def forward(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        data = dict(data)
        data["audio_feats"] = torch.zeros_like(data["audio_feats"])
        return super().forward(data, **kwargs)


# ---------------------------------------------------------------------------
# AudioOnly — video zeroed at forward
# ---------------------------------------------------------------------------

class LongVALE_Omni_AudioOnly(LongVALE_Omni_Combined):
    """Uses audio features only; video is set to zero."""

    def forward(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        data = dict(data)
        data["video_feats"] = torch.zeros_like(data["video_feats"])
        return super().forward(data, **kwargs)


# ---------------------------------------------------------------------------
# RMask — random modality masking during training
# ---------------------------------------------------------------------------

class LongVALE_Omni_RMask(LongVALE_Omni_Combined):
    """
    Random modality masking during training.
    With probability rmask_p per sample:
      • first draw  → mask video
      • second draw → mask audio
      (draws are independent per modality)
    """

    def forward(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        data = dict(data)
        if self.training:
            p = float(getattr(self.args, "rmask_p", 0.15))
            B = data["video_feats"].shape[0]
            for i in range(B):
                if random.random() < p:
                    data["video_feats"] = data["video_feats"].clone()
                    data["video_feats"][i] = 0.0
                if random.random() < p:
                    data["audio_feats"] = data["audio_feats"].clone()
                    data["audio_feats"][i] = 0.0
        return super().forward(data, **kwargs)
