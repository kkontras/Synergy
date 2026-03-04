import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VivitConfig, VivitModel, Wav2Vec2Model

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    # x: [B, T, D], mask: [B, T]
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(x.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


def _maybe_apply_peft_lora(backbone: nn.Module, cfg, default_targets, module_name: str) -> nn.Module:
    if not cfg or not cfg.get("use_lora", False):
        return backbone
    if get_peft_model is None or LoraConfig is None:
        raise ImportError("peft is required for LoRA models but is not installed in the current environment.")

    targets = list(cfg.get("lora_target_modules", default_targets))
    lora_kwargs = dict(
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        target_modules=targets,
        bias=str(cfg.get("lora_bias", "none")),
    )
    # Do not set task_type for non-text backbones (e.g. Wav2Vec2/ViViT), otherwise
    # PEFT may wrap with a text-oriented forward signature (`input_ids`).
    if cfg.get("task_type", None) is not None and TaskType is not None:
        lora_kwargs["task_type"] = getattr(TaskType, str(cfg.get("task_type")), None)
    lora_cfg = LoraConfig(**lora_kwargs)
    wrapped = get_peft_model(backbone, lora_cfg)
    try:
        wrapped.print_trainable_parameters()
    except Exception:
        logging.info("LoRA enabled for %s with targets=%s", module_name, targets)
    return wrapped


class LoRALinear(nn.Module):
    """Simple LoRA adapter on top of a frozen Linear layer."""

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.r = int(r)
        self.scale = float(alpha) / float(max(self.r, 1))
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        if self.r > 0:
            self.lora_A = nn.Linear(in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            out = out + self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return out


class Audio_Wav2Vec2_LoRA_Pool(nn.Module):
    """
    CREMAD audio encoder without conformer:
    - always initializes from HF pretrained Wav2Vec2 weights
    - optional LoRA on backbone attention projections
    - linear projection + masked mean pool
    """

    def __init__(self, args, encs=None):
        super().__init__()
        self.args = args
        self.num_classes = int(args.num_classes)
        self.d_model = int(args.d_model)
        hf_cache = getattr(args, "hf_cache", None) or getattr(args, "save_base_dir", None)

        model_name = str(args.get("hf_audio_name", "facebook/wav2vec2-large-robust"))
        self.backbone = Wav2Vec2Model.from_pretrained(model_name, cache_dir=hf_cache)
        try:
            self.backbone.freeze_feature_encoder()
        except Exception:
            pass

        # Keep parity with previous CREMAD audio encoder footprint.
        if self.backbone.config.hidden_size >= 1024 and hasattr(self.backbone, "encoder") and len(self.backbone.encoder.layers) > 12:
            del self.backbone.encoder.layers[12:]

        self.backbone = _maybe_apply_peft_lora(
            self.backbone,
            getattr(self.args, "lora_config", None),
            default_targets=["q_proj", "v_proj"],
            module_name="wav2vec2",
        )

        hidden = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(hidden, self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x, **kwargs):
        wav = x[2]
        attn = x.get("attention_mask_audio", None) if isinstance(x, dict) else None
        if attn is not None:
            out = self.backbone(input_values=wav, attention_mask=attn)
            try:
                feat_mask = self.backbone._get_feature_vector_attention_mask(out.last_hidden_state.shape[1], attn)
            except Exception:
                feat_mask = None
        else:
            out = self.backbone(input_values=wav)
            feat_mask = None

        seq = self.proj(out.last_hidden_state)  # [B,T,D]
        pooled = _masked_mean(seq, feat_mask)

        if kwargs.get("detach_enc0", False):
            pooled = pooled.detach()
            seq = seq.detach()

        pred = self.classifier(pooled if not kwargs.get("detach_pred", False) else pooled.detach())
        return {
            "preds": {"combined": pred},
            "features": {"combined": pooled},
            "nonaggr_features": {"combined": seq},
        }


class Video_ViViT_LoRA_Pool(nn.Module):
    """
    CREMAD visual encoder without conformer:
    - always initializes from HF pretrained ViViT weights
    - optional LoRA on ViViT attention projections
    - token projection + CLS/mean pooling
    """

    def __init__(self, args, encs=None):
        super().__init__()
        self.args = args
        self.num_classes = int(args.num_classes)
        self.d_model = int(args.d_model)
        hf_cache = getattr(args, "hf_cache", None) or getattr(args, "save_base_dir", None)

        num_frames = int(args.get("num_frame", 3))
        model_name = str(args.get("hf_video_name", "google/vivit-b-16x2-kinetics400"))
        self.backbone = VivitModel.from_pretrained(
            model_name, num_frames=num_frames, ignore_mismatched_sizes=True, cache_dir=hf_cache
        )

        self.backbone = _maybe_apply_peft_lora(
            self.backbone,
            getattr(self.args, "lora_config", None),
            default_targets=["query", "value"],
            module_name="vivit",
        )

        hidden = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(hidden, self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x, **kwargs):
        # Dataset returns [B, C, T, H, W]; HF ViViT expects [B, T, C, H, W]
        video = x[1]
        if video.ndim == 5 and video.shape[1] in (1, 3):
            video = video.permute(0, 2, 1, 3, 4).contiguous()

        out = self.backbone(pixel_values=video)
        seq_raw = out.last_hidden_state
        seq = self.proj(seq_raw)

        pooled = None
        if getattr(out, "pooler_output", None) is not None:
            pooled = self.proj(out.pooler_output)
        else:
            pooled = seq[:, 0]

        if kwargs.get("detach_enc1", False):
            pooled = pooled.detach()
            seq = seq.detach()

        pred = self.classifier(pooled if not kwargs.get("detach_pred", False) else pooled.detach())
        return {
            "preds": {"combined": pred},
            "features": {"combined": pooled},
            "nonaggr_features": {"combined": seq},
        }


class Video_FacesProj_LoRA_Pool(nn.Module):
    """
    Visual encoder for precomputed CREMAD face features (x[3]) without conformer.
    Uses a LoRA-projected linear pooling head.

    Note: face features are already extracted by an upstream pretrained visual model.
    """

    def __init__(self, args, encs=None):
        super().__init__()
        self.args = args
        self.num_classes = int(args.num_classes)
        self.d_model = int(args.d_model)
        in_dim = int(args.get("face_feat_dim", 1408))

        lora_cfg = getattr(self.args, "lora_config", None)
        use_proj_lora = bool(lora_cfg and lora_cfg.get("use_lora", False))
        if use_proj_lora:
            self.proj = LoRALinear(
                in_features=in_dim,
                out_features=self.d_model,
                r=int(lora_cfg.get("lora_r", 8)),
                alpha=float(lora_cfg.get("lora_alpha", 16)),
                dropout=float(lora_cfg.get("lora_dropout", 0.0)),
                bias=True,
            )
        else:
            self.proj = nn.Linear(in_dim, self.d_model)

        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x, **kwargs):
        faces = x[3]  # [B, T, 1408]
        face_mask = x.get("attention_mask_face", None) if isinstance(x, dict) else None
        seq = self.proj(faces)
        pooled = _masked_mean(seq, face_mask)

        if kwargs.get("detach_enc1", False):
            pooled = pooled.detach()
            seq = seq.detach()

        pred = self.classifier(pooled if not kwargs.get("detach_pred", False) else pooled.detach())
        return {
            "preds": {"combined": pred},
            "features": {"combined": pooled},
            "nonaggr_features": {"combined": seq},
        }


# Re-export aliases for clarity in configs while reusing existing fusion implementations.
from .MCR_Models import Base_Ensemble_Model, Base_Model, MCR_Model  # noqa: E402,F401
