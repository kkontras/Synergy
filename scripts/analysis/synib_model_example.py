import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict


def get_cfg(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class EMAFeatureStats(nn.Module):
    def __init__(self, feature_dim: int, ema_beta: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        self.register_buffer("ema_ex", torch.zeros(self.feature_dim))
        self.register_buffer("ema_ex2", torch.zeros(self.feature_dim))
        self.register_buffer("num_updates", torch.zeros(()))

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        if z.numel() == 0:
            return
        x = z.detach()
        x = x.reshape(-1, x.shape[-1]) if x.dim() > 1 else x[None, :]
        ex = x.mean(0)
        ex2 = (x * x).mean(0)
        beta = self.ema_beta if self.num_updates.item() > 0 else 0.0
        self.ema_ex.lerp_(ex, 1.0 - beta)
        self.ema_ex2.lerp_(ex2, 1.0 - beta)
        self.num_updates.add_(1)

    def sample_noise_like(self, z: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        mu = self.ema_ex
        var = (self.ema_ex2 - mu * mu).clamp_min(self.eps)
        shape = [1] * (z.dim() - 1) + [-1]
        mu = mu.view(*shape)
        std = (var.sqrt() * float(noise_scale)).view(*shape)
        return mu + torch.randn_like(z) * std


class SynIBMasker(nn.Module):
    """
    perturb.type:
      - "random": random corruption using p_min/p_max cosine schedule across K samples (returns K*B batch)
      - "learned": learn keep masks (returns B batch)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pcfg = get_cfg(cfg, "perturb", {}) or {}

        bias_cfg = get_cfg(cfg, "bias_infusion", {}) or {}
        self.synergy_weight = float(bias_cfg.get("l", 0.0))

        self.mask_mode = str(pcfg.get("type", "learned"))  # "random" or "learned"
        self.K = int(pcfg.get("num_samples", 1))

        self.noise_std = float(pcfg.get("noise_std", 1.0))
        self.p_min = float(pcfg.get("p_min", 0.3))
        self.p_max = float(pcfg.get("p_max", 0.9))
        self.cosine_s = 0.008

        self.steps = int(getattr(pcfg, "steps", 10))
        self.lr = float(getattr(pcfg, "lr", 1e-1))
        self.tau = float(getattr(pcfg, "tau", 1.0))
        self.lsparse = float(getattr(pcfg, "lsparse", 1.0))
        self.fill = str(getattr(pcfg, "fill", "ema"))  # "ema" | "zeros" | "shuffle"

        d = int(get_cfg(cfg, "fc_inner", 512))
        c = int(get_cfg(cfg, "num_classes", 10))
        self.logvar_head = nn.Linear(d, c)

        self.stats1 = EMAFeatureStats(d)
        self.stats2 = EMAFeatureStats(d)

    def _repeat_k(self, z: torch.Tensor) -> torch.Tensor:
        return z.unsqueeze(0).expand(self.K, *z.shape).reshape(self.K * z.shape[0], *z.shape[1:])

    def _keep_prob(self, k: int) -> float:
        if self.K <= 1:
            return self.p_max
        u = k / (self.K - 1)
        alpha_bar = math.cos(((u + self.cosine_s) / (1 + self.cosine_s)) * math.pi / 2) ** 2
        return self.p_min + (self.p_max - self.p_min) * alpha_bar

    def random_corrupt(self, z: torch.Tensor) -> torch.Tensor:
        zK = self._repeat_k(z)
        keep = torch.cat(
            [(torch.rand_like(z) < self._keep_prob(k)).to(z.dtype) for k in range(self.K)],
            dim=0,
        )
        shuffled = zK[torch.randperm(zK.size(0))]
        return (1 - keep) * zK + keep * shuffled

    def apply_fill(self, z: torch.Tensor, keep: torch.Tensor, stats: EMAFeatureStats) -> torch.Tensor:
        keep = keep.to(z.device).type_as(z)
        if self.fill == "zeros":
            eps = torch.zeros_like(z)
        elif self.fill == "ema":
            stats.update(z)
            eps = stats.sample_noise_like(z, noise_scale=self.noise_std)
        elif self.fill == "shuffle":
            eps = z[torch.randperm(z.size(0))]
        else:
            raise ValueError(f"Unknown fill mode {self.fill}")
        return keep * z + (1 - keep) * eps

    def learned_keep_mask(
        self,
        z: torch.Tensor,
        *,
        forward_unimodal_logits,
        label: torch.Tensor,
        stats: EMAFeatureStats,
    ) -> torch.Tensor:
        ell = torch.nn.Parameter(torch.ones_like(z))
        opt = torch.optim.Adam([ell], lr=self.lr)

        for _ in range(self.steps):
            keep = torch.sigmoid(ell / self.tau)
            zc = self.apply_fill(z, keep, stats)
            logits = forward_unimodal_logits(zc)
            obj = -F.cross_entropy(logits, label) + self.lsparse * (1 - keep).mean()
            opt.zero_grad(set_to_none=True)
            obj.backward()
            torch.nn.utils.clip_grad_norm_([ell], 1.0)
            opt.step()

        return torch.sigmoid(ell / self.tau).detach()

    @staticmethod
    def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, dim=1).mean()

    def synergy_kl(self, fusion_feat: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.synergy_weight <= 0:
            return fusion_feat.new_zeros(())
        logvar = self.logvar_head(fusion_feat)
        return self.gaussian_kl(logits, logvar) * self.synergy_weight


class FusionModelWithSynIB(nn.Module):
    def __init__(self, cfg, *, mod1_backbone, mod2_backbone, mod1_head, mod2_head, fusion_trunk, fusion_head):
        super().__init__()
        self.mod1_backbone = mod1_backbone
        self.mod2_backbone = mod2_backbone
        self.mod1_head = mod1_head
        self.mod2_head = mod2_head
        self.fusion_trunk = fusion_trunk
        self.fusion_head = fusion_head

        self.synib = SynIBMasker(cfg)
        self.synergy_weight = float(get_cfg(cfg, "bias_infusion", {}).get("l", 0.0))
        self.ending_epoch = int(get_cfg(cfg, "perturb", {}).get("ending_epoch", 10**9))

    def fuse_and_classify(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.fusion_trunk(z1, z2)
        logits = self.fusion_head(feat)
        return logits, feat

    def forward(self, x: torch.Tensor, *, label: Optional[torch.Tensor] = None, current_epoch: int = 0):
        z1 = self.mod1_backbone(x)
        z2 = self.mod2_backbone(x)
        z1 = F.layer_norm(z1, (z1.shape[-1],))
        z2 = F.layer_norm(z2, (z2.shape[-1],))

        uni1 = self.mod1_head(z1.detach())
        uni2 = self.mod2_head(z2.detach())

        fused_logits, fused_feat = self.fuse_and_classify(z1, z2)

        out = {
            "preds": {"combined": fused_logits, "mod1": uni1, "mod2": uni2},
            "features": {"fusion": fused_feat, "mod1_feat": z1, "mod2_feat": z2},
            "losses": {},
        }

        synib_on = (self.synergy_weight > 0) and (current_epoch <= self.ending_epoch)
        if not synib_on:
            return out
        if label is None:
            raise ValueError("When SynIB is enabled, pass label=...")

        z1_ = z1.detach()
        z2_ = z2.detach()

        if self.synib.mask_mode == "random":
            z1c = self.synib.random_corrupt(z1_)
            z2c = self.synib.random_corrupt(z2_)
            z1_rep = self.synib._repeat_k(z1_)
            z2_rep = self.synib._repeat_k(z2_)
        else:
            keep1 = self.synib.learned_keep_mask(
                z1_, forward_unimodal_logits=lambda zz: self.mod1_head(zz), label=label, stats=self.synib.stats1
            )
            keep2 = self.synib.learned_keep_mask(
                z2_, forward_unimodal_logits=lambda zz: self.mod2_head(zz), label=label, stats=self.synib.stats2
            )
            z1c = self.synib.apply_fill(z1_, keep1, self.synib.stats1)
            z2c = self.synib.apply_fill(z2_, keep2, self.synib.stats2)
            z1_rep = z1_
            z2_rep = z2_

        p0, f0 = self.fuse_and_classify(z1_rep, z2c)
        p1, f1 = self.fuse_and_classify(z1c, z2_rep)

        out["preds"].update({"mask0": p0, "mask1": p1})

        y0 = label.repeat(self.synib.K) if p0.shape[0] != label.shape[0] else label
        y1 = label.repeat(self.synib.K) if p1.shape[0] != label.shape[0] else label

        out["losses"].update(
            {
                "ce_mask0": F.cross_entropy(p0, y0),
                "ce_mask1": F.cross_entropy(p1, y1),
                "kl_synergy_1": self.synib.synergy_kl(f0, p0),
                "kl_synergy_2": self.synib.synergy_kl(f1, p1),
            }
        )
        return out


class SimpleBackbone(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleFusionTrunk(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(2 * feat_dim, feat_dim)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([z1, z2], dim=-1))


def build_cfg(*, mode: str, synergy_weight: float) -> EasyDict:
    return EasyDict(
        fc_inner=128,
        num_classes=5,
        bias_infusion={"l": float(synergy_weight)},
        perturb=EasyDict(
            type=mode,          # "random" or "learned"
            num_samples=3,      # used by random mode; learned mode will just ignore the repeat
            noise_std=1.0,
            p_min=0.3,
            p_max=0.9,
            steps=8,
            lr=1e-1,
            tau=1.0,
            lsparse=1.0,
            fill="ema",
            ending_epoch=1000,
        ),
    )


def run_one(mode: str, synergy_weight: float):
    cfg = build_cfg(mode=mode, synergy_weight=synergy_weight)
    in_dim = cfg.fc_inner
    feat_dim = cfg.fc_inner

    mod1_backbone = SimpleBackbone(in_dim, feat_dim)
    mod2_backbone = SimpleBackbone(in_dim, feat_dim)
    mod1_head = nn.Linear(feat_dim, cfg.num_classes)
    mod2_head = nn.Linear(feat_dim, cfg.num_classes)
    fusion_trunk = SimpleFusionTrunk(feat_dim)
    fusion_head = nn.Linear(feat_dim, cfg.num_classes)

    model = FusionModelWithSynIB(
        cfg,
        mod1_backbone=mod1_backbone,
        mod2_backbone=mod2_backbone,
        mod1_head=mod1_head,
        mod2_head=mod2_head,
        fusion_trunk=fusion_trunk,
        fusion_head=fusion_head,
    )

    x = torch.randn(4, in_dim)
    y = torch.randint(0, cfg.num_classes, (4,))

    out = model(x, label=y, current_epoch=0)

    print(f"\n=== mode={mode} synergy_weight={synergy_weight} ===")
    print("pred keys:", sorted(out["preds"].keys()))
    print("loss keys:", sorted(out["losses"].keys()))
    print("combined logits:", tuple(out["preds"]["combined"].shape))
    print("mask0 logits:", tuple(out["preds"]["mask0"].shape))
    for k, v in out["losses"].items():
        print(f"{k}: {float(v.detach()):.4f}")


def main():
    torch.manual_seed(0)
    run_one(mode="random", synergy_weight=0.2)
    run_one(mode="learned", synergy_weight=0.2)


if __name__ == "__main__":
    main()
