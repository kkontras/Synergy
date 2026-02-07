import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict


def _cfg(args, key, default=None):
    """Small helper to read attributes or dict keys with a default."""
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def _as_tensor_features(out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract feature tensors from an encoder output dict."""
    return out["z"], out.get("na_z", out["z"])


def _as_tensor_preds(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract prediction logits from an encoder output dict."""
    return out["pred"]


class FeatureStatsMasker(nn.Module):
    """Tracks feature-wise EMA stats and samples noise with matching mean/std."""

    def __init__(self, d1, ema_beta=0.99, eps=1e-6, device=None, dtype=None):
        super().__init__()
        factory_kwargs = dict(device=device, dtype=dtype)
        self.d1 = int(d1)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)

        self.register_buffer("ex", torch.zeros(self.d1, **factory_kwargs))
        self.register_buffer("ex2", torch.zeros(self.d1, **factory_kwargs))
        self.register_buffer("n", torch.zeros((), **factory_kwargs))

    @torch.no_grad()
    def ema_update(self, z: torch.Tensor) -> None:
        """Update EMA of E[x] and E[x^2] over all leading dims."""
        x = z.detach()
        if x.numel() == 0:
            return

        x = x.reshape(-1, x.shape[-1]) if x.dim() > 1 else x[None, :]
        if x.shape[-1] != self.d1:
            raise ValueError(f"Expected feature dim {self.d1}, got {x.shape[-1]}")

        batch_ex = x.mean(0)
        batch_ex2 = (x * x).mean(0)

        beta = self.ema_beta if self.n.item() > 0 else 0.0
        self.ex.lerp_(batch_ex, 1.0 - beta)
        self.ex2.lerp_(batch_ex2, 1.0 - beta)
        self.n.add_(1)

    def feature_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.ex
        var = (self.ex2 - mu * mu).clamp_min(self.eps)
        return mu, var

    def noise_like(self, z: torch.Tensor, noise_scale=1.0) -> torch.Tensor:
        mu, var = self.feature_stats()
        shape = [1] * (z.dim() - 1) + [-1]
        mu = mu.view(*shape)
        std = (var.sqrt() * float(noise_scale)).view(*shape)
        return mu + torch.randn_like(z) * std


class SynIB(nn.Module):
    """Synergy module that constructs random and learnable feature masks."""

    def __init__(self, args, encs, main):
        super().__init__()
        object.__setattr__(self, "main", main)

        self.args = args
        self.perturb = _cfg(args, "perturb", {}) or {}
        self.reestimate_features = bool(self.perturb.get("reestimate_features", False))

        bias = _cfg(args, "bias_infusion", {}) or {}
        self.synergy_weight = float(bias.get("l", 0.0))
        self.contrastive_weight = float(bias.get("contrcoeff", 0.0) or 0.0)
        self.synergy_type = getattr(args, "synergy_type", "gaussian")

        fc_inner = int(_cfg(args, "fc_inner"))
        num_classes = int(_cfg(args, "num_classes"))

        if self.synergy_type == "gaussian":
            self.logvar_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = None
        elif self.synergy_type == "dirichlet":
            self.evidence_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = float(_cfg(args, "dirichlet_prior_conc", 1.0))
        else:
            raise ValueError(f"Unknown synergy_type: {self.synergy_type}")

        self.cls_type = _cfg(args, "cls_type")
        self.noise_std = float(self.perturb.get("noise_std", 1.0))
        self.K = int(self.perturb.get("num_samples", 1))
        self.fill = self.perturb.get("fill", "ema")
        self.p_type = self.perturb.get("type", "diff")

        self.p_min = float(self.perturb.get("p_min", 0.3))
        self.p_max = float(self.perturb.get("p_max", 0.9))
        self.cosine_s = 0.008

        if self.cls_type == "mlp":
            self.stats_z1 = FeatureStatsMasker(d1=512, ema_beta=0.99)
            self.stats_z2 = FeatureStatsMasker(d1=512, ema_beta=0.99)
        elif self.cls_type == "tf":
            self.stats_na_z1 = FeatureStatsMasker(d1=512, ema_beta=0.99)
            self.stats_na_z2 = FeatureStatsMasker(d1=512, ema_beta=0.99)

    @staticmethod
    def _gaussian_kl(mu, logvar):
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, dim=1).mean()

    @staticmethod
    def _dirichlet_kl(alpha, prior_conc=1.0):
        alpha0 = torch.full_like(alpha, prior_conc) if isinstance(prior_conc, float) else prior_conc
        alpha0_sum = alpha0.sum(dim=1, keepdim=True)
        alpha_sum = alpha.sum(dim=1, keepdim=True)

        lgamma = torch.lgamma
        digamma = torch.digamma

        logB_alpha = torch.sum(lgamma(alpha), dim=1) - lgamma(alpha_sum.squeeze(1))
        logB_alpha0 = torch.sum(lgamma(alpha0), dim=1) - lgamma(alpha0_sum.squeeze(1))

        term1 = logB_alpha0 - logB_alpha
        term2 = torch.sum((alpha - alpha0) * (digamma(alpha) - digamma(alpha_sum)), dim=1)
        return (term1 + term2).mean()

    def _get_diff_p(self, t):
        """Cosine schedule from p_max to p_min across K steps."""
        u = (torch.as_tensor(t) / (self.K - 1)).clamp(0.0, 1.0)
        alpha_bar = torch.cos(((u + self.cosine_s) / (1.0 + self.cosine_s)) * (math.pi / 2.0)) ** 2
        return self.p_min + (self.p_max - self.p_min) * alpha_bar

    def get_random_mask_multiclass(self, features, **kwargs) -> Dict[str, torch.Tensor]:
        z1, z2, na_z1, na_z2 = features["z1"], features["z2"], features["na_z1"], features["na_z2"]

        def repeat_k(z):
            return z.unsqueeze(0).expand(self.K, *z.shape).reshape(self.K * z.shape[0], *z.shape[1:])

        def make_keep(z, p):
            return (torch.rand_like(z) < p).to(z.dtype)

        def make_keep_token(z, p):
            return (torch.rand_like(z[:, :, 0]) < p).to(z.dtype)

        def noise_fn(z, ema):
            if self.fill == "zeros":
                return torch.zeros_like(z)
            if self.fill == "noise":
                return torch.randn_like(z) * self.noise_std
            return ema.noise_like(z, self.noise_std)

        def fill_func(z, keep, ema=None):
            eps = z[torch.randperm(z.size(0))]
            return (1 - keep) * z + keep * eps

        def make_tilde_once(z, ema):
            zK = repeat_k(z)
            if self.fill == "token":
                token_mask = make_keep_token(zK, self.p)
                return zK, None, token_mask
            keep = make_keep(zK, self.p)
            tzK = fill_func(zK, keep, ema)
            return zK, tzK, None

        def make_tilde_diff(z, ema):
            zK = repeat_k(z)
            if self.fill == "token":
                token_mask = torch.cat([make_keep_token(z, self._get_diff_p(k)) for k in range(self.K)], dim=0)
                return zK, zK, token_mask
            keep = torch.cat([make_keep(z, self._get_diff_p(k)) for k in range(self.K)], dim=0)
            tzK = fill_func(zK, keep, ema)
            return zK, tzK, None

        make_tilde_fn = make_tilde_diff if self.p_type == "diff" else make_tilde_once

        if self.cls_type == "mlp":
            self.stats_z1.ema_update(z1)
            self.stats_z2.ema_update(z2)
            z1K, tz1K, token_mask1 = make_tilde_fn(z1, self.stats_z1)
            z2K, tz2K, token_mask2 = make_tilde_fn(z2, self.stats_z2)
            na_z1K, na_z2K, na_tz1K, na_tz2K = na_z1, na_z2, na_z1, na_z2
        else:
            self.stats_na_z1.ema_update(na_z1)
            self.stats_na_z2.ema_update(na_z2)
            na_z1K, na_tz1K, token_mask1 = make_tilde_fn(na_z1, self.stats_na_z1)
            na_z2K, na_tz2K, token_mask2 = make_tilde_fn(na_z2, self.stats_na_z2)
            z1K, tz1K, z2K, tz2K = z1, z1, z2, z2

        return {
            "z1K": z1K,
            "tz1K": tz1K,
            "z2K": z2K,
            "tz2K": tz2K,
            "na_z1K": na_z1K,
            "na_tz1K": na_tz1K,
            "na_z2K": na_z2K,
            "na_tz2K": na_tz2K,
            "mask1": token_mask1,
            "mask2": token_mask2,
        }

    def get_learnable_mask_multiclass(self, x, features, preds, **kwargs) -> Dict[str, torch.Tensor]:
        y = kwargs["label"]
        device = y.device
        pcfg = self.args.perturb
        steps = int(getattr(pcfg, "steps", 20))
        lr = float(getattr(pcfg, "lr", 1e-1))
        tau = float(getattr(pcfg, "tau", 1.0))
        noise_std = float(getattr(pcfg, "noise_std", 1.0))
        lsparse = float(getattr(pcfg, "lsparse", 1))
        hard_thresh = float(getattr(pcfg, "hard_thresh", 0.5))
        fill_mode = getattr(pcfg, "fill", "ema")

        def _forward_probs(feat_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            p_f, _ = self.main._compute_logits(
                feat_dict["z1"], feat_dict["z2"], feat_dict["na_z1"], feat_dict["na_z2"]
            )
            p_u1 = self.main.enc_0.forward_uni(feat_dict["z1"], feat_dict["na_z1"], detach_pred=False)
            p_u2 = self.main.enc_1.forward_uni(feat_dict["z2"], feat_dict["na_z2"], detach_pred=False)
            return p_f, p_u1, p_u2

        def _apply_destroy(z: torch.Tensor, g: torch.Tensor, ema_stats=None, inv_mask=False) -> torch.Tensor:
            g_ = g.view(1, -1) if g.dim() == 1 else g
            g_ = g_.to(z.device).type_as(z)

            if fill_mode == "zeros":
                eps = torch.zeros_like(z)
            elif fill_mode == "ema":
                eps = ema_stats.noise_like(z, noise_std) if ema_stats is not None else (torch.randn_like(z) * noise_std)
            elif fill_mode == "shuffle":
                eps = z[torch.randperm(z.size(0))]
            else:
                raise ValueError(f"Unknown fill mode {fill_mode}")

            return g_ * z + (1 - g_) * eps if inv_mask else (1 - g_) * z + g_ * eps

        def freeze_model_(m: nn.Module) -> None:
            for p in m.parameters():
                p.requires_grad_(False)

        req = [p.requires_grad for p in self.main.parameters()]
        freeze_model_(self.main)
        try:
            z1, z2, na_z1, na_z2 = features["z1"], features["z2"], features["na_z1"], features["na_z2"]
            z1 = z1.detach()
            z2 = z2.detach()

            _ = preds["c"], preds["g"], preds["combined"]
            _ = _forward_probs(features)

            with torch.enable_grad():
                this_z1 = z1 if self.cls_type == "mlp" else na_z1
                this_key = "z1" if self.cls_type == "mlp" else "na_z1"
                this_stats1 = self.stats_z1 if self.cls_type == "mlp" else self.stats_na_z1
                this_stats1.ema_update(this_z1)
                ell1 = torch.nn.Parameter(torch.ones(this_z1.shape, device=device), requires_grad=True)
                opt1 = torch.optim.Adam([ell1], lr=lr)

                for _ in range(steps):
                    g1 = torch.sigmoid(ell1 / tau)
                    tz1 = _apply_destroy(this_z1, g1, ema_stats=this_stats1, inv_mask=True)
                    feat_t = dict(features)
                    feat_t[this_key] = tz1
                    _, p_u_t, _ = _forward_probs(feat_t)
                    sparsity = (1 - g1).mean()
                    obj1 = -F.cross_entropy(p_u_t, y) + float(lsparse) * sparsity
                    opt1.zero_grad(set_to_none=True)
                    obj1.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_([ell1], 1.0)
                    opt1.step()

                g1_final = torch.sigmoid(ell1 / tau)

                this_z2 = z2 if self.cls_type == "mlp" else na_z2
                this_key = "z2" if self.cls_type == "mlp" else "na_z2"
                this_stats2 = self.stats_z2 if self.cls_type == "mlp" else self.stats_na_z2
                this_stats2.ema_update(this_z2)
                ell2 = torch.nn.Parameter(torch.ones(this_z2.shape, device=device), requires_grad=True)
                opt2 = torch.optim.Adam([ell2], lr=lr)

                for _ in range(steps):
                    g2 = torch.sigmoid(ell2 / tau)
                    tz2 = _apply_destroy(this_z2, g2, ema_stats=this_stats2, inv_mask=True)
                    feat_t = dict(features)
                    feat_t[this_key] = tz2
                    _, _, p_u_t = _forward_probs(feat_t)
                    sparsity = (1 - g2).mean()
                    obj2 = -F.cross_entropy(p_u_t, y) + float(lsparse) * sparsity
                    opt2.zero_grad(set_to_none=True)
                    obj2.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_([ell2], 1.0)
                    opt2.step()

                g2_final = torch.sigmoid(ell2 / tau)

        finally:
            self.main.train()
            for p, r in zip(self.main.parameters(), req):
                p.requires_grad_(r)

        if self.cls_type == "mlp":
            z1K, z2K = z1, z2
            tz1K = _apply_destroy(this_z1, g1_final, ema_stats=this_stats1, inv_mask=True)
            tz2K = _apply_destroy(this_z2, g2_final, ema_stats=this_stats2, inv_mask=True)
            na_z1K, na_z2K = na_z1, na_z2
            na_tz1K, na_tz2K = na_z1, na_z2
        else:
            na_z1K, na_z2K = na_z1, na_z2
            na_tz1K = _apply_destroy(this_z1, g1_final, ema_stats=this_stats1, inv_mask=True)
            na_tz2K = _apply_destroy(this_z2, g2_final, ema_stats=this_stats2, inv_mask=True)
            z1K, z2K = z1, z2
            tz1K, tz2K = z1, z2

        return {
            "z1K": z1K,
            "tz1K": tz1K,
            "z2K": z2K,
            "tz2K": tz2K,
            "na_z1K": na_z1K,
            "na_tz1K": na_tz1K,
            "na_z2K": na_z2K,
            "na_tz2K": na_tz2K,
            "mask1": g1_final.detach(),
            "mask2": g2_final.detach(),
        }

    def _kl_loss(self, mu, feat):
        if self.synergy_type == "gaussian":
            logvar = self.logvar_head(feat)
            kl = self._gaussian_kl(mu, logvar)
        else:
            evidence = F.softplus(self.evidence_head(feat))
            alpha = evidence + 1.0
            kl = self._dirichlet_kl(alpha, prior_conc=self.dirichlet_prior_conc)
        return kl * self.synergy_weight

    def _kl_pass(self, feat, mu, name, **kwargs):
        return {name: self._kl_loss(mu, feat)}

    def ce_losses(self, base_output, **kwargs):
        loss = {}
        for k, pred in base_output["preds"].items():
            this_label = kwargs["label"].repeat(self.K) if pred.shape[0] != kwargs["label"].shape[0] else kwargs["label"]
            loss.update({k: F.cross_entropy(pred, this_label)})
        return loss

    def compute_training_losses(self, base_output, **kwargs):
        losses = {}
        ce_losses = self.ce_losses(base_output, **kwargs)
        losses.update({"ce_mask0": ce_losses["mask0"]})
        losses.update({"ce_mask1": ce_losses["mask1"]})
        losses.update(
            self._kl_pass(base_output["features"]["mask0"], base_output["preds"]["mask0"], name="kl_synergy_1", **kwargs)
        )
        losses.update(
            self._kl_pass(base_output["features"]["mask1"], base_output["preds"]["mask1"], name="kl_synergy_2", **kwargs)
        )
        return losses


class FusionIBModel_Mask(nn.Module):
    """Fusion model that integrates SynIB masking into the forward path."""

    def __init__(self, args, encs):
        super().__init__()
        self.args = args
        self.cls_type = _cfg(args, "cls_type")
        self.norm_decision = _cfg(args, "norm_decision", False)

        self.num_classes = int(_cfg(args, "num_classes"))
        d_model = int(args.get("d_model", 512))
        fc_inner = int(args.get("fc_inner", 512))
        dropout = float(_cfg(args, "dropout", 0.1))

        self.synergy_weight = float(_cfg(args, "bias_infusion", {}).get("l", 0.0))
        self.ending_epoch = int(_cfg(args, "perturb", {}).get("ending_epoch", 1000.0))

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "mlp":
            if len(encs) > 2:
                self.enc_2 = encs[2]
                self.enc_3 = encs[3]
            else:
                self.enc_2 = FusionTrunkLinear(args)
                self.enc_3 = LinearHead(args)
        else:
            self.enc_2 = TF_Fusion_Transformer(input_dim=d_model, dim=d_model, layers=2, output_dim=fc_inner)
            self.enc_3 = LinearHead(args)

        if len(encs) > 4:
            self.enc_4 = encs[4]
            self.enc_5 = encs[5]
        else:
            self.enc_4 = MLPHead(args)
            self.enc_5 = MLPHead(args)

        self.synib = SynIB(args, [], main=self)

    def _get_features(self, x, **kwargs):
        out0 = self.enc_0(x, detach_pred=True, **kwargs)
        out1 = self.enc_1(x, detach_pred=True, **kwargs)
        z1, na_z1 = _as_tensor_features(out0)
        z2, na_z2 = _as_tensor_features(out1)
        preds1 = _as_tensor_preds(out0)
        preds2 = _as_tensor_preds(out1)
        z1 = F.layer_norm(z1, (z1.shape[-1],))
        z2 = F.layer_norm(z2, (z2.shape[-1],))
        na_z1 = F.layer_norm(na_z1, (na_z1.shape[-1],))
        na_z2 = F.layer_norm(na_z2, (na_z2.shape[-1],))
        return preds1, preds2, z1, z2, na_z1, na_z2

    def _compute_logits(self, z1, z2, na_z1=None, na_z2=None, att_mask1=None, att_mask2=None, **kwargs):
        if self.cls_type == "tf":
            feat = self.enc_2({0: na_z1, 1: na_z2}, att_mask1=att_mask1, att_mask2=att_mask2)
        else:
            feat = self.enc_2(z1, z2)
        logits = self.enc_3(feat)
        return logits, feat

    def _base_forward(self, x, **kwargs):
        uni_pred_1, uni_pred_2, z1, z2, na_z1, na_z2 = self._get_features(x, **kwargs)
        pred, feat = self._compute_logits(z1, z2, na_z1, na_z2)

        return {
            "preds": {"combined": pred, "c": uni_pred_1, "g": uni_pred_2},
            "features": {"combined": feat, "z1": z1, "z2": z2, "na_z1": na_z1, "na_z2": na_z2},
            "losses": {},
        }

    def _base_forward_synib(self, x, **kwargs):
        uni_pred_1, uni_pred_2, z1, z2, na_z1, na_z2 = self._get_features(x, **kwargs)
        pred, feat = self._compute_logits(z1, z2, na_z1, na_z2)

        features = {"combined": feat, "z1": z1, "z2": z2, "na_z1": na_z1, "na_z2": na_z2}
        preds = {"combined": pred, "c": uni_pred_1, "g": uni_pred_2}
        feat_tilde_random = self.synib.get_random_mask_multiclass(features)
        feat_tilde = self.synib.get_learnable_mask_multiclass(x, features, preds, **kwargs)

        pred_mask0, feat_mask0 = self._compute_logits(
            feat_tilde["z1K"], feat_tilde["tz2K"], feat_tilde["na_z1K"], feat_tilde["na_tz2K"], att_mask1=feat_tilde["mask1"]
        )
        pred_mask1, feat_mask1 = self._compute_logits(
            feat_tilde["tz1K"], feat_tilde["z2K"], feat_tilde["na_tz1K"], feat_tilde["na_z2K"], att_mask2=feat_tilde["mask2"]
        )
        pred_mask01, feat_mask01 = self._compute_logits(
            feat_tilde["tz1K"],
            feat_tilde["tz2K"],
            feat_tilde["na_tz1K"],
            feat_tilde["na_tz2K"],
            att_mask1=feat_tilde["mask1"],
            att_mask2=feat_tilde["mask2"],
        )

        pred_randmask0, feat_randmask0 = self._compute_logits(
            feat_tilde_random["z1K"],
            feat_tilde_random["tz2K"],
            feat_tilde_random["na_z1K"],
            feat_tilde_random["na_tz2K"],
            att_mask1=feat_tilde_random["mask1"],
        )
        pred_randmask1, feat_randmask1 = self._compute_logits(
            feat_tilde_random["tz1K"],
            feat_tilde_random["z2K"],
            feat_tilde_random["na_tz1K"],
            feat_tilde_random["na_z2K"],
            att_mask2=feat_tilde_random["mask2"],
        )
        pred_randmask01, feat_randmask01 = self._compute_logits(
            feat_tilde_random["tz1K"],
            feat_tilde_random["tz2K"],
            feat_tilde_random["na_tz1K"],
            feat_tilde_random["na_tz2K"],
            att_mask1=feat_tilde_random["mask1"],
            att_mask2=feat_tilde_random["mask2"],
        )

        preds.update(
            {
                "mask0": pred_mask0,
                "mask1": pred_mask1,
                "mask01": pred_mask01,
                "randmask0": pred_randmask0,
                "randmask1": pred_randmask1,
                "randmask01": pred_randmask01,
            }
        )
        features.update(
            {
                "mask0": feat_mask0,
                "mask1": feat_mask1,
                "mask01": feat_mask01,
                "randmask0": feat_randmask0,
                "randmask1": feat_randmask1,
                "randmask01": feat_randmask01,
            }
        )

        losses = {}
        ce_losses = self.synib.ce_losses({"preds": preds, "features": features}, **kwargs)
        losses.update({"ce_mask0": ce_losses["randmask0"]})
        losses.update({"ce_mask1": ce_losses["randmask1"]})
        losses.update(self.synib._kl_pass(feat_mask0, pred_mask0, name="kl_synergy_1", **kwargs))
        losses.update(self.synib._kl_pass(feat_mask1, pred_mask1, name="kl_synergy_2", **kwargs))

        return {"preds": preds, "features": features, "losses": losses}

    def forward(self, x, **kwargs):
        okay_epoch = False if "current_epoch" in kwargs and self.ending_epoch < kwargs["current_epoch"] else True
        if self.synergy_weight > 0 and okay_epoch:
            return self._base_forward_synib(x, **kwargs)
        return self._base_forward(x, **kwargs)


class FusionTrunkLinear(nn.Module):
    def __init__(self, args):
        super().__init__()
        fc_inner = int(_cfg(args, "fc_inner", 512))
        self.proj = nn.Linear(fc_inner * 2, fc_inner)

    def forward(self, z1, z2):
        return self.proj(torch.cat([z1, z2], dim=-1))


class LinearHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        fc_inner = int(_cfg(args, "fc_inner", 512))
        num_classes = int(_cfg(args, "num_classes", 10))
        self.fc = nn.Linear(fc_inner, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        fc_inner = int(_cfg(args, "fc_inner", 512))
        num_classes = int(_cfg(args, "num_classes", 10))
        self.net = nn.Sequential(nn.Linear(fc_inner, fc_inner), nn.ReLU(), nn.Linear(fc_inner, num_classes))

    def forward(self, x):
        return self.net(x)


class TF_Fusion_Transformer(nn.Module):
    def __init__(self, input_dim, dim, layers, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, inputs, att_mask1=None, att_mask2=None):
        z1 = inputs[0]
        z2 = inputs[1]
        return self.proj(z1 + z2)


class DummyEncoder(nn.Module):
    """Minimal encoder used to demo instantiation without project dependencies."""

    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.backbone = nn.Linear(feature_dim, feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x, detach_pred=False, **kwargs):
        z = self.backbone(x)
        pred = self.head(z)
        if detach_pred:
            pred = pred.detach()
        return {"z": z, "na_z": z, "pred": pred}

    def forward_uni(self, z, na_z, detach_pred=False):
        pred = self.head(z)
        if detach_pred:
            pred = pred.detach()
        return pred


def build_config() -> EasyDict:
    """Create an EasyDict config with commonly-tuned parameters."""
    return EasyDict(
        cls_type="mlp",
        num_classes=10,
        fc_inner=512,
        d_model=512,
        dropout=0.1,
        synergy_type="gaussian",
        bias_infusion={"l": 0.1, "contrcoeff": 0.0},
        perturb=EasyDict(
            num_samples=1,
            noise_std=1.0,
            type="diff",
            p_min=0.3,
            p_max=0.9,
            steps=10,
            lr=1e-1,
            tau=1.0,
            lsparse=1.0,
            hard=True,
            hard_thresh=0.5,
            fill="ema",
            ending_epoch=1000,
        ),
    )


def main() -> None:
    cfg = build_config()
    encs = [DummyEncoder(), DummyEncoder()]
    model = FusionIBModel_Mask(cfg, encs)

    batch = torch.randn(4, cfg.fc_inner)
    labels = torch.randint(0, cfg.num_classes, (4,))
    output = model(batch, label=labels, current_epoch=0)
    print("Output keys:", output.keys())


if __name__ == "__main__":
    main()
