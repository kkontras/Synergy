import copy

from models.model_utils.fusion_gates import *
from models.VAVL_git.VAVL.conformer.model import Conformer
from models.MCR_Models import TF_Fusion
from pytorch_metric_learning.losses import NTXentLoss
from torch.nn.utils import spectral_norm as SN
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from transformers import AutoTokenizer
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
from peft import LoraConfig, get_peft_model
import torch

class TF_Proc(nn.Module):
    def __init__(self, input_dim, dim, layers, output_dim):
        super(TF_Proc, self).__init__()
        self.common_net = Conformer(
                            input_dim=input_dim,
                            encoder_dim=dim,
                            num_encoder_layers=layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)

        self.common_fc = nn.Linear(dim, output_dim)


    def forward(self, x, **kwargs):
        x_0 = x.permute(0,2,1)

        x_0 = self.cls_token.repeat(x_0.shape[0], x_0.shape[1], 1) + x_0

        if "detach_feat" in kwargs and kwargs["detach_feat"]:
            x_0 = x_0.detach()

        feat_mm = torch.concatenate([self.cls_token.repeat(x_0.shape[0], 1, 1), x_0], dim=1)
        feat_mm = self.common_net(feat_mm)
        aggr_feat_mm = feat_mm[:,0]

        pred = self.common_fc(aggr_feat_mm)
        if kwargs.get("return_all", False):
            return pred, aggr_feat_mm, feat_mm
        else:
            return pred
def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)
def cosine_clamp_loss(z12, z12_hat):
    cos_sim = F.cosine_similarity(z12, z12_hat, dim=-1)
    # want similarity low → hinge on (1 - cos_sim)
    return torch.clamp(cos_sim, min=0)
def nt_xent_loss(z_x, z_y, z_k=None, label=None, temperature=0.5):
    # z_x = F.normalize(z_x, p=2, dim=1)
    # z_y = F.normalize(z_y, p=2, dim=1)
    z = torch.cat([z_x, z_y], dim=0)
    if label is not None:
        labels = torch.cat([label, label], dim=0)
    else:
        labels = torch.cat([torch.arange(z_x.shape[0]), torch.arange(z_y.shape[0])], dim=0)
    if z_k is not None:
        z_k = F.normalize(z_k, p=2, dim=1)
        z = torch.cat([z, z_k], dim=0)
        if label is not None:
            labels = torch.cat([label, label, label], dim=0)
        else:
            labels = torch.cat([torch.arange(z_x.shape[0]), torch.arange(z_y.shape[0]), torch.arange(z_k.shape[0])],
                               dim=0)

    loss = NTXentLoss(temperature=temperature)(z, labels)

    return loss

    def conditional_alignment_loss(feat1, feat2, labels, temperature=0.1):
        """
        InfoNCE-style loss that encourages alignment between feat1 and feat2
        for samples sharing the same label.
        """
        # Normalize features
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)

        # Compute similarity matrix
        sim = torch.matmul(feat1, feat2.T) / temperature  # [N, N]

        # Build mask for same labels
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()  # [N, N]
        mask.fill_diagonal_(0)  # remove self-similarity

        # Log-softmax over similarities
        log_sim = F.log_softmax(sim, dim=1)

        # Positive pairs = same label
        pos = (log_sim * mask).sum(1) / mask.sum(1).clamp(min=1)

        # Take mean over batch
        loss = -pos.mean()
        return loss
def synergy_confidence_loss(pred_fusion, unimodal_preds, labels, margin=0.0):
    labels = labels.view(-1, 1)
    p_fusion = F.softmax(pred_fusion, dim=1).gather(1, labels)
    p_unis = [F.softmax(p, dim=1).gather(1, labels).detach() for p in unimodal_preds]
    p_uni_mean = torch.stack(p_unis, dim=0).mean(0)
    # Penalize only when fusion < unimodal - margin
    loss = torch.mean(F.relu(p_uni_mean - p_fusion + margin))
    return loss


class Fusion_Synprom_IB_Dir_SupRem(nn.Module):
    def __init__(self, args, encs):
        super().__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.num_classes = args.num_classes

        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        bias_infusion = args.get("bias_infusion", {})
        self.l = bias_infusion.get("l", 0.0)
        self.contrcoeff = bias_infusion.get("contrcoeff", False)

        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner),
            )

        elif self.cls_type == "conformer":
            self.common_fc_1 = TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6,
                output_dim=d_model,
            )
            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner),
            )

        else:
            raise ValueError(f"Unknown cls_type: {self.cls_type}")

        self.alpha_head = nn.Linear(fc_inner, self.num_classes)
        prior_conc = args.get("dir_prior_conc", 1.0)
        self.register_buffer("alpha0", prior_conc * torch.ones(self.num_classes))

        self.d_model = d_model
        self.sup_probe_head = nn.Linear(2 * d_model, self.num_classes)
        self.sup_probe_coeff = args.get("sup_probe_coeff", 0.0)

        self.register_buffer("I_K", torch.eye(self.num_classes))
        self.register_buffer("I_d", torch.eye(d_model))

    def _get_features(self, x, **kwargs):
        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)
        return a, v

    def _fuse_and_dirichlet(self, a, v, **kwargs):
        if self.cls_type == "mlp":
            z = torch.cat(
                (a["features"]["combined"], v["features"]["combined"]), dim=1
            )
            aggr_feat = self.common_fc_1(z)
            feat = self.common_fc_2(aggr_feat)

        elif self.cls_type == "conformer":
            _, aggr_feat, _ = self.common_fc_1(
                [a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]],
                return_all=True,
            )
            feat = self.common_fc_2(aggr_feat)

        alpha_logits = self.alpha_head(feat)
        alpha = F.softplus(alpha_logits) + 1e-4
        return alpha, aggr_feat

    def _supervised_project_pair(self, z1, z2):
        with torch.no_grad():
            W = self.sup_probe_head.weight
            d = self.d_model
            W1 = W[:, :d]
            W2 = W[:, d:]

            M1 = W1 @ W1.t() + 1e-6 * self.I_K
            M1_inv = torch.inverse(M1)
            P_S1 = W1.t() @ M1_inv @ W1
            P_perp1 = self.I_d - P_S1

            M2 = W2 @ W2.t() + 1e-6 * self.I_K
            M2_inv = torch.inverse(M2)
            P_S2 = W2.t() @ M2_inv @ W2
            P_perp2 = self.I_d - P_S2

            tilde_z1 = z1 @ P_perp1.t()
            tilde_z2 = z2 @ P_perp2.t()

        return tilde_z1, tilde_z2

    def forward_once(
        self,
        x,
        *,
        stochastic: bool = False,
        perturb_x1: bool = False,
        perturb_x2: bool = False,
        **kwargs,
    ):
        a, v = self._get_features(x, **kwargs)

        if perturb_x1 or perturb_x2:
            z1 = a["features"]["combined"]
            z2 = v["features"]["combined"]
            tilde_z1, tilde_z2 = self._supervised_project_pair(z1, z2)
            if perturb_x1:
                a["features"]["combined"] = tilde_z1.detach()
            if perturb_x2:
                v["features"]["combined"] = tilde_z2.detach()

        alpha, _ = self._fuse_and_dirichlet(a, v, **kwargs)

        if stochastic:
            return self._kl_loss(alpha)

        pred_mean = alpha / alpha.sum(dim=-1, keepdim=True)

        output = {
            "preds": {
                "combined": pred_mean,
                "p1": a["preds"]["combined"],
                "p2": v["preds"]["combined"],
            },
            "features": {
                "z1": a["features"]["combined"],
                "z2": v["features"]["combined"],
            },
            "dirichlet": {
                "alpha": alpha,
            },
            "losses": {},
        }
        return output

    def _kl_loss(self, alpha):
        alpha0 = self.alpha0
        sum_alpha = alpha.sum(dim=-1)
        sum_alpha0 = alpha0.sum()

        log_B_alpha0 = torch.lgamma(alpha0).sum() - torch.lgamma(sum_alpha0)
        log_B_alpha = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(sum_alpha)
        term1 = log_B_alpha0 - log_B_alpha

        digamma_alpha = torch.digamma(alpha)
        digamma_sum_alpha = torch.digamma(sum_alpha).unsqueeze(-1)
        term2 = ((alpha - alpha0) * (digamma_alpha - digamma_sum_alpha)).sum(dim=-1)

        kl = term1 + term2
        return kl.mean()

    def _dirichlet_classification_loss(self, alpha, label):
        """
        Expected NLL under Dirichlet over class probs:
        E_{p ~ Dir(alpha)}[-log p_y] = psi(sum_k alpha_k) - psi(alpha_y)
        """
        sum_alpha = alpha.sum(dim=-1)                 # (B,)
        alpha_y = alpha[torch.arange(alpha.size(0), device=alpha.device), label]
        loss = torch.digamma(sum_alpha) - torch.digamma(alpha_y)
        return loss.mean()

    def forward(self, x, **kwargs):
        output = self.forward_once(x, stochastic=False, **kwargs)

        alpha = output["dirichlet"]["alpha"]

        if self.training and ("label" in kwargs):
            label = kwargs["label"]
            cls_loss = self._dirichlet_classification_loss(alpha, label)
            output["losses"]["cls_dir"] = cls_loss

        if self.training and self.sup_probe_coeff > 0.0 and (
            ("label" in kwargs) or ("y" in kwargs)
        ):
            label = kwargs.get("label", kwargs.get("y"))
            z1 = output["features"]["z1"]
            z2 = output["features"]["z2"]
            concat = torch.cat([z1, z2], dim=1)
            logits_probe = self.sup_probe_head(concat)
            ce_probe = F.cross_entropy(logits_probe, label)
            output["losses"]["sup_probe_ce"] = ce_probe * self.sup_probe_coeff

        if self.training:
            kl_1 = self.forward_once(
                x,
                stochastic=True,
                perturb_x1=True,
                perturb_x2=False,
                **kwargs,
            )
            kl_2 = self.forward_once(
                x,
                stochastic=True,
                perturb_x1=False,
                perturb_x2=True,
                **kwargs,
            )

            infonce_loss = nt_xent_loss(
                output["features"]["z1"],
                output["features"]["z2"],
                temperature=1.0,
            )

            wandb_loss = {
                "reg_loss": {
                    "kl_1": kl_1,
                    "kl_2": kl_2,
                    "infonce": infonce_loss,
                }
            }
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            # else:
            #     wandb.log(wandb_loss)

            output["losses"].update(
                {
                    "sl_1": kl_1 * self.l,
                    "sl_2": kl_2 * self.l,
                    "infonce": infonce_loss * self.contrcoeff,
                }
            )

        else:
            out_z1 = self.forward_once(
                x,
                stochastic=False,
                perturb_x1=True,
                perturb_x2=False,
                **kwargs,
            )
            out_z2 = self.forward_once(
                x,
                stochastic=False,
                perturb_x1=False,
                perturb_x2=True,
                **kwargs,
            )
            output["features"]["~z1"] = out_z1["features"]["z1"]
            output["features"]["~z2"] = out_z2["features"]["z2"]

        return output


class FusionIBModel(nn.Module):
    def __init__(self, args, encs):
        super(FusionIBModel, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)
        self.synergy_weight = args.get("bias_infusion",{}).get("l", 0)

        # main encoders
        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        # classifier trunk
        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner),
            )

        elif self.cls_type == "conformer":
            self.common_fc_1 = TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6,
                output_dim=d_model,
            )

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner),
            )
        else:
            raise ValueError("Unknown cls_type")

        self.mu_head = nn.Linear(fc_inner, num_classes)

        self.synergy = SynIB(args, encs, main=self)

    def _get_features(self, x, **kwargs):
        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)
        return a, v

    def _compute_logits(self, a, v, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(
                torch.cat((a["features"]["combined"], v["features"]["combined"]), dim=1)
            )
            feat = self.common_fc_2(aggr_feat)

        elif self.cls_type == "conformer":
            pred_fused, aggr_feat, feat = self.common_fc_1(
                [a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]],
                return_all=True,
            )
            feat = self.common_fc_2(aggr_feat)

        pred = self.mu_head(feat)
        return pred, feat

    def _base_forward(self, x, **kwargs):
        """
        Clean deterministic pass, no perturbations / synergy.
        """
        a, v = self._get_features(x, **kwargs)
        pred, feat = self._compute_logits(a, v)

        output = {
            "preds": {
                "combined": pred,
                "p1": a["preds"]["combined"],
                "p2": v["preds"]["combined"],
            },
            "features": {
                "z1": a["features"]["combined"],
                "z2": v["features"]["combined"],
            },
            "losses": {},
        }
        return output

    def forward(self, x, **kwargs):

        output = self._base_forward(x, **kwargs)

        if self.synergy_weight > 0:
            synergy_losses = self.synergy.compute_training_losses(
                x, base_output=output, **kwargs
            )
            output["losses"].update(synergy_losses)

        return output


class SynIB(nn.Module):
    def __init__(self, args, encs, main):
        super().__init__()
        object.__setattr__(self, "main", main)
        # self.main = weakref.proxy(main)  #main
        if len(encs) == 4:
            self.gen_x1 = encs[2]
            self.gen_x2 = encs[3]

        self.perturb = args.get("perturb", {})
        self.perturb.reestimate_features = self.perturb.get("reestimate_features", False)


        bias = args.get("bias_infusion", {})
        self.synergy_weight = bias.get("l", 0.0)
        self.contrastive_weight = bias.get("contrcoeff", 0.0) or 0.0
        self.synergy_type = getattr(args, "synergy_type", "gaussian")  # "gaussian" or "dirichlet"

        fc_inner = args.fc_inner
        num_classes = args.num_classes

        if self.synergy_type == "gaussian":
            self.logvar_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = None
        elif self.synergy_type == "dirichlet":
            self.evidence_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = args.get("dirichlet_prior_conc", 1.0)
        else:
            raise ValueError(f"Unknown synergy_type: {self.synergy_type}")

    @staticmethod
    def _gaussian_kl(mu, logvar):
        return 0.5 * torch.sum(
            torch.exp(logvar) + mu**2 - 1 - logvar, dim=1
        ).mean()

    def _log(self, d, **kwargs):
        if "current_step" in kwargs:
            wandb.log(d, step=kwargs["current_step"] + 1)
        # else:
        #     wandb.log(d)

    @staticmethod
    def _dirichlet_kl(alpha, prior_conc=1.0):
        """
        KL(Dir(alpha) || Dir(alpha0)), with alpha0_k = prior_conc (scalar or tensor).
        Batch over dim=0, classes over dim=1.
        """
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

    def _perturb(self, zt, zc, direction, **kwargs):

        ptype = self.perturb.get("type", "none")

        if ptype in {"mask", "cmn"}:
            p = float(self.perturb.get("p", 0.5))
            s = float(self.perturb.get("s", 0.1))
            c = float(self.perturb.get("c", 1.0))
            K = int(self.perturb.get("num_samples", 1))
            B = zt.shape[0]

            ztK = zt.unsqueeze(0).expand(K, *zt.shape).reshape(K * B, *zt.shape[1:])
            zcK = zc.unsqueeze(0).expand(K, *zc.shape).reshape(K * B, *zc.shape[1:])

            keep = (torch.rand_like(ztK) > p).to(ztK.dtype)

            if ptype == "mask":
                fill = self.perturb.get("fill", "zeros")

                if fill == "zeros":
                    z_fill = torch.zeros_like(ztK)

                elif fill == "noise":
                    # scale: [K*B, 1] or [K*B, S, 1] -> broadcasts over feature dim
                    scale = s * (1.0 + torch.tanh(zcK.norm(dim=-1, keepdim=True) / c))
                    z_fill = torch.randn_like(ztK) * scale

                elif fill == "mean":
                    # per-position mean: [1, F] or [1, S, F] -> broadcasts cleanly
                    mean = zt.mean(dim=0, keepdim=True)
                    z_fill = mean.expand_as(zt).unsqueeze(0).expand(K, B, *zt.shape[1:]).reshape_as(ztK)

                else:
                    raise ValueError(f"Unknown mask fill: {fill}")

                tilde = keep * ztK + (1.0 - keep) * z_fill
                return {"tilde": tilde, "mask": keep, "z_c": zcK}

            # ---- cmn ----
            scale = s * (1.0 + torch.tanh(zcK.norm(dim=-1, keepdim=True) / c))
            noise = torch.randn_like(ztK) * scale
            tilde = keep * ztK + (1.0 - keep) * noise
            return {"tilde": tilde, "mask": keep, "z_c": zcK}

        # --------------------------
        # 3) Generator-based perturbations
        # --------------------------
        gen = self.gen_x1 if direction == "x1" else self.gen_x2
        key = "px1" if direction == "x1" else "px2"

        gkw = {
            "compute_loss": self.training,
            "add_noise": (ptype == "gennoise"),
        }
        if "current_step" in kwargs:
            gkw["current_step"] = kwargs["current_step"]

        out = gen(zt, zc, **gkw)

        if isinstance(out, dict) and "losses" in out:
            self._log({key: out["losses"]}, **kwargs)

        return out if isinstance(out, dict) else {"tilde": out}

    def _encode_and_perturb(self, x, z1, z2, px1, px2, **kwargs):
        self.main.eval()
        if self.perturb.reestimate_features:
            a, v = self.main._get_features(x, **kwargs)
            z1, z2 = a["features"]["combined"], v["features"]["combined"]

        losses = {}
        if px1:
            out = self._perturb(z1, z2, "x1", **kwargs)
            z1, z2 = out["tilde"], out["z_c"]
            if "losses" in out: losses["px1"] = out["losses"]
        if px2:
            out = self._perturb(z2, z1, "x2", **kwargs)
            z2, z1 = out["tilde"], out["z_c"]
            if "losses" in out: losses["px2"] = out["losses"]

        self.main.train()

        return z1, z2, losses

    # ------------------ KL passes ------------------

    def _kl_pass(self, x, z1, z2, px1, px2, **kwargs):
        a, v, losses = self._encode_and_perturb(x, z1, z2, px1, px2, **kwargs)
        mu, feat = self.main._compute_logits({"features":{"combined":a}}, {"features":{"combined":v}})
        if self.synergy_type == "gaussian":
            logvar = self.logvar_head(feat)
            kl = self._gaussian_kl(mu, logvar)
        else:  # dirichlet
            evidence = F.softplus(self.evidence_head(feat))
            alpha = evidence + 1.0
            kl = self._dirichlet_kl(alpha, prior_conc=self.dirichlet_prior_conc)
        return kl, losses


    def compute_training_losses(self, x, base_output, **kwargs):
        z1, z2 = base_output["features"]["z1"], base_output["features"]["z2"]
        kl1, loss1 = self._kl_pass(x, z1, z2, px1=True,  px2=False, **kwargs)
        kl2, loss2 = self._kl_pass(x, z1, z2, px1=False, px2=True,  **kwargs)

        infonce = nt_xent_loss( z1, z2, temperature=1.0)
        kl_diff_mse = torch.mean((kl1 - kl2) ** 2)

        if self.training:
            self._log({"reg_loss": {"kl_1": kl1, "kl_2": kl2, "kl_diff_mse": kl_diff_mse, "infonce": infonce}}, **kwargs)

        losses = {}
        losses.update(loss1)
        losses.update(loss2)
        losses["sl_1"] = kl1 * self.synergy_weight
        losses["sl_2"] = kl2 * self.synergy_weight
        # losses["sl_diff"] = kl_diff_mse * self.synergy_weight
        losses["infonce"] = infonce * self.contrastive_weight
        return losses

class LinearHead_Qwen(nn.Module):
    def __init__(self, args, encs=[], **kwargs):
        super().__init__()
        self.args = args
        self.num_classes = getattr(args, "num_classes")
        self.hidden_size = getattr(args, "hidden_size", 2048)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, **kwargs):
        return self.linear(x)

class QwenVL_ScienceQA_Synergy_Frozen(nn.Module):

    def __init__(self, args, encs=[], **kwargs):
        super().__init__()

        self.args = args
        self.synergy_coeff = getattr(args, "synergy_coeff", 0.0)
        self.max_new_tokens = getattr(args, "max_new_tokens", 32)
        self.num_classes = getattr(args, "num_classes")

        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        HF_CACHE = self.args.save_base_dir

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=HF_CACHE)
        if hasattr(self.processor, "tokenizer"):
            tok = self.processor.tokenizer
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
        else:
            tok = self.processor.tokenizer

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
            device_map="cuda:0",
            cache_dir=HF_CACHE,
        )

        cfg = self.backbone.config

        self.image_token_id = cfg.image_token_id
        self.vision_start_token_id = getattr(cfg, "vision_start_token_id", None)
        self.vision_end_token_id = getattr(cfg, "vision_end_token_id", None)
        self.image_token_str = tok.convert_ids_to_tokens(self.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            d_model = cfg.text_config.hidden_size
        else:
            d_model = cfg.hidden_size
        self.d_model = d_model

        self.enc_0 = encs[0]

        self._joint_unfrozen = False
        self._set_joint_training_for_epoch(0)

        # proj_dim = getattr(args, "proj_dim", d_model)
        # self.text_proj = nn.Linear(d_model, proj_dim)
        # self.image_proj = nn.Linear(d_model, proj_dim)

        # if not hasattr(args, "fc_inner"):
        #     setattr(args, "fc_inner", d_model)
        #
        # self.synib = SynIB(args, encs, main=self)

    def _set_joint_training(self, train_last_n_layers: int = 0):

        for p in self.backbone.parameters():
            p.requires_grad = False

        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "visual"):
            for p in self.backbone.model.visual.parameters():
                p.requires_grad = False

        lm = None
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "language_model"):
            lm = self.backbone.model.language_model

        if lm is not None:
            if hasattr(lm, "embed_tokens"):
                for p in lm.embed_tokens.parameters():
                    p.requires_grad = False
            if hasattr(lm, "norm"):
                for p in lm.norm.parameters():
                    p.requires_grad = False

            if hasattr(lm, "layers"):
                layers = lm.layers
                for layer in layers:
                    for p in layer.parameters():
                        p.requires_grad = False

                if train_last_n_layers > 0:
                    for layer in layers[-train_last_n_layers:]:
                        for p in layer.parameters():
                            p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        if hasattr(self, "text_proj"):
            for p in self.text_proj.parameters():
                p.requires_grad = True
        if hasattr(self, "image_proj"):
            for p in self.image_proj.parameters():
                p.requires_grad = True

        # for p in self.synib.parameters():
        #     p.requires_grad = (self.synergy_coeff > 0.0)

        any_backbone_trainable = any(
            p.requires_grad for p in self.backbone.parameters()
        )
        self.backbone_trainable = any_backbone_trainable


    def _set_joint_training_for_epoch(self, epoch: int):
        """
        Epoch 0: backbone fully frozen.
        First time epoch >= 1: unfreeze last 5 text layers as joint model.
        After that, do nothing (keep that setting forever).
        """
        if epoch <= 0 and not self._joint_unfrozen:
            # still in warmup: keep backbone frozen
            self._set_joint_training(train_last_n_layers=0)
        elif epoch >= 1 and not self._joint_unfrozen:
            # first time we see epoch >= 1 → switch to 5 trainable layers
            self._set_joint_training(train_last_n_layers=5)
            self._joint_unfrozen = True
        # if self._joint_unfrozen is already True, do nothing

    # -----------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------
    def _encode(self, input_ids, attention_mask, pixel_values, image_grid_thw=None):
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
        return hidden  # (B, T, d)

    def _get_cls_repr(self, hidden, attention_mask):
        B, T, _ = hidden.size()
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(B, 1, 1).expand(-1, 1, hidden.size(-1))
        h_cls = hidden.gather(1, idx).squeeze(1)  # (B, d)
        h_cls = F.layer_norm(h_cls, (h_cls.shape[-1],))

        # mask = attention_mask.unsqueeze(-1).to(dtype=torch.float32)  # (B,T,1)
        # hidden_f = hidden.to(torch.float32)  # (B,T,d)
        # den = mask.sum(dim=1).clamp(min=1.0)  # (B,1)
        # num = (hidden_f * mask).sum(dim=1)  # (B,d) fp32 sum
        # h_mean = num / den  # (B,d) fp32
        # h_cls = F.layer_norm(h_mean, (h_mean.shape[-1],))

        return h_cls

    def _get_image_text_repr(self, hidden, image_token_mask, text_token_mask):
        img_mask = image_token_mask.float()
        txt_mask = text_token_mask.float()

        img_mask = img_mask / (img_mask.sum(dim=1, keepdim=True) + 1e-8)
        txt_mask = txt_mask / (txt_mask.sum(dim=1, keepdim=True) + 1e-8)

        h_img = torch.bmm(img_mask.unsqueeze(1), hidden).squeeze(1)
        h_txt = torch.bmm(txt_mask.unsqueeze(1), hidden).squeeze(1)
        return h_img, h_txt

    def _build_prompts_with_choices(self, hint_texts, qa_texts, letters_list):
        """
        hint_texts: list[str] (lecture+hint)
        qa_texts:   list[str] (question + lettered choices)
        letters_list: list[list[str]] (letters used for that question)
        """
        prompts = []
        for hint, qa, letters in zip(hint_texts, qa_texts, letters_list):
            parts = []
            if hint is not None and hint.strip():
                parts.append(hint.strip())
            if qa is not None and qa.strip():
                parts.append(qa.strip())

            if letters:
                letters_str = ", ".join(f"({L})" for L in letters)
                parts.append("")
                parts.append(f"Answer with only one of: {letters_str}.")

            prompts.append("\n\n".join(parts))
        return prompts

    def _generate_raw_answers(self, proc, input_ids, *, letters_list):
        gen_inputs = {
            k: v
            for k, v in proc.items()
            if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        }
        gen_inputs = {k: v.to(self.backbone.device) for k, v in gen_inputs.items()}

        with torch.no_grad():
            gen_ids = self.backbone.generate(
                **gen_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        gen_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(input_ids, gen_ids)
        ]

        raw_answers = self.processor.batch_decode(
            gen_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        import re

        def clean_answer(ans: str):
            lines = [l.strip() for l in ans.splitlines() if l.strip()]
            if not lines:
                return ans.strip()
            first = lines[0]

            m = re.search(r"\(([A-Za-z])\)", first)
            if m:
                return f"({m.group(1).upper()})"

            m2 = re.search(r"\b([A-Za-z])\b", first)
            if m2:
                return f"({m2.group(1).upper()})"

            return first

        cleaned = [clean_answer(ans) for ans in raw_answers]

        pred_indices = []
        for ans, letters in zip(cleaned, letters_list):
            if not letters:
                pred_indices.append(-1)
                continue
            letters_upper = [L.upper() for L in letters]

            m = re.search(r"\(([A-Za-z])\)", ans)
            if not m:
                pred_indices.append(-1)
                continue
            letter = m.group(1).upper()

            if letter in letters_upper:
                idx = letters_upper.index(letter)
                pred_indices.append(idx)
            else:
                pred_indices.append(-1)

        pred_indices = torch.tensor(
            pred_indices, device=input_ids.device, dtype=torch.long
        )
        return cleaned, pred_indices

    def _mc_ce_loss(self, logits, labels):
        if "class_weights" in self.args:
            class_weights = self.args.class_weights.to(logits.device)
            return F.cross_entropy(logits, labels, weight=class_weights)
        return F.cross_entropy(logits, labels)

    def _set_train_epoch(self, epoch: int):
        """Call this at the start of each epoch."""
        if epoch == 0:
            self._set_joint_training(0)
        else:
            self._set_joint_training(5)

    def forward(
        self,
        x,
        *,
        label=None,
        return_features=False,
        current_step=None,
        image_token_mask=None,
        text_token_mask=None,
        **kwargs,
    ):
        # if self.training and "current_epoch" in kwargs:
        #     self._set_joint_training_for_epoch(kwargs["current_epoch"])
            # self._set_train_epoch(kwargs["current_epoch"])

        hint_texts = x[0]
        qa_texts = x[1]
        images = x[2]
        choices_list = x[3] if len(x) > 3 else kwargs.get("choices", None)
        letters_list = x[4] if len(x) > 4 else kwargs.get("letters", None)

        if choices_list is None:
            raise ValueError("choices_list (x[3] or kwargs['choices']) is required for MC setup.")
        if letters_list is None:
            raise ValueError("letters_list (x[4] or kwargs['letters']) is required for zero-shot parsing.")

        device = images.device


        prompts = self._build_prompts_with_choices(hint_texts, qa_texts, letters_list)
        prompts_with_image = [self.image_token_str + "\n" + p for p in prompts]
        image_list = [img for img in images]

        proc_kwargs = dict(
            text=prompts_with_image,
            images=image_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        proc = self.processor(**proc_kwargs)
        proc = {k: v.to(device) for k, v in proc.items()}

        input_ids = proc["input_ids"]
        attention_mask = proc["attention_mask"]
        pixel_values = proc["pixel_values"]
        image_grid_thw = proc.get("image_grid_thw")

        hidden = self._encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        h_cls = self._get_cls_repr(hidden, attention_mask)
        h_cls = h_cls.to(self.enc_0.linear.weight.dtype)
        head_logits = self.enc_0(h_cls)

        if not self.training:
            raw_text_answers, mc_from_text = self._generate_raw_answers(
                proc,
                input_ids,
                letters_list=letters_list,
            )

            B = head_logits.size(0)
            zeroshot_probs = torch.zeros(
                B, self.num_classes, device=device, dtype=head_logits.dtype
            )
            valid = mc_from_text >= 0
            if valid.any():
                zeroshot_probs[valid] = F.one_hot(
                    mc_from_text[valid].clamp(min=0), num_classes=self.num_classes
                ).to(head_logits.dtype)

        losses = {}
        if label is not None:
            ce_head = self._mc_ce_loss(head_logits, label)
            losses["ce_head"] = ce_head

        if self.synergy_coeff > 0.0:
            if image_token_mask is None or text_token_mask is None:
                raise ValueError("image_token_mask and text_token_mask are required for SynIB synergy.")

            image_token_mask = image_token_mask.to(hidden.device)
            text_token_mask = text_token_mask.to(hidden.device)

            h_img, h_txt = self._get_image_text_repr(
                hidden, image_token_mask, text_token_mask
            )
            h_img = h_img.to(self.image_proj.weight.dtype)
            h_txt = h_txt.to(self.text_proj.weight.dtype)

            z_img = self.image_proj(h_img)
            z_txt = self.text_proj(h_txt)

            synib_losses = self.synib.compute_from_reps(
                mu=head_logits,
                feat=h_cls,
                z1=z_img,
                z2=z_txt,
                current_step=current_step,
            )

            for k, v in synib_losses.items():
                losses[f"synib_{k}"] = self.synergy_coeff * v

        features = {"h_cls": h_cls}
        if return_features:
            features["hidden"] = hidden

        return {
            "preds": {
                "combined": head_logits,
                # "g": zeroshot_probs,
                # "c": mc_from_text,
            },
            "features": features,
            # "raw_text": raw_text_answers,
            "losses": losses,
        }


class QwenVL_ScienceQA_Synergy_FrozenCLS(nn.Module):
    """
    Multimodal (image+text) ScienceQA as 5-way classification.
    Backbone is frozen EXCEPT:
      - classifier head enc_0 (always trainable)
      - (optional) learnable <CLS> embedding row ONLY (via gradient masking hook)
      - (optional) final LM norm (cheap, sometimes helps)

    Readout is the hidden state at the appended <CLS> token (placed at end of prompt).
    """

    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []

        self.args = args
        self.synergy_coeff = getattr(args, "synergy_coeff", 0.0)
        self.max_new_tokens = getattr(args, "max_new_tokens", 32)
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        HF_CACHE = getattr(self.args, "save_base_dir", None)

        # -----------------------------
        # Processor / Tokenizer
        # -----------------------------
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=HF_CACHE)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Add <CLS> token to tokenizer
        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
            device_map="cuda:0",
            cache_dir=HF_CACHE,
        )

        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = cfg.image_token_id
        self.image_token_str = tok.convert_ids_to_tokens(self.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = cfg.text_config.hidden_size
        else:
            self.d_model = cfg.hidden_size

        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the 5-way classifier head.")
        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

    def _setup_trainables(self):
        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model

        # Optionally train final norm (cheap and often stabilizes)
        # if getattr(self.args, "train_lm_norm", False) and lm is not None and hasattr(lm, "norm"):
        #     for p in lm.norm.parameters():
        #         p.requires_grad = True

        # Make <CLS> embedding learnable WITHOUT unfreezing whole embedding table
        # (default True; set args.train_cls_row=False to disable)
        if self.args.cls_finetune:
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                # ensure grads flow to emb.weight (we'll mask them)
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                # build a (vocab, hidden) mask with 1s only for cls row
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                # register once
                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

        # NOTE: if you enabled synergy modules, mark them trainable here.

    def _load_cls_embedding(self):

        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)

        self.load_cls_embedding(cls_path)

    def load_cls_embedding(self, path, strict_dim=True):

        assert os.path.isfile(path), f"CLS embedding file not found: {path}"

        ckpt = torch.load(path, map_location="cpu")

        if "cls_row" not in ckpt:
            raise KeyError("CLS checkpoint must contain 'cls_row'")

        cls_row = ckpt["cls_row"]
        saved_cls_id = ckpt.get("cls_token_id", self.cls_token_id)

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        current_cls_id = int(self.cls_token_id)

        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(
                f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}"
            )

        if saved_cls_id != current_cls_id:
            print(
                f"[WARN] saved cls_token_id={saved_cls_id} "
                f"!= current cls_token_id={current_cls_id} — copying to current index"
            )

        with torch.no_grad():
            emb.weight[current_cls_id].copy_(
                cls_row.to(emb.weight.device, emb.weight.dtype)
            )

        print(f"[OK] Loaded CLS embedding from {path}")

    def _apply_lora(self):
        cfg = getattr(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
        )

        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _build_prompts_with_choices(self, hint_texts, qa_texts, letters_list):
        prompts = []
        for hint, qa, letters in zip(hint_texts, qa_texts, letters_list):
            parts = []
            if hint is not None and hint.strip():
                parts.append(hint.strip())
            if qa is not None and qa.strip():
                parts.append(qa.strip())

            if letters:
                letters_str = ", ".join(f"({L})" for L in letters)
                parts.append(f"Answer with only one of: {letters_str}.")

            # Put CLS token at the END so it can attend to all previous tokens (causal LM)
            parts.append("<CLS>")

            prompts.append("\n\n".join(parts))
        return prompts

    # ============================================================
    #  Encoding / readout
    # ============================================================
    def _encode(self, input_ids, attention_mask, pixel_values, image_grid_thw=None):
        # IMPORTANT: no torch.no_grad() here; we need grads at least to CLS row + head
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1]  # (B, T, d)

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        device = input_ids.device

        # position of <CLS> (assumes exactly once per sample)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)  # (B,)
        h = hidden[torch.arange(B, device=device), cls_pos]             # (B,d)
        h = F.layer_norm(h, (h.shape[-1],))
        return h

    # ============================================================
    #  (Optional) generation for eval-time parsing (unchanged)
    # ============================================================
    def _generate_raw_answers(self, proc, input_ids, *, letters_list):
        gen_inputs = {
            k: v for k, v in proc.items()
            if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        }
        gen_inputs = {k: v.to(self.backbone.device) for k, v in gen_inputs.items()}

        with torch.no_grad():
            gen_ids = self.backbone.generate(
                **gen_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, gen_ids)]
        raw_answers = self.processor.batch_decode(
            gen_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        import re

        def clean_answer(ans: str):
            lines = [l.strip() for l in ans.splitlines() if l.strip()]
            if not lines:
                return ans.strip()
            first = lines[0]
            m = re.search(r"\(([A-Za-z])\)", first)
            if m:
                return f"({m.group(1).upper()})"
            m2 = re.search(r"\b([A-Za-z])\b", first)
            if m2:
                return f"({m2.group(1).upper()})"
            return first

        cleaned = [clean_answer(ans) for ans in raw_answers]

        pred_indices = []
        for ans, letters in zip(cleaned, letters_list):
            if not letters:
                pred_indices.append(-1)
                continue
            letters_upper = [L.upper() for L in letters]
            m = re.search(r"\(([A-Za-z])\)", ans)
            if not m:
                pred_indices.append(-1)
                continue
            letter = m.group(1).upper()
            pred_indices.append(letters_upper.index(letter) if letter in letters_upper else -1)

        pred_indices = torch.tensor(pred_indices, device=input_ids.device, dtype=torch.long)
        return cleaned, pred_indices

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            class_weights = self.args.class_weights.to(logits.device)
            return F.cross_entropy(logits, labels, weight=class_weights)
        return F.cross_entropy(logits, labels)

    # ============================================================
    #  Forward
    # ============================================================
    def forward(
        self,
        x,
        *,
        label=None,
        return_features=False,
        current_step=None,
        image_token_mask=None,  # unused here (CLS readout); keep for compatibility
        text_token_mask=None,   # unused here (CLS readout); keep for compatibility
        **kwargs,
    ):
        hint_texts = x[0]
        qa_texts = x[1]
        images = x[2]
        choices_list = x[3] if len(x) > 3 else kwargs.get("choices", None)
        letters_list = x[4] if len(x) > 4 else kwargs.get("letters", None)

        if choices_list is None:
            raise ValueError("choices_list (x[3] or kwargs['choices']) is required for MC setup.")
        if letters_list is None:
            raise ValueError("letters_list (x[4] or kwargs['letters']) is required for zero-shot parsing.")

        device = images.device

        prompts = self._build_prompts_with_choices(hint_texts, qa_texts, letters_list)
        prompts_with_image = [self.image_token_str + "\n" + p for p in prompts]
        image_list = [img for img in images]

        proc = self.processor(
            text=prompts_with_image,
            images=image_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        proc = {k: v.to(device) for k, v in proc.items()}

        input_ids = proc["input_ids"]
        attention_mask = proc["attention_mask"]
        pixel_values = proc["pixel_values"]
        image_grid_thw = proc.get("image_grid_thw")

        hidden = self._encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # CLS readout (stable position)
        h_cls = self._get_cls_token_repr(hidden, input_ids)
        h_cls = h_cls.to(self.enc_0.linear.weight.dtype)

        head_logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_head"] = self._mc_ce_loss(head_logits, label)

        # Optional eval-time generation parsing (kept off by default)
        preds = {"combined": head_logits}
        features = {"h_cls": h_cls}
        if return_features:
            features["hidden"] = hidden

        # If you want zero-shot text parsing at eval:
        if (not self.training) and getattr(self.args, "do_zeroshot_parse", False):
            raw_text_answers, mc_from_text = self._generate_raw_answers(
                proc, input_ids, letters_list=letters_list
            )
            preds["raw_text"] = raw_text_answers
            preds["mc_from_text"] = mc_from_text

        return {"preds": preds, "features": features, "losses": losses}

class SynIB_Qwen(nn.Module):
    def __init__(self, args, encs, main):
        super().__init__()
        object.__setattr__(self, "main", main)

        self.perturb = args.get("perturb", {})

        bias = args.get("bias_infusion", {})
        self.synergy_weight = bias.get("l", 0.0)
        self.synergy_type = getattr(args, "synergy_type", "gaussian")  # "gaussian" or "dirichlet"

        fc_inner = 2048
        num_classes = args.num_classes

        if self.synergy_type == "gaussian":
            self.logvar_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = None
        elif self.synergy_type == "dirichlet":
            self.evidence_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = args.get("dirichlet_prior_conc", 1.0)
        else:
            raise ValueError(f"Unknown synergy_type: {self.synergy_type}")

    @staticmethod
    def _gaussian_kl(mu, logvar):
        return 0.5 * torch.sum(
            torch.exp(logvar) + mu**2 - 1 - logvar, dim=1
        ).mean()

    def _log(self, d, **kwargs):
        if "current_step" in kwargs:
            wandb.log(d, step=kwargs["current_step"] + 1)
        # else:
        #     wandb.log(d)

    @staticmethod
    def _dirichlet_kl(alpha, prior_conc=1.0):
        """
        KL(Dir(alpha) || Dir(alpha0)), with alpha0_k = prior_conc (scalar or tensor).
        Batch over dim=0, classes over dim=1.
        """
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

    def _random_masks(self, m1, m2, px1, px2, **kwargs):
        """
        Mask-only perturbation:
          - sample keep mask with prob (1-p) of keeping each entry
          - masked entries are replaced with EMA values (per token, per feature)
        Shapes:
          zt, zc: [B, T, F]
        Returns:
          tilde: [K*B, T, F]
          mask:  [K*B, T, F] (1=kept, 0=masked)
          z_c:   [K*B, T, F]
        """
        # --- config ---
        p = float(self.perturb.get("p", 0.5))  # mask probability
        m1_t, m2_t = None, None
        if px1:
            mask_1 = (torch.rand_like(m1[m1==True].float()) > p).to(dtype=m1.dtype)  # [K*B, T, F] in {0,1}
            m1_t = copy.deepcopy(m1)
            m1_t[m1] = mask_1
        if px2:
            mask_2 = (torch.rand_like(m2[m2==True].float()) > p).to(dtype=m2.dtype)  # [K*B, T, F] in {0,1}
            m2_t = copy.deepcopy(m2)
            m2_t[m2] = mask_2

        return m1_t, m2_t

    def _kl_pass(self, x, m1, m2, px1, px2, **kwargs):
        m1t, m2t = self._random_masks(m1, m2, px1, px2, **kwargs)
        mu, feat, _ = self.main._compute_logits(x, m1t, m2t)
        if self.synergy_type == "gaussian":
            logvar = self.logvar_head(feat["h_cls"])
            kl = self._gaussian_kl(mu, logvar)
        else:  # dirichlet
            evidence = F.softplus(self.evidence_head(feat["h_cls"]))
            alpha = evidence + 1.0
            kl = self._dirichlet_kl(alpha, prior_conc=self.dirichlet_prior_conc)
        return kl


    def compute_training_losses(self, x, base_output, **kwargs):
        m1, m2 = base_output["masks"]["hint_mask"], base_output["masks"]["image_mask"]
        kl1 = self._kl_pass(x, m1, m2, px1=True,  px2=False, **kwargs)
        kl2 = self._kl_pass(x, m1, m2, px1=False, px2=True,  **kwargs)
        kl_diff_mse = torch.mean((kl1 - kl2) ** 2)

        if self.training:
            self._log({"reg_loss": {"kl_1": kl1, "kl_2": kl2, "kl_diff_mse": kl_diff_mse}}, **kwargs)

        losses = {"sl_1": kl1 * self.synergy_weight, "sl_2": kl2 * self.synergy_weight}
        # losses["sl_diff"] = kl_diff_mse * self.synergy_weight
        return losses

# class QwenVL_ScienceQA_Synergy_SynIB(nn.Module):
#     """
#     Multimodal (image+text) ScienceQA as 5-way classification.
#     Backbone is frozen EXCEPT:
#       - classifier head enc_0 (always trainable)
#       - (optional) learnable <CLS> embedding row ONLY (via gradient masking hook)
#       - (optional) final LM norm (cheap, sometimes helps)
#
#     Readout is the hidden state at the appended <CLS> token (placed at end of prompt).
#     """
#
#     def __init__(self, args, encs=None, **kwargs):
#         super().__init__()
#         encs = encs or []
#
#         self.args = args
#         self.synergy_coeff =  args.get("bias_infusion",{}).get("l", 0)
#         self.max_new_tokens = getattr(args, "max_new_tokens", 32)
#         self.num_classes = getattr(args, "num_classes")
#
#         model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
#         HF_CACHE = getattr(self.args, "save_base_dir", None)
#
#         # -----------------------------
#         # Processor / Tokenizer
#         # -----------------------------
#         self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=HF_CACHE)
#         tok = self.processor.tokenizer
#         tok.padding_side = "left"
#         if tok.pad_token is None:
#             tok.pad_token = tok.eos_token
#
#         # Add <CLS> token to tokenizer
#         added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
#         self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")
#
#         self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
#             model_name,
#             dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
#             device_map="cuda:0",
#             cache_dir=HF_CACHE,
#         )
#
#         if added > 0:
#             self.backbone.resize_token_embeddings(len(tok))
#
#         cfg = self.backbone.config
#         self.image_token_id = cfg.image_token_id
#         self.image_token_str = tok.convert_ids_to_tokens(self.image_token_id)
#
#         if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
#             self.d_model = cfg.text_config.hidden_size
#         else:
#             self.d_model = cfg.hidden_size
#
#         if len(encs) < 1:
#             raise ValueError("encs[0] must be provided as the 5-way classifier head.")
#         self.enc_0 = encs[0]
#
#         self._apply_lora()
#         self._load_cls_embedding()
#         self._setup_trainables()
#
#         self.synib = SynIB_Qwen(args, [], self)
#
#     def _setup_trainables(self):
#         # Freeze everything
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#
#         if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
#             for n, p in self.backbone.named_parameters():
#                 if "lora_" in n:
#                     p.requires_grad = True
#
#         for p in self.enc_0.parameters():
#             p.requires_grad = True
#
#         lm = self.backbone.model.language_model
#
#         # Optionally train final norm (cheap and often stabilizes)
#         # if getattr(self.args, "train_lm_norm", False) and lm is not None and hasattr(lm, "norm"):
#         #     for p in lm.norm.parameters():
#         #         p.requires_grad = True
#
#         # Make <CLS> embedding learnable WITHOUT unfreezing whole embedding table
#         # (default True; set args.train_cls_row=False to disable)
#         if self.args.cls_finetune:
#             if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
#                 emb = lm.embed_tokens
#                 # ensure grads flow to emb.weight (we'll mask them)
#                 emb.weight.requires_grad = True
#
#                 cls_id = int(self.cls_token_id)
#                 # build a (vocab, hidden) mask with 1s only for cls row
#                 mask = torch.zeros_like(emb.weight, dtype=torch.float32)
#                 mask[cls_id].fill_(1.0)
#
#                 def grad_mask_hook(grad):
#                     return grad * mask.to(grad.device, grad.dtype)
#
#                 # register once
#                 if not hasattr(self, "_cls_grad_hooked"):
#                     emb.weight.register_hook(grad_mask_hook)
#                     self._cls_grad_hooked = True
#
#         # NOTE: if you enabled synergy modules, mark them trainable here.
#
#     def _load_cls_embedding(self):
#
#         cls_path = getattr(self.args, "cls_emb_path", None)
#         save_base_dir = getattr(self.args, "save_base_dir", None)
#         if save_base_dir is None or cls_path is None:
#             return
#         cls_path = os.path.join(save_base_dir, cls_path)
#
#         self.load_cls_embedding(cls_path)
#
#     def load_cls_embedding(self, path, strict_dim=True):
#
#         assert os.path.isfile(path), f"CLS embedding file not found: {path}"
#
#         ckpt = torch.load(path, map_location="cpu")
#
#         if "cls_row" not in ckpt:
#             raise KeyError("CLS checkpoint must contain 'cls_row'")
#
#         cls_row = ckpt["cls_row"]
#         saved_cls_id = ckpt.get("cls_token_id", self.cls_token_id)
#
#         lm = self.backbone.model.language_model
#         if lm is None or not hasattr(lm, "embed_tokens"):
#             raise RuntimeError("Language model embedding table not found")
#
#         emb = lm.embed_tokens
#         current_cls_id = int(self.cls_token_id)
#
#         if strict_dim and cls_row.numel() != emb.weight.shape[1]:
#             raise ValueError(
#                 f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}"
#             )
#
#         if saved_cls_id != current_cls_id:
#             print(
#                 f"[WARN] saved cls_token_id={saved_cls_id} "
#                 f"!= current cls_token_id={current_cls_id} — copying to current index"
#             )
#
#         with torch.no_grad():
#             emb.weight[current_cls_id].copy_(
#                 cls_row.to(emb.weight.device, emb.weight.dtype)
#             )
#
#         print(f"[OK] Loaded CLS embedding from {path}")
#
#     def _apply_lora(self):
#         cfg = getattr(self.args, "lora_config", None)
#         if not cfg or not cfg.get("use_lora", False):
#             return
#
#         lora_cfg = LoraConfig(
#             r=int(cfg.get("lora_r", 8)),
#             lora_alpha=int(cfg.get("lora_alpha", 8)),
#             lora_dropout=float(cfg.get("lora_dropout", 0.0)),
#             target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
#             bias=str(cfg.get("lora_bias", "none")),
#             task_type="CAUSAL_LM",
#         )
#
#         self.backbone = get_peft_model(self.backbone, lora_cfg)
#
#     def _build_prompts_with_choices(self, hint_texts, qa_texts, letters_list):
#         prompts = []
#         for hint, qa, letters in zip(hint_texts, qa_texts, letters_list):
#             parts = []
#             if hint is not None and hint.strip():
#                 parts.append(hint.strip())
#             if qa is not None and qa.strip():
#                 parts.append(qa.strip())
#
#             if letters:
#                 letters_str = ", ".join(f"({L})" for L in letters)
#                 parts.append(f"Answer with only one of: {letters_str}.")
#
#             # Put CLS token at the END so it can attend to all previous tokens (causal LM)
#             parts.append("<CLS>")
#
#             prompts.append("\n\n".join(parts))
#         return prompts
#
#     # ============================================================
#     #  Encoding / readout
#     # ============================================================
#     def _encode(self, input_ids, attention_mask, pixel_values, image_grid_thw=None):
#         # IMPORTANT: no torch.no_grad() here; we need grads at least to CLS row + head
#         outputs = self.backbone(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             pixel_values=pixel_values,
#             image_grid_thw=image_grid_thw,
#             output_hidden_states=True,
#         )
#         return outputs.hidden_states[-1]  # (B, T, d)
#
#     def _get_cls_token_repr(self, hidden, input_ids):
#         B = input_ids.size(0)
#         device = input_ids.device
#
#         # position of <CLS> (assumes exactly once per sample)
#         cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)  # (B,)
#         h = hidden[torch.arange(B, device=device), cls_pos]             # (B,d)
#         h = F.layer_norm(h, (h.shape[-1],))
#         return h
#
#     # ============================================================
#     #  (Optional) generation for eval-time parsing (unchanged)
#     # ============================================================
#     def _generate_raw_answers(self, proc, input_ids, *, letters_list):
#         gen_inputs = {
#             k: v for k, v in proc.items()
#             if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
#         }
#         gen_inputs = {k: v.to(self.backbone.device) for k, v in gen_inputs.items()}
#
#         with torch.no_grad():
#             gen_ids = self.backbone.generate(
#                 **gen_inputs,
#                 max_new_tokens=self.max_new_tokens,
#                 do_sample=False,
#             )
#
#         gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, gen_ids)]
#         raw_answers = self.processor.batch_decode(
#             gen_ids_trimmed,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=True,
#         )
#
#         import re
#
#         def clean_answer(ans: str):
#             lines = [l.strip() for l in ans.splitlines() if l.strip()]
#             if not lines:
#                 return ans.strip()
#             first = lines[0]
#             m = re.search(r"\(([A-Za-z])\)", first)
#             if m:
#                 return f"({m.group(1).upper()})"
#             m2 = re.search(r"\b([A-Za-z])\b", first)
#             if m2:
#                 return f"({m2.group(1).upper()})"
#             return first
#
#         cleaned = [clean_answer(ans) for ans in raw_answers]
#
#         pred_indices = []
#         for ans, letters in zip(cleaned, letters_list):
#             if not letters:
#                 pred_indices.append(-1)
#                 continue
#             letters_upper = [L.upper() for L in letters]
#             m = re.search(r"\(([A-Za-z])\)", ans)
#             if not m:
#                 pred_indices.append(-1)
#                 continue
#             letter = m.group(1).upper()
#             pred_indices.append(letters_upper.index(letter) if letter in letters_upper else -1)
#
#         pred_indices = torch.tensor(pred_indices, device=input_ids.device, dtype=torch.long)
#         return cleaned, pred_indices
#
#     def _mc_ce_loss(self, logits, labels):
#         if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
#             class_weights = self.args.class_weights.to(logits.device)
#             return F.cross_entropy(logits, labels, weight=class_weights)
#         return F.cross_entropy(logits, labels)
#
#
#     def _prepare_modality_tokens(self, x, **kwargs):
#         """
#         Returns:
#           z1, z2, z3: dicts with (ids, mask) for each modality region in the *full* sequence
#           meta: extra tensors/fields needed downstream (proc fields, letters_list, etc.)
#         """
#         hint_texts = x[0]
#         qa_texts = x[1]
#         images = x[2]
#         choices_list = x[3] if len(x) > 3 else kwargs.get("choices", None)
#         letters_list = x[4] if len(x) > 4 else kwargs.get("letters", None)
#
#         if choices_list is None:
#             raise ValueError("choices_list (x[3] or kwargs['choices']) is required for MC setup.")
#         if letters_list is None:
#             raise ValueError("letters_list (x[4] or kwargs['letters']) is required for zero-shot parsing.")
#
#         device = images.device
#
#         # Build two text regions per sample
#         hint_parts, qa_parts = self._build_hint_and_qa_parts(
#             hint_texts, qa_texts, letters_list, choices_list=choices_list
#         )
#
#         # Full prompt that goes to the multimodal processor (same as your original)
#         # You control separators; must match the way you compute spans below.
#         full_texts = [self.image_token_str + "\n" + h + "\n" + q for h, q in zip(hint_parts, qa_parts)]
#         image_list = [img for img in images]
#
#         proc = self.processor(
#             text=full_texts,
#             images=image_list,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#         )
#         proc = {k: v.to(device) for k, v in proc.items()}
#
#         input_ids = proc["input_ids"]  # [B, L]
#         attention_mask = proc["attention_mask"]  # [B, L]
#
#         # ---- Identify image placeholder token positions (z2) ----
#         tok = self.processor.tokenizer
#         image_token_id = tok.convert_tokens_to_ids(self.image_token_str)
#         if image_token_id is None or image_token_id == tok.unk_token_id:
#             raise ValueError(
#                 f"Could not resolve image token id for {self.image_token_str}. "
#                 "Check your tokenizer special tokens."
#             )
#
#         z2_mask = (input_ids == image_token_id) & attention_mask.bool()  # [B, L]
#
#         # ---- Compute hint (z1) and qa (z3) spans as masks over the *full* sequence ----
#         # We estimate lengths by tokenizing hint_parts and qa_parts separately.
#         # We do NOT add special tokens so we get raw token counts.
#         # NOTE: truncation can break perfect alignment; we’ll clamp using actual seq lengths.
#
#         hint_ids = tok(
#             list(hint_parts),
#             add_special_tokens=False,
#             padding=False,
#             truncation=True,
#             return_tensors=None,
#         )["input_ids"]
#
#         qa_ids = tok(
#             list(qa_parts),
#             add_special_tokens=False,
#             padding=False,
#             truncation=True,
#             return_tensors=None,
#         )["input_ids"]
#
#         # Build masks per sample.
#         B, L = input_ids.shape
#         z1_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
#         z3_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
#
#         # We assume the full text layout is:
#         #   [<image>] "\n" [hint tokens] "\n" [qa tokens]
#         #
#         # Tokenizers may tokenize "\n" into 1+ tokens, and some processors may insert BOS.
#         # So instead of hardcoding offsets, we find the first non-padding token index,
#         # then find the image token position(s) and locate text segments relative to that.
#         #
#         # Strategy:
#         # - find first occurrence of image token in each row -> idx_img
#         # - treat "hint region" as the next `len(hint_ids[i])` tokens *after* idx_img + gap
#         # - treat qa region as the next `len(qa_ids[i])` tokens after hint + gap
#         #
#         # We detect "gap" by searching forward until we hit the first token that matches
#         # the first hint token id sequence (best-effort).
#
#         for i in range(B):
#             row_ids = input_ids[i]
#             row_attn = attention_mask[i].bool()
#             valid_len = int(row_attn.sum().item())
#
#             img_positions = torch.nonzero(row_ids[:valid_len] == image_token_id, as_tuple=False).view(-1)
#             if len(img_positions) == 0:
#                 # If processor doesn't actually include the token (some models handle images separately),
#                 # you can set z2 empty and start from beginning.
#                 start_after_img = 0
#             else:
#                 # pick first <image> token
#                 start_after_img = int(img_positions[0].item()) + 1
#
#             # Best-effort: skip any immediate separator tokens that come from "\n" etc.
#             # We just advance until we find a position where the remaining tail is long enough.
#             hlen = len(hint_ids[i])
#             qlen = len(qa_ids[i])
#
#             # Clamp if truncation shortened things.
#             # We try to allocate hint then qa within valid_len.
#             # If not enough room, we shrink qa then hint.
#             max_room = max(0, valid_len - start_after_img)
#             hlen_eff = min(hlen, max_room)
#             max_room2 = max(0, max_room - hlen_eff)
#             qlen_eff = min(qlen, max_room2)
#
#             # Mark z1
#             h_start = start_after_img
#             h_end = min(valid_len, h_start + hlen_eff)
#             if h_end > h_start:
#                 z1_mask[i, h_start:h_end] = True
#
#             # Mark z3 right after hint
#             q_start = h_end
#             q_end = min(valid_len, q_start + qlen_eff)
#             if q_end > q_start:
#                 z3_mask[i, q_start:q_end] = True
#
#             # Make sure we don't label the <image> token itself as hint/qa
#             if len(img_positions) > 0:
#                 z1_mask[i, img_positions[0]] = False
#                 z3_mask[i, img_positions[0]] = False
#
#         # Convenience: also return the per-modality token ids by masking the full ids
#         # (these will be ragged if you try to pack them; masks are usually better).
#         z1 = {"ids": input_ids, "mask": z1_mask}
#         z2 = {"ids": input_ids, "mask": z2_mask}
#         z3 = {"ids": input_ids, "mask": z3_mask}
#
#         meta = {
#             "proc": proc,
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "pixel_values": proc["pixel_values"],
#             "image_grid_thw": proc.get("image_grid_thw"),
#             "letters_list": letters_list,
#             "choices_list": choices_list,
#         }
#         return z1, z2, z3, meta
#
#     def forward(
#             self,
#             x,
#             *,
#             label=None,
#             return_features=False,
#             current_step=None,
#             image_token_mask=None,
#             text_token_mask=None,
#             **kwargs,
#     ):
#         z1, z2, z3, meta = self._prepare_modality_tokens(x, **kwargs)
#         output = self._forward_from_tokens(
#             z1=z1, z2=z2, z3=z3, meta=meta,
#             label=label,
#             return_features=return_features,
#             current_step=current_step,
#             **kwargs,
#         )
#
#         if self.synergy_coeff > 0:
#             synergy_losses = self.synib.compute_training_losses(
#                 x, base_output=output, **kwargs
#             )
#             output["losses"].update(synergy_losses)
#
#         return output
#
#     def _build_hint_and_qa_parts(self, hint_texts, qa_texts, letters_list, choices_list=None):
#         """
#         Returns:
#           hint_parts: List[str] length B
#           qa_parts:   List[str] length B
#         """
#         # Minimal assumption:
#         # - hint_texts[i] is the hint string
#         # - qa_texts[i] already includes question+answers OR you build it with choices elsewhere
#         #
#         # If your existing _build_prompts_with_choices already formats everything,
#         # it’s worth refactoring it to return (hint_part, qa_part) instead of a single string.
#         hint_parts = list(hint_texts)
#
#         # Example: if qa_texts already contains "Q: ... A: ..." including choices formatting:
#         qa_parts = list(qa_texts)
#
#         # If you actually need to inject letters/choices into qa_parts, do it here.
#         # (Leaving it as-is because your original snippet doesn't show where choices_list is used.)
#         return hint_parts, qa_parts
#
#     def _forward_from_tokens(
#             self,
#             *,
#             z1, z2, z3, meta,
#             label=None,
#             return_features=False,
#             current_step=None,
#             **kwargs,
#     ):
#         # You still run the encoder on the full packed batch, as before:
#         hidden = self._encode(
#             input_ids=meta["input_ids"],
#             attention_mask=meta["attention_mask"],
#             pixel_values=meta["pixel_values"],
#             image_grid_thw=meta["image_grid_thw"],
#         )
#
#         # Your existing CLS readout:
#         h_cls = self._get_cls_token_repr(hidden, meta["input_ids"])
#         h_cls = h_cls.to(self.enc_0.linear.weight.dtype)
#         head_logits = self.enc_0(h_cls)
#
#         losses = {}
#         if label is not None:
#             losses["ce_head"] = self._mc_ce_loss(head_logits, label)
#
#         preds = {"combined": head_logits}
#         features = {
#             "h_cls": h_cls,
#             # expose modality masks for downstream use/debug
#             "z1": z1["ids"],
#             "z2": z2["ids"],
#             "z3": z3["ids"],
#             "z1_mask": z1["mask"],
#             "z2_mask": z2["mask"],
#             "z3_mask": z3["mask"],
#         }
#         if return_features:
#             features["hidden"] = hidden
#
#         if (not self.training) and getattr(self.args, "do_zeroshot_parse", False):
#             raw_text_answers, mc_from_text = self._generate_raw_answers(
#                 meta["proc"], meta["input_ids"], letters_list=meta["letters_list"]
#             )
#             preds["raw_text"] = raw_text_answers
#             preds["mc_from_text"] = mc_from_text
#
#         return {"preds": preds, "features": features, "losses": losses}

class QwenVL_ScienceQA_Synergy_SynIB(nn.Module):
    """
    Multimodal (image+text) ScienceQA as 5-way classification.
    Backbone is frozen EXCEPT:
      - classifier head enc_0 (always trainable)
      - (optional) learnable <CLS> embedding row ONLY (via gradient masking hook)
      - (optional) final LM norm (cheap, sometimes helps)

    Readout is the hidden state at the appended <CLS> token (placed at end of prompt).
    """

    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []

        self.args = args
        self.synergy_weight = float(self.args.get("bias_infusion", {}).get("l", 0.0))
        self.max_new_tokens = getattr(args, "max_new_tokens", 32)
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        HF_CACHE = getattr(self.args, "save_base_dir", None)

        # -----------------------------
        # Processor / Tokenizer
        # -----------------------------
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=HF_CACHE)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Add <CLS> token to tokenizer
        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
            device_map="cuda:0",
            cache_dir=HF_CACHE,
        )

        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = cfg.image_token_id
        self.image_token_str = tok.convert_ids_to_tokens(self.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = cfg.text_config.hidden_size
        else:
            self.d_model = cfg.hidden_size

        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the 5-way classifier head.")
        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

        self.synib = SynIB_Qwen(args, [], self)

    def _setup_trainables(self):
        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model

        # Optionally train final norm (cheap and often stabilizes)
        # if getattr(self.args, "train_lm_norm", False) and lm is not None and hasattr(lm, "norm"):
        #     for p in lm.norm.parameters():
        #         p.requires_grad = True

        # Make <CLS> embedding learnable WITHOUT unfreezing whole embedding table
        # (default True; set args.train_cls_row=False to disable)
        if self.args.cls_finetune:
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                # ensure grads flow to emb.weight (we'll mask them)
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                # build a (vocab, hidden) mask with 1s only for cls row
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                # register once
                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

        # NOTE: if you enabled synergy modules, mark them trainable here.

    def _load_cls_embedding(self):

        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)

        self.load_cls_embedding(cls_path)

    def load_cls_embedding(self, path, strict_dim=True):

        assert os.path.isfile(path), f"CLS embedding file not found: {path}"

        ckpt = torch.load(path, map_location="cpu")

        if "cls_row" not in ckpt:
            raise KeyError("CLS checkpoint must contain 'cls_row'")

        cls_row = ckpt["cls_row"]
        saved_cls_id = ckpt.get("cls_token_id", self.cls_token_id)

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        current_cls_id = int(self.cls_token_id)

        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(
                f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}"
            )

        if saved_cls_id != current_cls_id:
            print(
                f"[WARN] saved cls_token_id={saved_cls_id} "
                f"!= current cls_token_id={current_cls_id} — copying to current index"
            )

        with torch.no_grad():
            emb.weight[current_cls_id].copy_(
                cls_row.to(emb.weight.device, emb.weight.dtype)
            )

        print(f"[OK] Loaded CLS embedding from {path}")

    def _apply_lora(self):
        cfg = getattr(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
        )

        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _build_prompts_with_choices(self, hint_texts, qa_texts, letters_list):
        prompts = []
        for hint, qa, letters in zip(hint_texts, qa_texts, letters_list):
            parts = []
            if hint is not None and hint.strip():
                parts.append(hint.strip())
            if qa is not None and qa.strip():
                parts.append(qa.strip())

            if letters:
                letters_str = ", ".join(f"({L})" for L in letters)
                parts.append(f"Answer with only one of: {letters_str}.")

            # Put CLS token at the END so it can attend to all previous tokens (causal LM)
            parts.append("<CLS>")

            prompts.append("\n\n".join(parts))
        return prompts

    def _encode(self, input_ids, attention_mask, pixel_values, image_grid_thw=None):
        # IMPORTANT: no torch.no_grad() here; we need grads at least to CLS row + head
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1]  # (B, T, d)

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        device = input_ids.device

        # position of <CLS> (assumes exactly once per sample)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)  # (B,)
        h = hidden[torch.arange(B, device=device), cls_pos]             # (B,d)
        h = F.layer_norm(h, (h.shape[-1],))
        return h

    def _find_subsequence(self, haystack, needle):
        """
        Return (start, end) indices of needle in haystack (both python lists of ints),
        or None if not found.
        """
        n = len(needle)
        if n == 0:
            return None
        # simple O(T*n) is fine for debug
        for i in range(len(haystack) - n + 1):
            if haystack[i:i + n] == needle:
                return (i, i + n)
        return None

    def debug_token_regions(self, processor, proc, prompts_with_image, hint_texts, image_token_str, max_print=2):
        """
        processor: your AutoProcessor
        proc: output of processor(... return_tensors="pt") with keys input_ids, attention_mask, etc.
        prompts_with_image: list[str] length B (your final text prompts you fed into processor)
        hint_texts: list[str|None] length B (your hint per example)
        image_token_str: the prefix string you add (self.image_token_str)
        """

        tok = processor.tokenizer
        input_ids = proc["input_ids"]
        attn = proc["attention_mask"]
        B, T = input_ids.shape

        hint_mask = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)
        image_mask = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)

        # 1) Identify content region because you set padding_side="left"
        # content starts at: T - sum(attn)
        content_starts = (T - attn.sum(dim=1)).tolist()
        content_lens = attn.sum(dim=1).tolist()

        # 2) Try to discover likely vision special tokens / ids
        # Qwen-VL usually has some of these; we check what actually exists in your tokenizer vocab.
        candidate_vision_tokens = [
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
            "<image>", "<img>", "img", "<|image|>"
        ]

        existing_vision = []
        for s in candidate_vision_tokens:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid != tok.unk_token_id:
                existing_vision.append((s, tid))

        # Also tokenize your image_token_str itself (this is often the most reliable)
        image_prefix_ids = tok(image_token_str, add_special_tokens=False).input_ids
        image_prefix_id_set = set(image_prefix_ids)

        # 3) For each example: locate hint subsequence and vision tokens inside the content slice
        spans = []
        for i in range(B):
            cs = content_starts[i]
            clen = int(content_lens[i])
            content_ids = input_ids[i, cs:cs + clen].tolist()

            # --- Hint: tokenize hint alone and search inside content_ids
            hint = hint_texts[i] if hint_texts is not None else None
            hint_span_global = None

            if hint is not None and str(hint).strip():
                # Important: match how it appears in your prompt. You append hint.strip() as a "parts" item
                # and later join with "\n\n". So the hint in the full prompt is exactly hint.strip()
                # (no extra prefix), BUT it is followed by "\n\n" if there is more after it.
                hint_ids = tok(hint.strip(), add_special_tokens=False).input_ids
                local = self._find_subsequence(content_ids, hint_ids)

                # If exact match fails (newlines/spacing differences), try a couple fallback variants:
                if local is None:
                    hint_ids = tok(hint.strip() + "\n\n", add_special_tokens=False).input_ids
                    local = self._find_subsequence(content_ids, hint_ids)
                if local is None:
                    hint_ids = tok("\n\n" + hint.strip(), add_special_tokens=False).input_ids
                    local = self._find_subsequence(content_ids, hint_ids)

                if local is not None:
                    hs, he = local
                    # convert to global positions in the padded sequence
                    ghs, ghe = cs + hs, cs + he
                    hint_mask[i, ghs:ghe] = True
                    hint_span_global = (ghs, ghe)

            # --- Image tokens: mark positions that correspond to vision placeholders in input_ids
            # Strategy A: mark any token that is part of your image_token_str tokenization
            # Strategy B: if tokenizer has explicit <|image_pad|> etc, mark those too
            row_ids = input_ids[i].tolist()

            # A) image_token_str token ids
            for j, tid in enumerate(row_ids):
                if tid in image_prefix_id_set:
                    image_mask[i, j] = True

            # B) known explicit vision tokens
            for (s, tid) in existing_vision:
                # mark all occurrences
                # (this can include <|image_pad|> repeated many times)
                for j, t in enumerate(row_ids):
                    if t == tid:
                        image_mask[i, j] = True

            # Collect some debug info
            spans.append({
                "i": i,
                "content_start": cs,
                "content_len": clen,
                "hint_span_global": hint_span_global,
                "n_hint_tokens": int(hint_mask[i].sum().item()),
                "n_image_tokens": int(image_mask[i].sum().item()),
            })

        # # 4) Print a small debug preview
        # print("=== Vision token ids that exist in this tokenizer ===")
        # print(existing_vision)
        # print("=== image_token_str tokenization ===")
        # print("image_token_str:", repr(image_token_str))
        # print("image_prefix_ids:", image_prefix_ids)
        #
        # if "image_grid_thw" in proc and proc["image_grid_thw"] is not None:
        #     print("=== image_grid_thw ===")
        #     print(proc["image_grid_thw"])

        # Show a couple rows decoded around hint and image positions
        for i in range(min(B, max_print)):
            print(f"\n--- Example {i} ---")
            print("spans:", spans[i])

            # Decode only content (non-pad) for readability
            cs = spans[i]["content_start"]
            clen = spans[i]["content_len"]
            content_ids = input_ids[i, cs:cs + clen]
            print("Decoded content:\n", tok.decode(content_ids, skip_special_tokens=False))

            # Show the indices where masks are true (first ~80 indices)
            hint_pos = torch.where(hint_mask[i])[0].tolist()
            img_pos = torch.where(image_mask[i])[0].tolist()
            print("hint positions (global, first 80):", hint_pos[:80])
            print("image positions (global, first 80):", img_pos[:80])

        masks = {"hint_mask": hint_mask, "image_mask": image_mask}
        return masks

    # ============================================================
    #  (Optional) generation for eval-time parsing (unchanged)
    # ============================================================
    def _generate_raw_answers(self, proc, input_ids, *, letters_list):
        gen_inputs = {
            k: v for k, v in proc.items()
            if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        }
        gen_inputs = {k: v.to(self.backbone.device) for k, v in gen_inputs.items()}

        with torch.no_grad():
            gen_ids = self.backbone.generate(
                **gen_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, gen_ids)]
        raw_answers = self.processor.batch_decode(
            gen_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        import re

        def clean_answer(ans: str):
            lines = [l.strip() for l in ans.splitlines() if l.strip()]
            if not lines:
                return ans.strip()
            first = lines[0]
            m = re.search(r"\(([A-Za-z])\)", first)
            if m:
                return f"({m.group(1).upper()})"
            m2 = re.search(r"\b([A-Za-z])\b", first)
            if m2:
                return f"({m2.group(1).upper()})"
            return first

        cleaned = [clean_answer(ans) for ans in raw_answers]

        pred_indices = []
        for ans, letters in zip(cleaned, letters_list):
            if not letters:
                pred_indices.append(-1)
                continue
            letters_upper = [L.upper() for L in letters]
            m = re.search(r"\(([A-Za-z])\)", ans)
            if not m:
                pred_indices.append(-1)
                continue
            letter = m.group(1).upper()
            pred_indices.append(letters_upper.index(letter) if letter in letters_upper else -1)

        pred_indices = torch.tensor(pred_indices, device=input_ids.device, dtype=torch.long)
        return cleaned, pred_indices

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            class_weights = self.args.class_weights.to(logits.device)
            return F.cross_entropy(logits, labels, weight=class_weights)
        return F.cross_entropy(logits, labels)

    def apply_custom_masks(self, base_att_mask, hint_attention_mask: Optional[torch.Tensor] = None, image_attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if hint_attention_mask is None and image_attention_mask is None:
            return base_att_mask
        combined = base_att_mask.clone()
        if hint_attention_mask is not None:
            combined = combined * hint_attention_mask
        if image_attention_mask is not None:
            combined = combined * image_attention_mask
        return combined

    def _compute_logits(self, x, m0=None, m1=None, **kwargs):
        hint_texts = x[0]
        qa_texts = x[1]
        images = x[2]
        choices_list = x[3] if len(x) > 3 else kwargs.get("choices", None)
        letters_list = x[4] if len(x) > 4 else kwargs.get("letters", None)

        if choices_list is None:
            raise ValueError("choices_list (x[3] or kwargs['choices']) is required for MC setup.")
        if letters_list is None:
            raise ValueError("letters_list (x[4] or kwargs['letters']) is required for zero-shot parsing.")

        device = images.device

        prompts = self._build_prompts_with_choices(hint_texts, qa_texts, letters_list)
        prompts_with_image = [self.image_token_str + "\n" + p for p in prompts]
        image_list = [img for img in images]

        proc = self.processor(  text=prompts_with_image,
                                images=image_list,
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        proc = {k: v.to(device) for k, v in proc.items()}

        if m0 is None and m1 is None:
            masks = self.debug_token_regions(
                processor=self.processor,
                proc=proc,
                prompts_with_image=prompts_with_image,
                hint_texts=hint_texts,
                image_token_str=self.image_token_str,
                max_print=0
            )
        else:
            masks = None

        att_mask = self.apply_custom_masks(proc["attention_mask"], m0, m1)

        hidden = self._encode(  input_ids=proc["input_ids"],
                                attention_mask=att_mask,
                                pixel_values=proc["pixel_values"],
                                image_grid_thw=proc.get("image_grid_thw"))

        h_cls = self._get_cls_token_repr(hidden, proc["input_ids"])
        h_cls = h_cls.to(self.enc_0.linear.weight.dtype)
        head_logits = self.enc_0(h_cls)

        return head_logits, {"hidden": hidden, "h_cls": h_cls}, masks

    def forward(self, x, label=None, **kwargs):
        head_logits, features, masks = self._compute_logits(x, **kwargs)
        losses = {}
        if label is not None:
            losses["ce_head"] = self._mc_ce_loss(head_logits, label)

        output = {"preds": {"combined": head_logits}, "features": features, "losses": losses, "masks": masks}
        if self.synergy_weight > 0:
            synergy_losses = self.synib.compute_training_losses( x, base_output=output, **kwargs)
            output["losses"].update(synergy_losses)

        return output


# ============================================================
#  Standalone extraction of CLS embedding from checkpoint
# ============================================================
if __name__ == "__main__":
    import torch
    import os

    CKPT_PATH = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ScienceQA/Synprom_IBInput2_fold0_lr0.0001_wd0.0001.pth.tar"

    assert os.path.isfile(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"

    CKPT_DIR = os.path.dirname(CKPT_PATH)
    CKPT_BASE = os.path.basename(CKPT_PATH)
    CLS_PATH = os.path.join(
        CKPT_DIR,
        CKPT_BASE.replace(".pth.tar", "_cls_embedding.pt"),
    )

    device = "cpu"

    print(f"[INFO] Loading checkpoint from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    print(ckpt.keys())

    if "best_model_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain 'best_model_weights'")

    state = ckpt["best_model_state_dict"]

    # ---- find CLS embedding table ----
    cls_weight_key = None
    for k in state.keys():
        if k.endswith("embed_tokens.weight"):
            cls_weight_key = k
            break

    if cls_weight_key is None:
        raise RuntimeError("Could not find embed_tokens.weight in checkpoint")

    # ---- tokenizer to get CLS id ----
    from transformers import AutoProcessor
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    tok = processor.tokenizer
    tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
    cls_token_id = tok.convert_tokens_to_ids("<CLS>")

    emb_weight = state[cls_weight_key]            # (vocab, d_model)
    cls_row = emb_weight[cls_token_id].detach().cpu()

    torch.save(
        {
            "cls_token_id": cls_token_id,
            "cls_row": cls_row,
        },
        CLS_PATH,
    )

    print(f"[OK] Saved CLS embedding to {CLS_PATH}")
    print(f"[OK] CLS row shape: {tuple(cls_row.shape)}")

