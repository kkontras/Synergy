import copy

from models.model_utils.fusion_gates import *
from models.VAVL_git.VAVL.conformer.model import Conformer
# from models.MCR_Models import TF_Fusion
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
from typing import Any, Dict, List, Optional, Sequence

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        self.hidden_size = getattr(args, "d_model", 2048)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, **kwargs):
        return self.linear(x)
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


    def _load_cls_embedding(self):

        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)

        self.load_cls_embedding(cls_path)

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

        # # CLS readout (stable position)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
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
class QwenVL_ScienceQA_Unimodal_Image(nn.Module):
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


    def _load_cls_embedding(self):

        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)

        self.load_cls_embedding(cls_path)

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

    def _build_prompts_with_choices(self, qa_texts, letters_list):
        prompts = []
        for qa, letters in zip(qa_texts, letters_list):
            parts = []
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
        qa_texts = x[1]
        images = x[2]
        choices_list = x[3] if len(x) > 3 else kwargs.get("choices", None)
        letters_list = x[4] if len(x) > 4 else kwargs.get("letters", None)

        if choices_list is None:
            raise ValueError("choices_list (x[3] or kwargs['choices']) is required for MC setup.")
        if letters_list is None:
            raise ValueError("letters_list (x[4] or kwargs['letters']) is required for zero-shot parsing.")

        device = images.device

        prompts = self._build_prompts_with_choices(qa_texts, letters_list)
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

        # # CLS readout (stable position)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
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
class QwenVL_ScienceQA_Unimodal_Text(nn.Module):
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


    def _load_cls_embedding(self):

        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)

        self.load_cls_embedding(cls_path)

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
    def _encode(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        # IMPORTANT: no torch.no_grad() here; we need grads at least to CLS row + head
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
        choices_list = x[3] if len(x) > 3 else kwargs.get("choices", None)
        letters_list = x[4] if len(x) > 4 else kwargs.get("letters", None)

        if choices_list is None:
            raise ValueError("choices_list (x[3] or kwargs['choices']) is required for MC setup.")
        if letters_list is None:
            raise ValueError("letters_list (x[4] or kwargs['letters']) is required for zero-shot parsing.")

        device = self.enc_0.linear.weight.device

        prompts = self._build_prompts_with_choices(hint_texts, qa_texts, letters_list)
        prompts_with_image = [p for p in prompts]

        proc = self.processor(
            text=prompts_with_image,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        proc = {k: v.to(device) for k, v in proc.items()}

        input_ids = proc["input_ids"]
        attention_mask = proc["attention_mask"]
        image_grid_thw = proc.get("image_grid_thw")

        hidden = self._encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
        )

        # # CLS readout (stable position)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
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
class SynIB_QwenFaster(nn.Module):
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
        p = float(self.perturb.get("p_min", 0.5))  # mask probability
        m1_t, m2_t = None, None
        if px1:
            mask_1 = (torch.rand_like(m1[m1==True].float()) > p).to(dtype=m1.dtype)  # [K*B, T, F] in {0,1}
            m1_t = m1.clone()
            m1_t[m1] = mask_1
        if px2:
            mask_2 = (torch.rand_like(m2[m2==True].float()) > p).to(dtype=m2.dtype)  # [K*B, T, F] in {0,1}
            m2_t = m2.clone()
            m2_t[m2] = mask_2

        return m1_t, m2_t

    def _learned_masks(self, m1, m2, px1, px2, **kwargs):
        """
        Learn per-token keep/mask gates on the subset indicated by m1/m2.

        Returns:
          m1_t, m2_t: bool [B,T] masks, where True means KEEP
        """
        label = kwargs["label"]  # [B]
        proc = kwargs["proc"]  # dict with input_ids, attention_mask
        debug = bool(kwargs.get("debug", False))
        debug_every = int(kwargs.get("debug_every", 5))

        pcfg = getattr(self, "perturb", {}) if hasattr(self, "perturb") else getattr(self.main.args, "perturb", {})
        steps = int(pcfg.get("steps", 1))
        lr = float(pcfg.get("lr", 1e-1))
        tau = float(pcfg.get("tau", 1.0))
        lsparse = float(pcfg.get("lsparse", 1.0))
        hard = bool(pcfg.get("hard", True))
        hard_thresh = float(pcfg.get("hard_thresh", 0.5))
        noise_std = float(pcfg.get("noise_std", 1.0))
        fill_mode = pcfg.get("fill", "noise")  # "noise"/"zeros"

        input_ids = proc["input_ids"]
        attn = proc["attention_mask"]
        device = input_ids.device
        B, T = input_ids.shape

        def _pct(num, den):
            den = float(den)
            return 0.0 if den <= 0 else 100.0 * float(num) / den

        if debug:
            print(f"[learned_masks] B={B} T={T} steps={steps} lr={lr} tau={tau} lsparse={lsparse} "
                  f"hard={hard} hard_thresh={hard_thresh} fill={fill_mode} noise_std={noise_std}")
            if m1 is not None:
                print(
                    f"[learned_masks] m1 eligible: {int(m1.sum().item())} / {B * T} ({_pct(m1.sum().item(), B * T):.2f}%)")
            if m2 is not None:
                print(
                    f"[learned_masks] m2 eligible: {int(m2.sum().item())} / {B * T} ({_pct(m2.sum().item(), B * T):.2f}%)")

        # Freeze backbone + head while learning gates
        req = [p.requires_grad for p in self.main.parameters()]
        for p in self.main.parameters():
            p.requires_grad_(False)

        try:
            with torch.no_grad():
                lm = self.main.backbone.model.language_model
                emb = lm.embed_tokens(input_ids)  # (B,T,D)

            def make_eps_like(x):
                if fill_mode == "zeros":
                    return torch.zeros_like(x)
                return torch.randn_like(x) * noise_std

            def apply_gate(emb0, g_keep):
                eps = make_eps_like(emb0)
                return g_keep * emb0 + (1.0 - g_keep) * eps

            def run_logits_from_embeds(emb_t):
                hidden = self.main._forward_from_embeds(emb_t, attn)
                h_cls = self.main._get_cls_token_repr(hidden, input_ids).to(self.main.enc_0.linear.weight.dtype)
                logits = self.main.enc_0(h_cls)
                return logits

            # Optional: baseline CE on clean embeddings (for comparison)
            if debug:
                with torch.no_grad():
                    logits_clean = run_logits_from_embeds(emb)
                    ce_clean = float(F.cross_entropy(logits_clean, label).item())
                    print(f"[learned_masks] clean CE: {ce_clean:.4f}")

            def optimize_for(mask_eligible, name="m?"):
                if mask_eligible is None or mask_eligible.sum() == 0:
                    if debug:
                        print(f"[learned_masks:{name}] no eligible tokens -> skip")
                    return None

                eligible = int(mask_eligible.sum().item())
                grad_mask = mask_eligible.float()  # (B,T)

                ell = torch.zeros((B, T), device=device, dtype=torch.float32, requires_grad=True)
                opt = torch.optim.Adam([ell], lr=lr)

                for i in range(steps):
                    g = torch.sigmoid(ell / tau)  # (B,T) keep-prob
                    g3 = g.unsqueeze(-1)  # (B,T,1)

                    emb_t = apply_gate(emb, g3)
                    logits_t = run_logits_from_embeds(emb_t)

                    ce = F.cross_entropy(logits_t, label)
                    sparsity = ((1.0 - g) * grad_mask).sum() / (grad_mask.sum() + 1e-6)  # mean mask rate over eligible
                    obj = (-ce) + lsparse * sparsity

                    opt.zero_grad(set_to_none=True)
                    obj.backward()

                    if ell.grad is not None:
                        ell.grad.mul_(grad_mask)

                    torch.nn.utils.clip_grad_norm_([ell], 1.0)
                    opt.step()

                    if debug and (i == 0 or i == steps - 1 or (debug_every > 0 and (i + 1) % debug_every == 0)):
                        with torch.no_grad():
                            # how many eligible tokens are currently being "kept" vs "masked"
                            keep_frac = (g * grad_mask).sum() / (grad_mask.sum() + 1e-6)
                            mask_frac = 1.0 - keep_frac
                            ce_val = float(ce.item())
                            obj_val = float(obj.item())
                            sp_val = float(sparsity.item())
                            print(f"[learned_masks:{name}] step {i + 1:02d}/{steps} "
                                  f"CE={ce_val:.4f} obj={obj_val:.4f} "
                                  f"mask%={100.0 * mask_frac.item():.2f} keep%={100.0 * keep_frac.item():.2f} "
                                  f"sparsity={sp_val:.4f}")

                            if torch.isnan(g).any() or torch.isnan(ell).any():
                                print(f"[learned_masks:{name}] WARNING: NaNs detected (g or ell)")

                g_final = torch.sigmoid(ell / tau).detach()  # (B,T)
                if hard:
                    keep = (g_final >= hard_thresh)
                else:
                    keep = (g_final > 0.5)

                keep_full = torch.ones((B, T), device=device, dtype=torch.bool)
                keep_full[mask_eligible] = keep[mask_eligible]

                if debug:
                    with torch.no_grad():
                        kept_eligible = int(keep_full[mask_eligible].sum().item())
                        masked_eligible = eligible - kept_eligible
                        print(f"[learned_masks:{name}] final eligible={eligible} "
                              f"kept={kept_eligible} ({_pct(kept_eligible, eligible):.2f}%) "
                              f"masked={masked_eligible} ({_pct(masked_eligible, eligible):.2f}%) "
                              f"overall_masked={_pct(masked_eligible, B * T):.2f}% of all tokens")

                return keep_full

            m1_t = optimize_for(m1, name="m1") if px1 else None
            m2_t = optimize_for(m2, name="m2") if px2 else None

            if m1_t is None:
                m1_t = torch.ones_like(m1, dtype=torch.bool)
            if m2_t is None:
                m2_t = torch.ones_like(m2, dtype=torch.bool)

            # if debug:
            #     # quick check: only eligible positions should differ from True
            #     if m1 is not None:
            #         changed_outside = (~m1) & (~m1_t)  # would be bad: masking outside eligible
            #         if changed_outside.any():
            #             print(
            #                 f"[learned_masks] WARNING: m1_t masked outside eligible: {int(changed_outside.sum().item())}")
            #     if m2 is not None:
            #         changed_outside = (~m2) & (~m2_t)
            #         if changed_outside.any():
            #             print(
            #                 f"[learned_masks] WARNING: m2_t masked outside eligible: {int(changed_outside.sum().item())}")

            return m1_t, m2_t

        finally:
            for p, r in zip(self.main.parameters(), req):
                p.requires_grad_(r)

    def _kl_pass(self, base_output, px1, px2, **kwargs):
        if px1:
            feat = base_output["features"]["mask0"]
            mu = base_output["preds"]["mask0"]
        elif px2:
            feat = base_output["features"]["mask1"]
            mu = base_output["preds"]["mask1"]

        if self.synergy_type == "gaussian":
            logvar = self.logvar_head(feat)
            kl = self._gaussian_kl(mu, logvar)
        else:  # dirichlet
            evidence = F.softplus(self.evidence_head(feat))
            alpha = evidence + 1.0
            kl = self._dirichlet_kl(alpha, prior_conc=self.dirichlet_prior_conc)
        return kl


    def compute_training_losses(self, base_output, **kwargs):
        kl1 = self._kl_pass(base_output, px1=True,  px2=False, **kwargs)
        kl2 = self._kl_pass(base_output, px1=False, px2=True,  **kwargs)
        kl_diff_mse = torch.mean((kl1 - kl2) ** 2)

        if self.training:
            self._log({"reg_loss": {"kl_1": kl1, "kl_2": kl2, "kl_diff_mse": kl_diff_mse}}, **kwargs)

        losses = {"sl_1": kl1 * self.synergy_weight, "sl_2": kl2 * self.synergy_weight}
        # losses["sl_diff"] = kl_diff_mse * self.synergy_weight
        return losses
class QwenVL_ScienceQA_Synergy_SynIBFaster(nn.Module):
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
        self.device = torch.device("cuda:0")

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
            # dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
            device_map="cuda:0",
            cache_dir=HF_CACHE,
        )
        self.backbone.config.use_cache = False

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

        self.synib = SynIB_QwenFaster(args, [], self)

        self._precompute_mask_token_ids()

    def _precompute_mask_token_ids(self):
        tok = self.processor.tokenizer

        # tokens that indicate vision placeholders
        vision_candidates = [
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
            "<image>", "<img>", "<|image|>"
        ]
        vision_ids = set()
        for s in vision_candidates:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid != tok.unk_token_id:
                vision_ids.add(int(tid))

        # also include the model's configured image token string (might be multi-token)
        for tid in tok(self.image_token_str, add_special_tokens=False).input_ids:
            vision_ids.add(int(tid))

        self._vision_ids = torch.tensor(sorted(vision_ids), dtype=torch.long)  # keep on CPU; move later

        # tokenize separators used by your prompt builder
        self._nl_id = tok("\n", add_special_tokens=False).input_ids
        self._nlnl_id = tok("\n\n", add_special_tokens=False).input_ids

        # tokenize "<image>\n" prefix in token space (whatever your image_token_str is)
        self._image_prefix_ids = tok(self.image_token_str + "\n", add_special_tokens=False).input_ids

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

    def _encode(self, input_ids, attention_mask, pixel_values, image_grid_thw=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
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

    def apply_custom_masks(self, base_att_mask, m1, m2, m1_t, m2_t):

        combined_hint = base_att_mask.clone()
        combined_hint[m1.bool()] = m1_t[m1.bool()].long()

        combined_img = base_att_mask.clone()
        combined_img[m2.bool()] = m2_t[m2.bool()].long()

        return combined_hint, combined_img

    def get_masks_only_B0( self, processor, proc, prompts_with_image, hint_texts, image_token_str):
        """
        Returns (B0, T) boolean masks:
          hint_mask:  True where token overlaps the hint substring in the prompt text
          image_mask: True where token is a vision-related placeholder token

        Uses:
          - offsets mapping from tokenizer for hint mask (fast, no subsequence search)
          - torch.isin for image mask (vectorized)
        """

        tok = processor.tokenizer

        input_ids = proc["input_ids"][:]
        device = input_ids.device
        B_total, T = input_ids.shape
        B0 = len(prompts_with_image)
        assert B0 <= B_total, f"B0={B0} > proc batch={B_total}"


        # --- 1) Compute offsets for B0 prompts only (CPU is fine) ---
        # Make sure tokenizer is "fast" (offset_mapping is supported); most HF tokenizers are.
        enc = tok(
            prompts_with_image,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Offsets come back padded to the tokenizer's max length for this batch.
        # We will slice to match proc's T (they should match if prompts match).
        offsets = enc["offset_mapping"]  # (B0, T_enc, 2) on CPU
        T_enc = offsets.shape[1]
        if T_enc != T:
            # If they differ, you likely used different padding/truncation settings
            # between this call and the original processor(...) call.
            # We can still align by slicing to the min length.
            T_use = min(T_enc, T)
        else:
            T_use = T

        off_s = offsets[:, :T_use, 0]  # (B0, T_use)
        off_e = offsets[:, :T_use, 1]  # (B0, T_use)

        # --- 2) Vectorized hint span computation (no for-loop over B0) ---
        # Assumption: when hint exists, it starts right after prefix: image_token_str + "\n"
        prefix_len = len(image_token_str) + 1  # newline

        # Build tensor of hint lengths (0 for empty/None)
        # (This is “vectorized enough”: one Python pass over B0 strings; no tensor copies per token.)
        hint_clean = [("" if h is None else str(h).strip()) for h in hint_texts]
        hint_len_t = torch.tensor([len(h) for h in hint_clean], dtype=torch.long)  # (B0,) on CPU

        has_hint = hint_len_t > 0
        hint_start = torch.full((B0,), prefix_len, dtype=torch.long)  # (B0,)
        hint_end = hint_start + hint_len_t  # (B0,)
        # Mark absent hints with -1 so they produce all-False masks
        hint_start = torch.where(has_hint, hint_start, torch.full_like(hint_start, -1))
        hint_end = torch.where(has_hint, hint_end, torch.full_like(hint_end, -1))

        hs = hint_start[:, None]  # (B0, 1)
        he = hint_end[:, None]  # (B0, 1)

        # pad tokens typically have offsets (0,0)
        not_pad = ~((off_s == 0) & (off_e == 0))

        # overlap: token span intersects [hs, he)
        valid_hint = (hs >= 0)
        hint_mask_cpu = valid_hint & not_pad & (off_e > hs) & (off_s < he)  # (B0, T_use)

        # Move to device and expand to full T if needed
        hint_mask = torch.zeros((B0, T), dtype=torch.bool, device=device)
        hint_mask[:, :T_use] = hint_mask_cpu.to(device)

        # --- 3) Vectorized image mask on GPU using proc's input_ids (B0 slice) ---
        vision_candidates = [
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
            "<image>", "<img>", "<|image|>"
        ]
        vision_ids = set()
        for s in vision_candidates:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid != tok.unk_token_id:
                vision_ids.add(int(tid))
        for tid in tok(image_token_str, add_special_tokens=False).input_ids:
            vision_ids.add(int(tid))

        if len(vision_ids) == 0:
            image_mask = torch.zeros((B0, T), dtype=torch.bool, device=device)
        else:
            vision_ids_t = torch.tensor(sorted(vision_ids), device=device, dtype=input_ids.dtype)
            image_mask = torch.isin(input_ids[:B0], vision_ids_t)  # (B0, T)

        return hint_mask, image_mask

    def get_masks_from_input_ids_only(self, proc, hint_texts):
        """
        Returns:
          hint_mask:  (B,T) tokens belonging to hint span
          image_mask: (B,T) tokens belonging to vision placeholders
        No extra tokenization.
        """
        input_ids = proc["input_ids"]
        device = input_ids.device
        B, T = input_ids.shape

        # --- image mask: vectorized ---
        if getattr(self, "_vision_ids", None) is None:
            self._precompute_mask_token_ids()

        vision_ids_t = self._vision_ids.to(device=device, dtype=input_ids.dtype)
        image_mask = torch.isin(input_ids, vision_ids_t)

        # --- hint mask ---
        # If hint is empty -> all false for that row
        hint_clean = [("" if h is None else str(h).strip()) for h in hint_texts]
        has_hint = torch.tensor([len(h) > 0 for h in hint_clean], device=device)  # (B,)

        hint_mask = torch.zeros((B, T), dtype=torch.bool, device=device)

        # We assume: prefix is "<image_token>\n" then hint tokens until the first "\n\n".
        prefix_ids = torch.tensor(self._image_prefix_ids, device=device, dtype=input_ids.dtype)
        sep_ids = torch.tensor(self._nlnl_id, device=device, dtype=input_ids.dtype)

        # Helper: find first occurrence of a short pattern in each row (small, cheap loop over B)
        # B is small vs T, and pattern lengths are tiny; this is *way* cheaper than tokenizing strings again.
        for b in range(B):
            if not has_hint[b]:
                continue

            row = input_ids[b]

            # find where the prefix ends (we expect it at the beginning, but be robust)
            # match prefix_ids at some position; usually 0
            start = 0
            # (optional robustness: verify prefix match)
            if len(prefix_ids) > 0 and not torch.equal(row[: len(prefix_ids)], prefix_ids):
                # if not found at beginning, try to find it once
                found = False
                for i in range(0, T - len(prefix_ids) + 1):
                    if torch.equal(row[i:i + len(prefix_ids)], prefix_ids):
                        start = i + len(prefix_ids)
                        found = True
                        break
                if not found:
                    continue
            else:
                start = len(prefix_ids)

            # find first separator "\n\n" after start
            end = None
            for i in range(start, T - len(sep_ids) + 1):
                if torch.equal(row[i:i + len(sep_ids)], sep_ids):
                    end = i
                    break

            # if no separator found, fall back: mark until padding (attention_mask==0) if exists
            if end is None:
                attn = proc.get("attention_mask", None)
                if attn is not None:
                    end = int(attn[b].sum().item())
                else:
                    end = T

            if end > start:
                hint_mask[b, start:end] = True

        return hint_mask, image_mask

    def _forward_from_embeds(self, inputs_embeds, attention_mask):
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        return hidden

    def _compute_logits_synib_from_proc(self, x, **kwargs ):
        hint_texts = x[0]
        qa_texts = x[1]
        images = x[2]
        letters_list = x[4]

        device = images.device

        def expand_batch(x, k=3):
            return x.unsqueeze(dim=0).expand(k,*x.shape).reshape(k * x.shape[0], *x.shape[1:])

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


        m1, m2 = self.get_masks_only_B0(processor=self.processor, proc=proc, prompts_with_image=prompts_with_image, hint_texts=hint_texts, image_token_str=self.image_token_str)

        if self.args.get("perturb", {}).get("type", "rand") == "rand":
            m1t, m2t = self.synib._random_masks(m1, m2, True, True, **kwargs)
        elif self.args.get("perturb", {}).get("type", "rand") == "learned":
            m1t, m2t = self.synib._learned_masks(m1, m2, True, True, proc=proc, **kwargs)
        else:
            raise ValueError("Unknown or missing perturb.type {}".format(self.args.get("perturb", {})))
        att_mask_0, att_mask_1 = self.apply_custom_masks(proc["attention_mask"], m1, m2, m1t, m2t)

        if self.args.get("run_multiple_forwards", False):
            combined_mask = torch.cat(
                [proc["attention_mask"].unsqueeze(dim=0), att_mask_0.unsqueeze(dim=0), att_mask_1.unsqueeze(dim=0)],
                dim=0)

            combined_hidden = torch.cat([self._encode(
                input_ids=proc["input_ids"],
                attention_mask=combined_mask[i],
                pixel_values=proc["pixel_values"],
                image_grid_thw=proc["image_grid_thw"],
            )
             for i in range(3)], dim=0)
        else:
            combined_mask = torch.cat(
                [proc["attention_mask"], att_mask_0, att_mask_1],
                dim=0)

            combined_hidden = self._encode(
                input_ids=expand_batch(proc["input_ids"]),
                attention_mask=combined_mask,
                pixel_values=expand_batch(proc["pixel_values"]),
                image_grid_thw=expand_batch(proc["image_grid_thw"]),
            )


        h_cls_combined = self._get_cls_token_repr(combined_hidden, expand_batch(proc["input_ids"]))
        head_logits_combined = self.enc_0(h_cls_combined)

        head_logits, head_logits_0, head_logits_1 = torch.chunk(head_logits_combined, chunks=3, dim=0)
        h_cls, featcls_0, featcls_1 = torch.chunk(h_cls_combined, chunks=3, dim=0)

        losses = {}
        if "label" in kwargs and kwargs["label"] is not None:
            losses["ce_head"] = self._mc_ce_loss(head_logits, kwargs["label"])

        preds = {"combined": head_logits, "mask0":head_logits_0, "mask1":head_logits_1}
        features = {"h_cls": h_cls, "mask0":featcls_0, "mask1":featcls_1}

        # # If you want zero-shot text parsing at eval:
        # if (not self.training) and getattr(self.args, "do_zeroshot_parse", False):
        #     raw_text_answers, mc_from_text = self._generate_raw_answers(
        #         proc, input_ids, letters_list=letters_list
        #     )
        #     preds["raw_text"] = raw_text_answers
        #     preds["mc_from_text"] = mc_from_text

        return {"preds": preds, "features": features, "losses": losses}

    def _compute_logits_from_proc( self, x, *, label=None, **kwargs,):
        hint_texts = x[0]
        qa_texts = x[1]
        images = x[2]
        letters_list = x[4]

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

        hidden = self._encode(
            input_ids=proc["input_ids"],
            attention_mask=proc["attention_mask"],
            pixel_values=proc["pixel_values"],
            image_grid_thw=proc.get("image_grid_thw"),
        )

        h_cls = self._get_cls_token_repr(hidden, proc["input_ids"]).to(self.enc_0.linear.weight.dtype)
        head_logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_head"] = self._mc_ce_loss(head_logits, label)

        preds = {"combined": head_logits}
        features = {"h_cls": h_cls}

        return {"preds": preds, "features": features, "losses": losses}

    def forward(self, x, **kwargs):

        if self.training:
            out = self._compute_logits_synib_from_proc(x, **kwargs)
        else:
            out =  self._compute_logits_from_proc(x, **kwargs)

        if self.training and self.synergy_weight > 0:
            synergy_losses = self.synib.compute_training_losses( out, **kwargs)
            out["losses"].update(synergy_losses)
        return out


class QwenVL_ESNLI_Synergy_FrozenCLS(nn.Module):
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


    def _load_cls_embedding(self):

        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)

        self.load_cls_embedding(cls_path)

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

    def build_prompt_no_cls(
            self,
            hypothesis: Sequence[str],
            label_options: List[str],
    ) -> List[str]:

        instr_text = (
            "Task: Decide whether the image and the hypothesis match.\n"
            "Entailment: the image matches the hypothesis (supported).\n"
            "Contradiction: the image does not match the hypothesis (refuted).\n"
            "Neutral: not enough information in the image to determine a match.\n"
            f"Answer format: Output exactly one label from: {label_options}.\n"
        )

        return [
            f"Hypothesis:\n{str(h).strip()}\n\n{instr_text}"
            for h in hypothesis
        ]
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
        images = x[1]
        device = images.device

        label_options = "entailment,neutral,contradiction"

        texts = self.build_prompt_no_cls(hypothesis=hint_texts, label_options=label_options)
        # prompts_with_image = [self.image_token_str + "\n" + p for p in prompts]

        # texts: List[str]
        messages_batch = [
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": t},
            ]}]
            for t in texts
        ]
        prompts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_batch
        ]

        image_list = [img for img in images]

        proc = self.processor(
            text=prompts,
            images=image_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        proc.pop("token_type_ids", None)
        proc = {k: v.to(device) for k, v in proc.items()}
        # gen = self.backbone.generate(
        #     **proc,
        #     max_new_tokens=8,  # labels are short
        #     do_sample=False,  # deterministic for classification
        #     temperature=0.0,
        # )
        # pred_texts = self.processor.batch_decode(gen, skip_special_tokens=True)
        # print(pred_texts)


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

        # # CLS readout (stable position)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        head_logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_pred_combined"] = self._mc_ce_loss(head_logits, label)

        # Optional eval-time generation parsing (kept off by default)
        preds = {"combined": head_logits}
        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        return {"preds": preds, "features": features, "losses": losses}


# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
# from peft import LoraConfig, get_peft_model


class QwenVL_ScienceQA_Cached(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the classifier head.")

        self.args = args
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
            device_map="cuda:0",
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = int(cfg.text_config.hidden_size)
        else:
            self.d_model = int(cfg.hidden_size)

        self.enc_0 = encs[0]

        print(self.enc_0)

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False):
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

    def load_cls_embedding(self, path, strict_dim=True):
        ckpt = torch.load(path, map_location="cpu")
        cls_row = ckpt["cls_row"]

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        current_cls_id = int(self.cls_token_id)

        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}")

        with torch.no_grad():
            emb.weight[current_cls_id].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        if os.path.isfile(cls_path):
            self.load_cls_embedding(cls_path)

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

    def _encode(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        return F.layer_norm(h, (h.shape[-1],))

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(logits, labels, weight=self.args.class_weights.to(logits.device))
        return F.cross_entropy(logits, labels)

    def forward(self, x, *, label=None, return_features=False, **kwargs):
        proc = x
        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)

        hidden = self._encode(input_ids, attention_mask)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_pred_combined"] = self._mc_ce_loss(logits, label)

        features = {"combined": h_cls}

        return {"preds": {"combined": logits}, "features":  features, "losses": losses}
class QwenVL_ScienceQA_Cached_Text_PastVersion(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the 5-way classifier head.")

        self.args = args
        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
            device_map="cuda:0",
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)

        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

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

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        if not os.path.isfile(cls_path):
            return
        ckpt = torch.load(cls_path, map_location="cpu")
        cls_row = ckpt["cls_row"]
        lm = self.backbone.model.language_model
        emb = lm.embed_tokens
        with torch.no_grad():
            emb.weight[int(self.cls_token_id)].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False) and getattr(self.args, "train_cls_row", True):
            emb = lm.embed_tokens
            emb.weight.requires_grad = True
            cls_id = int(self.cls_token_id)
            mask = torch.zeros_like(emb.weight, dtype=torch.float32)
            mask[cls_id].fill_(1.0)

            def grad_mask_hook(grad):
                return grad * mask.to(grad.device, grad.dtype)

            if not hasattr(self, "_cls_grad_hooked"):
                emb.weight.register_hook(grad_mask_hook)
                self._cls_grad_hooked = True

    def _encode(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]

    def _get_cls(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        return F.layer_norm(h, (h.shape[-1],))

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(logits, labels, weight=self.args.class_weights.to(logits.device))
        return F.cross_entropy(logits, labels)

    def forward(self, x, *, label=None, return_features=False, **kwargs):
        proc = x
        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)

        hidden = self._encode(input_ids, attention_mask)
        h_cls = self._get_cls(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_head"] = self._mc_ce_loss(logits, label)

        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        return {"preds": {"combined": logits}, "features": features, "losses": losses}
class QwenVL_ScienceQA_Cached_Text(nn.Module):
    """
    Text-only ScienceQA model.
    Uses image_mask from the dataset to mask out vision tokens,
    so the model answers using *text only*.
    """

    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the 5-way classifier head.")

        self.args = args
        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
            device_map="cuda:0",
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

    # ---------------------------------------------------------
    # LoRA / loading helpers
    # ---------------------------------------------------------

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

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return

        cls_path = os.path.join(save_base_dir, cls_path)
        if not os.path.isfile(cls_path):
            return

        ckpt = torch.load(cls_path, map_location="cpu")
        cls_row = ckpt["cls_row"]

        lm = self.backbone.model.language_model
        emb = lm.embed_tokens
        with torch.no_grad():
            emb.weight[int(self.cls_token_id)].copy_(
                cls_row.to(emb.weight.device, emb.weight.dtype)
            )

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        # Optional CLS finetuning (single-row gradient mask)
        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False) and getattr(self.args, "train_cls_row", True):
            emb = lm.embed_tokens
            emb.weight.requires_grad = True

            cls_id = int(self.cls_token_id)
            mask = torch.zeros_like(emb.weight, dtype=torch.float32)
            mask[cls_id].fill_(1.0)

            def grad_mask_hook(grad):
                return grad * mask.to(grad.device, grad.dtype)

            if not hasattr(self, "_cls_grad_hooked"):
                emb.weight.register_hook(grad_mask_hook)
                self._cls_grad_hooked = True

    # ---------------------------------------------------------
    # Core model logic
    # ---------------------------------------------------------

    def _encode(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]

    def _get_cls(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        return F.layer_norm(h, (h.shape[-1],))

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(
                logits,
                labels,
                weight=self.args.class_weights.to(logits.device),
            )
        return F.cross_entropy(logits, labels)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, x, *, label=None, return_features=False, **kwargs):
        device = self.backbone.device

        input_ids = x["input_ids"].to(device)
        attention_mask = x["attention_mask"].to(device)

        image_mask = x.get("image_mask", None)
        if image_mask is None:
            raise KeyError("image_mask is required for QwenVL_ScienceQA_Cached_Text")

        image_mask = image_mask.to(device).bool()

        # Mask OUT image tokens (mirror of hint masking in image model)
        keep = ~image_mask
        attention_mask = attention_mask * keep.to(attention_mask.dtype)

        hidden = self._encode(input_ids, attention_mask)
        h_cls = self._get_cls(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_head"] = self._mc_ce_loss(logits, label)

        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        return {
            "preds": {"combined": logits},
            "features": features,
            "losses": losses,
        }
class QwenVL_ScienceQA_Cached_Image(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the 5-way classifier head.")

        self.args = args
        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
            device_map="cuda:0",
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)

        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

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

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        if not os.path.isfile(cls_path):
            return
        ckpt = torch.load(cls_path, map_location="cpu")
        cls_row = ckpt["cls_row"]
        lm = self.backbone.model.language_model
        emb = lm.embed_tokens
        with torch.no_grad():
            emb.weight[int(self.cls_token_id)].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False) and getattr(self.args, "train_cls_row", True):
            emb = lm.embed_tokens
            emb.weight.requires_grad = True
            cls_id = int(self.cls_token_id)
            mask = torch.zeros_like(emb.weight, dtype=torch.float32)
            mask[cls_id].fill_(1.0)

            def grad_mask_hook(grad):
                return grad * mask.to(grad.device, grad.dtype)

            if not hasattr(self, "_cls_grad_hooked"):
                emb.weight.register_hook(grad_mask_hook)
                self._cls_grad_hooked = True

    def _encode(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]

    def _get_cls(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        return F.layer_norm(h, (h.shape[-1],))

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(logits, labels, weight=self.args.class_weights.to(logits.device))
        return F.cross_entropy(logits, labels)

    def forward(self, x, *, label=None, return_features=False, **kwargs):
        proc = x
        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        
        hint_mask = proc.get("hint_mask", None)
        if hint_mask is None:
            hint_mask = proc.get("text_mask", None)

        if hint_mask is None:
            print(proc.keys())
            raise KeyError("hint_mask or text_mask is required for QwenVL_ScienceQA_Cached_Image")

        hint_mask = hint_mask.to(device).bool()
        keep = (~hint_mask)

        attention_mask = attention_mask * keep.to(attention_mask.dtype)

        hidden = self._encode(input_ids, attention_mask)
        h_cls = self._get_cls(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_head"] = self._mc_ce_loss(logits, label)

        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        return {"preds": {"combined": logits}, "features":  features, "losses": losses}
class QwenVL_ScienceQA_Cached_SynIBFaster(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []

        self.args = args
        self.device = torch.device("cuda:0")

        self.synergy_weight = float(self.args.get("bias_infusion", {}).get("l", 0.0))
        self.max_new_tokens = getattr(args, "max_new_tokens", 32)
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="cuda:0",
            cache_dir=hf_cache,
        )
        self.backbone.config.use_cache = False

        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)
        self.image_token_str = tok.convert_ids_to_tokens(self.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = int(cfg.text_config.hidden_size)
        else:
            self.d_model = int(cfg.hidden_size)

        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the 5-way classifier head.")
        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

        self.synib = SynIB_QwenFaster(args, [], self)
        self._precompute_mask_token_ids()

    def _precompute_mask_token_ids(self):
        tok = self.processor.tokenizer

        vision_candidates = [
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
            "<image>", "<img>", "<|image|>"
        ]
        vision_ids = set()
        for s in vision_candidates:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid != tok.unk_token_id:
                vision_ids.add(int(tid))

        for tid in tok(self.image_token_str, add_special_tokens=False).input_ids:
            vision_ids.add(int(tid))

        self._vision_ids = torch.tensor(sorted(vision_ids), dtype=torch.long)
        self._nlnl_id = tok("\n\n", add_special_tokens=False).input_ids
        self._image_prefix_ids = tok(self.image_token_str + "\n", add_special_tokens=False).input_ids

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False):
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        self.load_cls_embedding(cls_path)

    def load_cls_embedding(self, path, strict_dim=True):
        if not os.path.isfile(path):
            return

        ckpt = torch.load(path, map_location="cpu")
        if "cls_row" not in ckpt:
            raise KeyError("CLS checkpoint must contain 'cls_row'")

        cls_row = ckpt["cls_row"]

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}")

        with torch.no_grad():
            emb.weight[int(self.cls_token_id)].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

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

    def _encode(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
        )
        return outputs.hidden_states[-1]

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        return F.layer_norm(h, (h.shape[-1],))

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            class_weights = self.args.class_weights.to(logits.device)
            return F.cross_entropy(logits, labels, weight=class_weights)
        return F.cross_entropy(logits, labels)

    def apply_custom_masks(self, base_att_mask, m1, m2, m1_t, m2_t):
        combined_hint = base_att_mask.clone()
        combined_hint[m1.bool()] = m1_t[m1.bool()].long()
        combined_img = base_att_mask.clone()
        combined_img[m2.bool()] = m2_t[m2.bool()].long()
        return combined_hint, combined_img

    def _forward_from_embeds(self, inputs_embeds, attention_mask):
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        return hidden

    def get_masks_from_input_ids_only(self, proc, hint_texts):
        input_ids = proc["input_ids"]
        device = input_ids.device
        B, T = input_ids.shape

        vision_ids_t = self._vision_ids.to(device=device, dtype=input_ids.dtype)
        image_mask = torch.isin(input_ids, vision_ids_t)

        hint_clean = [("" if h is None else str(h).strip()) for h in hint_texts]
        has_hint = torch.tensor([len(h) > 0 for h in hint_clean], device=device)

        hint_mask = torch.zeros((B, T), dtype=torch.bool, device=device)

        prefix_ids = torch.tensor(self._image_prefix_ids, device=device, dtype=input_ids.dtype)
        sep_ids = torch.tensor(self._nlnl_id, device=device, dtype=input_ids.dtype)

        for b in range(B):
            if not has_hint[b]:
                continue

            row = input_ids[b]

            if len(prefix_ids) > 0 and torch.equal(row[: len(prefix_ids)], prefix_ids):
                start = len(prefix_ids)
            else:
                start = 0
                found = False
                for i in range(0, T - len(prefix_ids) + 1):
                    if torch.equal(row[i:i + len(prefix_ids)], prefix_ids):
                        start = i + len(prefix_ids)
                        found = True
                        break
                if not found:
                    continue

            end = None
            for i in range(start, T - len(sep_ids) + 1):
                if torch.equal(row[i:i + len(sep_ids)], sep_ids):
                    end = i
                    break

            if end is None:
                end = int(proc["attention_mask"][b].sum().item())

            if end > start:
                hint_mask[b, start:end] = True

        return hint_mask, image_mask

    def _compute_logits_synib_from_proc(self, proc, **kwargs):
        input_ids = proc["input_ids"].to(self.backbone.device)
        attention_mask = proc["attention_mask"].to(self.backbone.device)
        pixel_values = proc.get("pixel_values", None)
        image_grid_thw = proc.get("image_grid_thw", None)

        m1 = proc.get("hint_mask", None)
        m2 = proc.get("image_mask", None)
        if m1 is None or m2 is None:
            raise KeyError("Need proc['hint_mask'] and proc['image_mask'] for SynIB cached mode.")

        m1 = m1.to(input_ids.device).bool()
        m2 = m2.to(input_ids.device).bool()

        if self.args.get("perturb", {}).get("type", "rand") == "rand":
            m1t, m2t = self.synib._random_masks(m1, m2, True, True, **kwargs)
        elif self.args.get("perturb", {}).get("type", "rand") == "learned":
            m1t, m2t = self.synib._learned_masks(m1, m2, True, True, proc={"input_ids": input_ids, "attention_mask": attention_mask}, **kwargs)
        else:
            raise ValueError(f"Unknown perturb.type: {self.args.get('perturb', {})}")

        att_mask_0, att_mask_1 = self.apply_custom_masks(attention_mask, m1, m2, m1t, m2t)

        def expand_batch(t, k=3):
            return t.unsqueeze(0).expand(k, *t.shape).reshape(k * t.shape[0], *t.shape[1:])

        if getattr(self.args, "run_multiple_forwards", False):
            masks = torch.stack([attention_mask, att_mask_0, att_mask_1], dim=0)
            hidden_all = torch.cat(
                [
                    self._encode(input_ids=input_ids, attention_mask=masks[i], pixel_values=pixel_values, image_grid_thw=image_grid_thw)
                    for i in range(3)
                ],
                dim=0,
            )
            ids_all = expand_batch(input_ids, k=3)
        else:
            masks = torch.cat([attention_mask, att_mask_0, att_mask_1], dim=0)
            ids_all = expand_batch(input_ids, k=3)
            pv_all = expand_batch(pixel_values, k=3) if pixel_values is not None else None
            thw_all = expand_batch(image_grid_thw, k=3) if image_grid_thw is not None else None
            hidden_all = self._encode(input_ids=ids_all, attention_mask=masks, pixel_values=pv_all, image_grid_thw=thw_all)

        h_cls_all = self._get_cls_token_repr(hidden_all, ids_all)
        logits_all = self.enc_0(h_cls_all)

        head_logits, head_logits_0, head_logits_1 = torch.chunk(logits_all, chunks=3, dim=0)
        h_cls, featcls_0, featcls_1 = torch.chunk(h_cls_all, chunks=3, dim=0)

        losses = {}
        if "label" in kwargs and kwargs["label"] is not None:
            losses["ce_head"] = self._mc_ce_loss(head_logits, kwargs["label"])

        preds = {"combined": head_logits, "mask0": head_logits_0, "mask1": head_logits_1}
        features = {"combined": h_cls, "mask0": featcls_0, "mask1": featcls_1}
        return {"preds": preds, "features": features, "losses": losses}

    def _compute_logits_from_proc(self, proc, *, label=None, **kwargs):
        input_ids = proc["input_ids"].to(self.backbone.device)
        attention_mask = proc["attention_mask"].to(self.backbone.device)
        pixel_values = proc.get("pixel_values", None)
        image_grid_thw = proc.get("image_grid_thw", None)

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.backbone.device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.backbone.device)

        hidden = self._encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        head_logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_head"] = self._mc_ce_loss(head_logits, label)

        return {"preds": {"combined": head_logits}, "features": {"combined": h_cls}, "losses": losses}

    def forward(self, x, **kwargs):
        if self.training:
            out = self._compute_logits_synib_from_proc(x, **kwargs)
        else:
            out = self._compute_logits_from_proc(x, **kwargs)

        if self.training and self.synergy_weight > 0:
            synergy_losses = self.synib.compute_training_losses(out, **kwargs)
            out["losses"].update(synergy_losses)
        return out


class QwenVL_ESNLI(nn.Module):
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


    def _load_cls_embedding(self):

        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)

        self.load_cls_embedding(cls_path)

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

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            class_weights = self.args.class_weights.to(logits.device)
            return F.cross_entropy(logits, labels, weight=class_weights)
        return F.cross_entropy(logits, labels)

    def _build_prompts_with_choices(self, hint_texts):
        prompts = []
        for hint in hint_texts:
            parts = []
            if hint is not None and hint.strip():
                parts.append(hint.strip())
            prompts.append("\n".join(parts))
        return prompts

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
        images = x[1]

        device = images.device

        prompts = self._build_prompts_with_choices(hint_texts)
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

        # # CLS readout (stable position)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        head_logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_head"] = self._mc_ce_loss(head_logits, label)

        # Optional eval-time generation parsing (kept off by default)
        preds = {"combined": head_logits}
        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        return {"preds": preds, "features": features, "losses": losses}


# ============================================================
#  Standalone extraction of CLS embedding from checkpoint
# ============================================================
if __name__ == "__main__":
    import torch
    import os

    # CKPT_PATH = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ScienceQA/Synprom_IBInput2_fold0_lr0.0001_wd0.0001.pth.tar"
    CKPT_PATH = "/scratch/kkontras/checkpoints/synergy/ESNLI/SynIBCache_LoRaEmb_fold0_lr0.0001_wd0.01_bs5.pth.tar"

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

    if "best_model_accuracy_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain 'best_model_accuracy_state_dict'")

    state = ckpt["best_model_accuracy_state_dict"]

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

