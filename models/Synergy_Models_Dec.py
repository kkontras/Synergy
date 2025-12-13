
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
            else:
                wandb.log(wandb_loss)

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

import torch
import torch.nn as nn
import weakref
# assumes:
#   - wandb is imported
#   - nt_xent_loss is defined
#   - TF_Fusion is defined


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

        self.perturb = {"type": args.get("perturb", None)}

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
    def _kl_loss(mu, logvar):
        return 0.5 * torch.sum(
            torch.exp(logvar) + mu**2 - 1 - logvar, dim=1
        ).mean()

    def _log(self, d, **kwargs):
        if "current_step" in kwargs:
            wandb.log(d, step=kwargs["current_step"] + 1)
        else:
            wandb.log(d)

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

    # ------------------ Perturbations ------------------

    def _perturb(self, zt, zc, direction, **kwargs):
        ptype = self.perturb["type"]

        if ptype == "zeros":
            return {"tilde": torch.zeros_like(zt)}

        if ptype == "cmn":
            mask = (torch.rand_like(zt) < self.perturb.get("p", 0.5)).float()
            scale = self.perturb.get("s", 0.1) * (
                1 + torch.tanh(zc.norm(dim=-1, keepdim=True) / self.perturb.get("c", 1))
            )
            noise = torch.randn_like(zt) * scale
            return {"tilde": (mask * zt + (1 - mask) * noise)}

        gen = self.gen_x1 if direction == "x1" else self.gen_x2
        key = "px1" if direction == "x1" else "px2"

        gkw = {"compute_loss": self.training}
        if ptype == "gennoise":
            gkw["add_noise"] = True
        if "current_step" in kwargs:
            gkw["current_step"] = kwargs["current_step"]

        out = gen(zt, zc, **gkw)
        if "losses" in out:
            self._log({key: out["losses"]}, **kwargs)
        return out

    def _encode_and_perturb(self, x, px1, px2, **kwargs):
        a, v = self.main._get_features(x, **kwargs)
        losses = {}

        if px1:
            out = self._perturb(a["features"]["combined"], v["features"]["combined"], "x1", **kwargs)
            a["features"]["combined"] = out["tilde"].detach()
            if "losses" in out: losses["px1"] = out["losses"]

        if px2:
            out = self._perturb(v["features"]["combined"], a["features"]["combined"], "x2", **kwargs)
            v["features"]["combined"] = out["tilde"].detach()
            if "losses" in out: losses["px2"] = out["losses"]

        return a, v, losses

    # ------------------ KL passes ------------------

    def _kl_pass(self, x, px1, px2, **kwargs):
        a, v, losses = self._encode_and_perturb(x, px1, px2, **kwargs)
        mu, feat = self.main._compute_logits(a, v)
        if self.synergy_type == "gaussian":
            logvar = self.logvar_head(feat)
            kl = self._gaussian_kl(mu, logvar)
        else:  # dirichlet
            evidence = F.softplus(self.evidence_head(feat))
            alpha = evidence + 1.0
            kl = self._dirichlet_kl(alpha, prior_conc=self.dirichlet_prior_conc)
        return kl, losses


    def compute_training_losses(self, x, base_output, **kwargs):
        kl1, loss1 = self._kl_pass(x, px1=True,  px2=False, **kwargs)
        kl2, loss2 = self._kl_pass(x, px1=False, px2=True,  **kwargs)

        infonce = nt_xent_loss(
            base_output["features"]["z1"],
            base_output["features"]["z2"],
            temperature=1.0,
        )
        self._log({"reg_loss": {"kl_1": kl1, "kl_2": kl2, "infonce": infonce}}, **kwargs)

        losses = {}
        losses.update(loss1)
        losses.update(loss2)
        losses["sl_1"] = kl1 * self.synergy_weight
        losses["sl_2"] = kl2 * self.synergy_weight
        losses["infonce"] = infonce * self.contrastive_weight
        return losses

class LinearHead(nn.Module):
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

        # -----------------------------
        # Backbone
        # -----------------------------
        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if getattr(args, "bf16", False) else torch.float16,
            device_map="cuda:0",
            cache_dir=HF_CACHE,
        )

        # resize embeddings if we added tokens
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = cfg.image_token_id
        self.image_token_str = tok.convert_ids_to_tokens(self.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = cfg.text_config.hidden_size
        else:
            self.d_model = cfg.hidden_size

        # -----------------------------
        # Classifier head (provided)
        # -----------------------------
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the 5-way classifier head.")
        self.enc_0 = encs[0]

        # -----------------------------
        # Optional: your synergy blocks (left disabled here)
        # -----------------------------
        # if self.synergy_coeff > 0:
        #     proj_dim = getattr(args, "proj_dim", self.d_model)
        #     self.text_proj = nn.Linear(self.d_model, proj_dim)
        #     self.image_proj = nn.Linear(self.d_model, proj_dim)
        #     if not hasattr(args, "fc_inner"):
        #         setattr(args, "fc_inner", self.d_model)
        #     self.synib = SynIB(args, encs, main=self)

        # -----------------------------
        # Training control
        # -----------------------------
        self._setup_trainables_no_lora()

    # ============================================================
    #  Trainable params: head + (optional) CLS row only
    # ============================================================
    def _get_language_model(self):
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "language_model"):
            return self.backbone.model.language_model
        return None

    def _setup_trainables_no_lora(self):
        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Always train classifier head
        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self._get_language_model()

        # Optionally train final norm (cheap and often stabilizes)
        if getattr(self.args, "train_lm_norm", False) and lm is not None and hasattr(lm, "norm"):
            for p in lm.norm.parameters():
                p.requires_grad = True

        # Make <CLS> embedding learnable WITHOUT unfreezing whole embedding table
        # (default True; set args.train_cls_row=False to disable)
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

    # ============================================================
    #  Prompt building
    # ============================================================
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

