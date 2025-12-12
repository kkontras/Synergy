import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse
from utils.deterministic_pytorch import deterministic
import torch.functional as F

# ---------------------------------------------------------------------
# 1. Synthetic dataset (two inputs)
# ---------------------------------------------------------------------
class NoisyBinaryXORDataset2(Dataset):
    def __init__(self, n_points=5000, dim=16, p_hat=0.2,
                 leak_prob=0.1, val_corr=0.5, noise_std=0.0, mode="train"):
        super().__init__()
        self.n = n_points
        self.dim = dim
        self.p_hat = p_hat
        self.leak_prob = leak_prob
        self.val_corr = val_corr
        self.noise_std = noise_std
        self.mode = mode
        self._generate()

    def _generate(self):
        n, d = self.n, self.dim

        # random binary features
        a = torch.bernoulli(0.5 * torch.ones(n, d))
        b = torch.bernoulli(0.5 * torch.ones(n, d))
        i = torch.bernoulli(torch.full((n, 1), self.p_hat))  # 1 = XOR, 0 = COPY

        xor_bit = (a[:,-2] != b[:,-2]).float()

        y = (xor_bit * i.squeeze(-1) +
             a[:,-2].float() * (1 - i.squeeze(-1))).long()

        # optional Gaussian noise
        if self.noise_std > 0:
            def noise(x): return (x + self.noise_std * torch.randn_like(x)).clamp(0, 1)
            a, b = noise(a), noise(b)

        a_idx = torch.arange(n)
        if self.mode == "train":
            leak_mask = torch.rand(n) < self.leak_prob
            a[a_idx[leak_mask], -1] = y[a_idx[leak_mask]].float()
        else:
            leak_mask = torch.rand(n) < self.val_corr
            a[a_idx[leak_mask], -1] = y[a_idx[leak_mask]].float()

        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "data": {0: self.a[idx], 1: self.b[idx]},
            "label": self.y[idx],
            "idx": idx,
        }
def get_dataloaders(batch_size=256, leak_prob=0.1, val_corr=0.5, noise_std=0.0):
    train_ds = NoisyBinaryXORDataset2(mode="train", leak_prob=leak_prob, val_corr=val_corr, noise_std=noise_std)
    val_ds   = NoisyBinaryXORDataset2(mode="val",   leak_prob=leak_prob, val_corr=val_corr, noise_std=noise_std)
    test_ds  = NoisyBinaryXORDataset2(mode="test",  leak_prob=leak_prob, val_corr=val_corr, noise_std=noise_std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# class BiModalFusionClassifier(nn.Module):
#     def __init__(self, l=0, num_classes=2, d_model=16):
#         super().__init__()
#         self.l = l
#         self.f_a = nn.Linear(16, d_model)
#         self.f_b = nn.Linear(16, d_model)
#         self.fc  = nn.Sequential(
#             nn.Linear(d_model * 2, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, num_classes)
#         )
#
#         self.comp_dim = 16
#         self.proj_z1 = nn.Conv1d(d_model, self.comp_dim, kernel_size=1, bias=False)
#         self.proj_z2 = nn.Conv1d(d_model, self.comp_dim, kernel_size=1, bias=False)
#
#         self.ln_z1 = nn.LayerNorm(self.comp_dim)
#         self.ln_z2 = nn.LayerNorm(self.comp_dim)
#
#     def forward(self, x, **kwargs):
#         feat_a = self.f_a(x[0])
#         feat_b = self.f_b(x[1])
#         joint_feat = torch.cat([feat_a, feat_b], dim=1)
#         pred = self.fc(joint_feat)
#
#         output = {"pred": pred}
#         if self.training and self.l!=0:
#             labels = kwargs["label"].view(-1, 1)
#
#             losses = self.z12z1mipd(feat_a, feat_b, labels)
#
#             output["losses"] = losses
#
#         return output
#     def z12z1mipd(self, z1_tokens, z2_tokens, labels):
#         def estimate_mipd(z1_tokens, z2_tokens, labels):
#             B = z1_tokens.size(0)
#             K = 16  # number of permutations
#
#             logits_pos1 = self.fc(torch.cat((z1_tokens, z2_tokens), dim=1))  # [B, C]
#             logp_pos1 = nn.LogSoftmax(dim=1)(logits_pos1).gather(1, labels).squeeze(1)
#
#             P = torch.argsort(torch.rand(K, B, device=z1_tokens.device), dim=1)
#             z2_perm = z2_tokens[P]
#             z1_rep = z1_tokens.unsqueeze(0).expand(K, -1, -1)
#
#
#             # flatten first two dims -> [K*B, T*, c]
#             z2_perm_flat = z2_perm.reshape(K * B, z2_perm.size(2))
#             z1_rep_flat = z1_rep.reshape(K * B, z1_rep.size(2))
#             labels_flat = labels.expand(K, -1, -1).reshape(K * B, 1)
#
#             # ---- permutation average E_pi[ log q( y | Z12^(pi), Z1 ) ] (vectorized) ----
#             logits_perm_flat = self.fc(torch.cat((z1_rep_flat, z2_perm_flat), dim=1))  # [K*B, C]
#             logp_perm_flat = nn.LogSoftmax(dim=1)(logits_perm_flat).gather(1, labels_flat).squeeze(1)
#             logp_perm_mean = logp_perm_flat.view(K, B).mean(0)  # [B]
#
#             # loss = E[ log q(y | Z12^pi, Z1) - log q(y | Z12, Z1) ]
#             comp = (logp_perm_mean - logp_pos1).mean()
#             return comp
#
#         comp1 = estimate_mipd(z1_tokens, z2_tokens, labels)
#         comp2 = estimate_mipd(z2_tokens, z1_tokens, labels)
#
#         losses = {"comp1": self.l * comp1, "comp2": self.l * comp2}
#
#         return losses
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


class BiModalFusionIBBA(nn.Module):


    def __init__(self, l, num_classes=2, d_model=16, comp_dim=16):
        super().__init__()
        self.l = l
        self.comp_dim = comp_dim

        self.f_a = nn.Linear(16, d_model)
        self.f_b = nn.Linear(16, d_model)

        self.backbone = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU()
        )

        self.mu_head = nn.Linear(d_model, num_classes)
        self.logvar_head = nn.Linear(d_model, num_classes)

        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("one", torch.tensor(1.0))

    def forward_once(self, x1, x2, stochastic=False, perturb_x1=False, perturb_x2=False):

        feat_a = self.f_a(x1)
        feat_b = self.f_b(x2)
        if perturb_x1:
            noise_a = torch.randn_like(feat_a) * 0.1
            feat_a = feat_a + noise_a

            #zero_like
            feat_a = torch.zeros_like(feat_a)
        if perturb_x2:
            noise_b = torch.randn_like(feat_b) * 0.1
            feat_b = feat_b + noise_b
            feat_b = torch.zeros_like(feat_b)

        joint_feat = self.backbone(torch.cat([feat_a, feat_b], dim=-1))

        mu = self.mu_head(joint_feat)
        logvar = self.logvar_head(joint_feat)

        if stochastic:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            logits = mu + std * eps
        else:
            logits = mu

        return logits, mu, logvar

    def forward(self, x, x1_tilde=None, label=None, **kwargs):

        x1 = x[0]
        x2 = x[1]

        out = {"losses":{}}

        # -------- clean pass: q_theta(y|x1,x2) --------
        logits_q, mu_q, logvar_q = self.forward_once(x1, x2, stochastic=False)
        out["pred"] = logits_q

        losses = {}

        if self.training:
            # BA term: negative log-likelihood (cross-entropy)
            # ba_loss = F.cross_entropy(logits_q, y)
            # losses["ba_loss"] = ba_loss

            logits_p, mu_p, logvar_p = self.forward_once( x1, x2, stochastic=True, perturb_x1=True, perturb_x2=False)
            kl_1 = 0.5 * torch.sum( torch.exp(logvar_p) + mu_p**2 - 1.0 - logvar_p, dim=1 ).mean()

            logits_p, mu_p, logvar_p = self.forward_once(x1, x2, stochastic=True, perturb_x1=False, perturb_x2=True)
            kl_2 = 0.5 * torch.sum(torch.exp(logvar_p) + mu_p ** 2 - 1.0 - logvar_p, dim=1).mean()

            losses["ib_kl_1"] = self.l * kl_1
            losses["ib_kl_2"] = self.l * kl_2

            out["losses"] = losses

        return out


def train_epoch(model, loader, criterion, opt, device):
    model.train()
    running_loss, correct = 0.0, 0
    for batch in loader:
        x = [batch["data"][0].to(device), batch["data"][1].to(device)]
        y = batch["label"].long().to(device)
        opt.zero_grad()
        output = model(x, label=y)
        loss = criterion(output["pred"], y)
        if "loss" in output:
            loss += output["loss"]
        correct += (output["pred"].argmax(1) == y).sum().item()
        loss.backward()
        opt.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            x = [batch["data"][0].to(device), batch["data"][1].to(device)]
            y = batch["label"].long().to(device)
            pred = model(x)["pred"]
            loss = criterion(pred, y)
            total_loss += loss.item() * y.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------
# 4. Run experiment
# ---------------------------------------------------------------------
def main(leak_prob=0.1, l=0, val_corr=0.5, noise_std=0.0, weight_decay=0.0, verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device: {device}")
        print(f"Training with leak_prob={leak_prob}, val_corr={val_corr}, "
              f"noise_std={noise_std}, weight_decay={weight_decay}")

    train_loader, val_loader, test_loader = get_dataloaders(
        leak_prob=leak_prob, val_corr=val_corr, noise_std=noise_std
    )

    batch = next(iter(train_loader))
    corr = torch.corrcoef(torch.stack([batch["data"][0][:, -1], batch["label"].float()]))[0, 1]
    if verbose:
        print("Corr(a[:,-1], y):", corr.item())

    model = BiModalFusionIBBA(l=l).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    best_loss, wait, patience = float("inf"), 0, 100
    best_state, best_metrics = None, (0, 0, 0, 0)

    for epoch in range(1, 1001):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, opt, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss, best_state = val_loss, model.state_dict()
            best_metrics = (train_loss, train_acc, val_loss, val_acc)
            wait = 0
        else:
            wait += 1

        if verbose:
            print(f"Epoch {epoch:04d} | train_loss={train_loss:.4f} | "
                  f"train_acc={train_acc*100:.2f}% | val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc*100:.2f}% | best_val_loss={best_loss:.4f}")

        if wait >= patience:
            if verbose:
                print(f"No improvement for {patience} epochs. Stopping early.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    if verbose:
        print(f"\nBest epoch results:")
        print(f"Train loss={best_metrics[0]:.4f} | Train acc={best_metrics[1]*100:.2f}%")
        print(f"Val   loss={best_metrics[2]:.4f} | Val   acc={best_metrics[3]*100:.2f}%")
        print(f"Test  loss={test_loss:.4f} | Test  acc={test_acc*100:.2f}%")
    else:
        print(f"Leak prob={leak_prob:.1f}% | "
              f"Train acc={best_metrics[1]*100:.2f}% | "
              f"Val acc={best_metrics[3]*100:.2f}% | "
              f"Test acc={test_acc*100:.2f}%")


# ---------------------------------------------------------------------
# 5. CLI entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-input synthetic XOR leak experiment")
    parser.add_argument("--leak_prob", type=float, default=0.1, help="Fraction of leaked samples in training set")
    parser.add_argument("--l", type=float, default=0)
    parser.add_argument("--val_corr", type=float, default=0.5, help="Correlation between leak and label in validation/test")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Stddev of Gaussian feature noise")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay for optimizer")
    parser.add_argument("--verbose", action="store_true", help="Print full training progress")

    args = parser.parse_args()
    deterministic(0)

    main(
        leak_prob=args.leak_prob,
        l=args.l,
        val_corr=args.val_corr,
        noise_std=args.noise_std,
        weight_decay=args.weight_decay,
        verbose=args.verbose,
    )
    deterministic(0)

    main(
        leak_prob=args.leak_prob,
        l=0,
        val_corr=args.val_corr,
        noise_std=args.noise_std,
        weight_decay=args.weight_decay,
        verbose=False,
    )
