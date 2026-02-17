import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse
from utils.deterministic_pytorch import deterministic

# class NoisyBinaryXORDataset(Dataset):
#
#     def __init__(self, n_points=5000, dim=16, p_hat=0.5,
#                  leak_prob=0.1, val_corr=0.5, noise_std=0.0, mode="train"):
#         super().__init__()
#         self.n = n_points
#         self.dim = dim
#         self.p_hat = p_hat
#         self.leak_prob = leak_prob
#         self.val_corr = val_corr
#         self.noise_std = noise_std
#         self.mode = mode
#         self._generate()
#
#     def _generate(self):
#         n, d = self.n, self.dim
#         a = torch.bernoulli(0.5 * torch.ones(n, d))
#         b = torch.bernoulli(0.5 * torch.ones(n, d))
#         i = torch.bernoulli(torch.full((n, 1), self.p_hat))
#         xor_part = (a != b).float()
#         c = xor_part * i + a * (1 - i)
#         y = i.squeeze(-1).long()
#
#         # --- feature noise ---
#         if self.noise_std > 0:
#             noise = lambda x: (x + self.noise_std * torch.randn_like(x)).clamp(0, 1)
#             a, b, c = noise(a), noise(b), noise(c)
#
#         # --- leakage control ---
#         if self.mode == "train":
#             leak_mask = torch.rand(n) < self.leak_prob
#             a[leak_mask, -1] = y[leak_mask].float()
#             a[~leak_mask, -1] = torch.bernoulli(0.5 * torch.ones((~leak_mask).sum()))
#         else:
#             flip_mask = torch.rand(n) < (1 - self.val_corr)
#             a[:, -1] = y.float()
#             a[flip_mask, -1] = 1 - a[flip_mask, -1]
#
#         self.a, self.b, self.c, self.y = a, b, c, y
#
#     def __len__(self):
#         return self.n
#
#     def __getitem__(self, idx):
#         return {
#             "data": {0: self.a[idx], 1: self.b[idx], 2: self.c[idx]},
#             "label": self.y[idx],
#             "idx": idx,
#         }

class NoisyBinaryXORDataset2(Dataset):
    """
    Each sample: (a,b)
      with prob p_hat -> label = XOR(a,b)  (synergy case)
      else            -> label = a[:, -1]  (copy case)
    Adds partial label leak in a[:, -1], and optional Gaussian feature noise.
    """

    def __init__(self, n_points=5000, dim=16, p_hat=1.0,
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
        a = torch.bernoulli(0.5 * torch.ones(n, d))
        b = torch.bernoulli(0.5 * torch.ones(n, d))

        # XOR indicator variable (which rule applies)
        i = torch.bernoulli(torch.full((n, 1), self.p_hat))

        # label: y = XOR(a,b) if i=1 else a
        xor_out = (a != b).float()
        y = (xor_out[:, -1] * i.squeeze(-1) + a[:, -1] * (1 - i.squeeze(-1))).long()

        # --- feature noise ---
        if self.noise_std > 0:
            noise = lambda x: (x + self.noise_std * torch.randn_like(x)).clamp(0, 1)
            a, b = noise(a), noise(b)

        # --- leakage control ---
        if self.mode == "train" and self.leak_prob !=0 :
            leak_mask = torch.rand(n) < self.leak_prob
            a[leak_mask, -1] = y[leak_mask].float()
            a[~leak_mask, -1] = torch.bernoulli(0.5 * torch.ones((~leak_mask).sum()))
        else:
            flip_mask = torch.rand(n) < (1 - self.val_corr)
            a[:, -1] = y.float()
            a[flip_mask, -1] = torch.bernoulli(0.5 * torch.ones(flip_mask.sum()))

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


class TriModalFusionClassifier(nn.Module):
    def __init__(self, num_classes=2, d_model=16):
        super().__init__()
        self.f_a = nn.Linear(16, d_model)
        self.f_b = nn.Linear(16, d_model)
        self.f_c = nn.Linear(16, d_model)
        self.fc  = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):
        feat_a = self.f_a(x[0])
        feat_b = self.f_b(x[1])
        feat_c = self.f_c(x[2])
        joint_feat = torch.cat([feat_a, feat_b, feat_c], dim=1)
        pred = self.fc(joint_feat)
        output = {"pred": pred}
        if self.training and self.l!=0:
            labels = kwargs["label"].view(-1, 1)

            z1_tokens = feat_a
            z2_tokens = feat_b

            if self.regby == "z12z1_z1":
                losses = self.z12z1_z1_gap(z1_tokens, z2_tokens, z12_tokens, labels)
            elif self.regby == "z12z1mipd":
                losses = self.z12z1mipd(z1_tokens, z2_tokens, z12_tokens, labels)

            output["losses"] = losses

        return output

    def z12z1_z1_gap(self, z1_tokens, z2_tokens, z12_tokens, labels):
        # project to compact comp_dim and norm
        z1_tokens = self.ln_z1(self.proj_z1(z1_tokens).transpose(1, 2))  # [B,T1,c]
        z2_tokens = self.ln_z2(self.proj_z2(z2_tokens).transpose(1, 2))  # [B,T2,c]
        z12_tokens = self.ln_z12(self.proj_z12(z12_tokens.transpose(1, 2)).transpose(1, 2))  # [B,c]

        # --- shared heads ---
        # q_theta(. | Z12, Z1), q_eta(. | Z1)
        logits_theta1 = self.comp_head_shared(torch.cat((z12_tokens, z1_tokens), dim=1))  # [B,C]
        logits_eta1 = self.base_head_shared(z1_tokens)
        logits_eta1 = logits_eta1.detach()  # stop-grad baseline

        logp_theta1 = F.log_softmax(logits_theta1, dim=1).gather(1, labels).squeeze(1)
        logp_eta1 = F.log_softmax(logits_eta1, dim=1).gather(1, labels).squeeze(1)
        # comp1 = F.relu(-logp_theta1 + logp_eta1 + self.comp_gamma).mean()
        comp1 = (-logp_theta1 + logp_eta1).mean()

        losses = {"comp1": self.l * comp1}

        if self.comp_sym:
            # reuse the SAME heads for Z2 branch (parameter sharing)
            logits_theta2 = self.comp_head_shared(torch.cat((z12_tokens, z2_tokens), dim=1))
            logits_eta2 = self.base_head_shared(z2_tokens).detach()

            logp_theta2 = F.log_softmax(logits_theta2, dim=1).gather(1, labels).squeeze(1)
            logp_eta2 = F.log_softmax(logits_eta2, dim=1).gather(1, labels).squeeze(1)
            # comp2 = F.relu(-logp_theta2 + logp_eta2 + self.comp_gamma).mean()
            comp2 = (-logp_theta2 + logp_eta2).mean()
            losses["comp2"] = self.l * comp2
        return losses

    def z12z1mipd(self, z1_tokens, z2_tokens, z12_tokens, labels):
        # ---- project to compact comp_dim and normalize ----
        # inputs expected as [B, F, S] -> proj -> [B, S, c]
        z1_tokens = self.ln_z1(self.proj_z1(z1_tokens).transpose(1, 2))  # [B, T1, c]
        z2_tokens = self.ln_z2(self.proj_z2(z2_tokens).transpose(1, 2))  # [B, T2, c]
        z12_tokens = self.proj_z12(z12_tokens.transpose(1, 2)).transpose(1, 2)  # [B, T12, c] or [B, c]
        z12_tokens = self.ln_z12(z12_tokens)
        if z12_tokens.dim() == 2:  # make sure z12 has time axis
            z12_tokens = z12_tokens.unsqueeze(1)  # [B, 1, c]

        B = z1_tokens.size(0)
        K = getattr(self, "perm_K", self.args.get("perm_K", 4))  # number of permutations

        # ---- positive (clean) log-prob with concatenated tokens ----
        logits_pos1 = self.comp_head_shared(torch.cat((z12_tokens, z1_tokens), dim=1))  # [B, C]
        logp_pos1 = F.log_softmax(logits_pos1, dim=1).gather(1, labels).squeeze(1)

        # ---- build K permutations in a vectorized way (no Python loop over K) ----
        # P: [K, B] each row a permutation of range(B)
        P = torch.argsort(torch.rand(K, B, device=z1_tokens.device), dim=1)  # sampled perms
        # z12_perm: [K, B, T12, c]
        z12_perm = z12_tokens[P]
        # repeat z1 along batch K times to match: [K, B, T1, c]
        z1_rep = z1_tokens.unsqueeze(0).expand(K, -1, -1, -1)
        # flatten first two dims -> [K*B, T*, c]
        z12_perm_flat = z12_perm.reshape(K * B, z12_perm.size(2), z12_perm.size(3))
        z1_rep_flat = z1_rep.reshape(K * B, z1_rep.size(2), z1_rep.size(3))
        labels_flat = labels.expand(K, -1, -1).reshape(K * B, 1)

        # ---- permutation average E_pi[ log q( y | Z12^(pi), Z1 ) ] (vectorized) ----
        logits_perm_flat = self.comp_head_shared(torch.cat((z12_perm_flat, z1_rep_flat), dim=1))  # [K*B, C]
        logp_perm_flat = F.log_softmax(logits_perm_flat, dim=1).gather(1, labels_flat).squeeze(1)
        logp_perm_mean = logp_perm_flat.view(K, B).mean(0)  # [B]

        # loss = E[ log q(y | Z12^pi, Z1) - log q(y | Z12, Z1) ]
        comp1 = (logp_perm_mean - logp_pos1).mean()
        losses = {"comp1": self.l * comp1}

        if self.comp_sym:
            # Positive with Z2
            logits_pos2 = self.comp_head_shared(torch.cat((z12_tokens, z2_tokens), dim=1))
            logp_pos2 = F.log_softmax(logits_pos2, dim=1).gather(1, labels).squeeze(1)

            # Vectorized permutations for Z2 branch
            z2_rep = z2_tokens.unsqueeze(0).expand(K, -1, -1, -1)  # [K,B,T2,c]
            z2_rep_flat = z2_rep.reshape(K * B, z2_rep.size(2), z2_rep.size(3))
            logits_perm2_flat = self.comp_head_shared(torch.cat((z12_perm_flat, z2_rep_flat), dim=1))
            logp_perm2_flat = F.log_softmax(logits_perm2_flat, dim=1).gather(1, labels_flat).squeeze(1)
            logp_perm2_mean = logp_perm2_flat.view(K, B).mean(0)

            comp2 = (logp_perm2_mean - logp_pos2).mean()
            losses["comp2"] = self.l * comp2

        return losses


def train_epoch(model, loader, criterion, opt, device):
    model.train()
    running_loss, correct = 0.0, 0
    for batch in loader:
        x = [batch["data"][0].to(device), batch["data"][1].to(device), batch["data"][2].to(device)]
        y = batch["label"].long().to(device)
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        correct += (pred.argmax(1) == y).sum().item()
        loss.backward()
        opt.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            x = [batch["data"][0].to(device), batch["data"][1].to(device), batch["data"][2].to(device)]
            y = batch["label"].long().to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * y.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------
# 4. Run experiment
# ---------------------------------------------------------------------
def main(leak_prob=0.1, val_corr=0.5, noise_std=0.0, weight_decay=0.0, verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device: {device}")
        print(f"Training with leak_prob={leak_prob}, val_corr={val_corr}, "
              f"noise_std={noise_std}, weight_decay={weight_decay}")

    train_loader, val_loader, test_loader = get_dataloaders(
        leak_prob=leak_prob, val_corr=val_corr, noise_std=noise_std
    )

    batch = next(iter(train_loader))
    corr = torch.corrcoef(torch.stack([batch["data"][0][:,-1], batch["label"].float()]))[0,1]
    if verbose:
        print("Corr(a[:,-1], y):", corr.item())

    model = TriModalFusionClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    best_loss, wait, patience = float("inf"), 0, 100
    best_state, best_metrics = None, (0,0,0,0)

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
              f"Train loss={train_loss:.5f} | "
              f"Train acc={train_acc*100:.2f}% | "
              f"BTrain acc={best_metrics[1]*100:.2f}% | "
              f"Val acc={best_metrics[3]*100:.2f}% | "
              f"Test acc={test_acc*100:.2f}%")


# ---------------------------------------------------------------------
# 5. Command-line entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic multimodal XOR leak experiment")
    parser.add_argument("--leak_prob", type=float, default=0.1, help="Fraction of leaked samples in training set")
    parser.add_argument("--val_corr", type=float, default=0.5, help="Correlation between leak and label in validation/test")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Stddev of Gaussian feature noise")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay for optimizer")
    parser.add_argument("--verbose", action="store_true", help="Print full training progress")

    args = parser.parse_args()
    if args.leak_prob == 0:
        print("Arguments:", args)
    deterministic(0)
    main(
        leak_prob=args.leak_prob,
        val_corr=args.val_corr,
        noise_std=args.noise_std,
        weight_decay=args.weight_decay,
        verbose=args.verbose,
    )
