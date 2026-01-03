import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class HighDimXORWithTrainSpurious(Dataset):
    """
    y = XOR(b0,b1) always.

    Mod0: small signal block for b0 + spurious block correlated with y in TRAIN only
    Mod1: small signal block for b1

    Spurious block is continuous Gaussian with corr(alpha) to y_sign in TRAIN:
        s = alpha*y_sign + sqrt(1-alpha^2)*eps
    TEST uses alpha_effective=0.
    """
    def __init__(
        self,
        n: int,
        dim0: int = 64,
        dim1: int = 64,
        signal_dims: int = 4,
        spur_dims: int = 6,          # <<<<<<<<<< reduce dims
        alpha: float = 0.0,
        mode: str = "train",
        seed: int = 0,
        # Make XOR slightly easier so alpha=0 is near-perfect:
        signal_scale: float = 2.2,   # <<<<<<<<<< increase
        signal_noise_std: float = 0.55,  # <<<<<<<<<< reduce
        # Make shortcut moderate (so it doesn’t dominate at alpha=0.1–0.2):
        spur_scale: float = 1.3,     # <<<<<<<<<< reduce
        base_noise_std: float = 1.0,
    ):
        super().__init__()
        assert mode in ["train", "test"]
        assert 0.0 <= alpha <= 1.0
        assert signal_dims + spur_dims <= dim0

        g = torch.Generator().manual_seed(seed)

        b0 = torch.randint(0, 2, (n,), generator=g).float()
        b1 = torch.randint(0, 2, (n,), generator=g).float()
        y = (b0 != b1).float()

        x0 = torch.randn(n, dim0, generator=g) * base_noise_std
        x1 = torch.randn(n, dim1, generator=g) * base_noise_std

        sig0 = slice(0, signal_dims)
        sig1 = slice(0, signal_dims)
        spur = slice(signal_dims, signal_dims + spur_dims)

        b0_sign = (2 * b0 - 1).unsqueeze(1)
        b1_sign = (2 * b1 - 1).unsqueeze(1)

        # true bits -> XOR
        x0[:, sig0] = signal_scale * b0_sign + torch.randn(n, signal_dims, generator=g) * signal_noise_std
        x1[:, sig1] = signal_scale * b1_sign + torch.randn(n, signal_dims, generator=g) * signal_noise_std

        # continuous spurious block
        y_sign = (2 * y - 1).unsqueeze(1)
        alpha_eff = alpha if mode == "train" else 0.0

        eps = torch.randn(n, spur_dims, generator=g)  # N(0,1)
        s = alpha_eff * y_sign + (1.0 - alpha_eff**2) ** 0.5 * eps  # corr(s, y_sign)=alpha_eff

        x0[:, spur] = spur_scale * s

        self.x0, self.x1, self.y = x0, x1, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"data": {0: self.x0[idx], 1: self.x1[idx]}, "y": self.y[idx]}


class ResearchModel(nn.Module):
    def __init__(self, dim0=64, dim1=64, hidden=16):
        super().__init__()
        self.enc0 = nn.Sequential(nn.Linear(dim0, hidden), nn.ReLU())
        self.enc1 = nn.Sequential(nn.Linear(dim1, hidden), nn.ReLU())

        self.clf0 = nn.Linear(hidden, 1)
        self.clf1 = nn.Linear(hidden, 1)

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        f0 = self.enc0(x[0])
        f1 = self.enc1(x[1])
        u0 = torch.sigmoid(self.clf0(f0)).squeeze(-1)
        u1 = torch.sigmoid(self.clf1(f1)).squeeze(-1)
        syn = torch.sigmoid(self.fusion(torch.cat([f0, f1], dim=1))).squeeze(-1)
        return syn, u0, u1

class ResearchModel_synib(nn.Module):
    """
    SynIB-lite model for the XOR toy:

    - Same architecture as your ResearchModel: two encoders + unimodal heads + fusion head.
    - Extra training signal is handled via `compute_losses(...)`:
        L_total = L_base + lambda_kl * ( KL(f(tilde_X1, X2) || 0.5) + KL(f(X1, tilde_X2) || 0.5) )
      where tilde_X keeps the spurious block but destroys the complementary block.
    """

    def __init__(self, dim0=64, dim1=64, hidden=16):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

        self.enc0 = nn.Sequential(nn.Linear(dim0, hidden), nn.ReLU())
        self.enc1 = nn.Sequential(nn.Linear(dim1, hidden), nn.ReLU())

        self.clf0 = nn.Linear(hidden, 1)  # predicts y from modality 0
        self.clf1 = nn.Linear(hidden, 1)  # predicts y from modality 1

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        x: dict {0: [B, dim0], 1: [B, dim1]}
        returns: (syn, u0, u1) in (0,1)
        """
        f0 = self.enc0(x[0])
        f1 = self.enc1(x[1])

        u0 = torch.sigmoid(self.clf0(f0)).squeeze(-1)
        u1 = torch.sigmoid(self.clf1(f1)).squeeze(-1)

        syn = torch.sigmoid(self.fusion(torch.cat([f0, f1], dim=1))).squeeze(-1)
        return syn, u0, u1

    # -------------------------
    # SynIB-lite helpers
    # -------------------------
    @staticmethod
    def kl_bern_to_uniform(p, eps: float = 1e-6):
        """
        KL(Bern(p) || Bern(0.5)) averaged over batch.
        KL = p log(2p) + (1-p) log(2(1-p))
        """
        p = p.clamp(eps, 1 - eps)
        return (p * torch.log(2 * p) + (1 - p) * torch.log(2 * (1 - p))).mean()

    @staticmethod
    def make_tilde_keep_spur_drop_comp(x, signal_dims: int, spur_dims: int, comp_noise_std: float = 1.0):
        """
        x shape: [B, D]
        We assume:
          - complementary block = dims [0:signal_dims]
          - spurious block      = dims [signal_dims:signal_dims+spur_dims]
        tilde keeps spurious as-is and replaces complementary with fresh noise.
        """
        x_t = x.clone()
        comp = slice(0, signal_dims)
        x_t[:, comp] = torch.randn_like(x_t[:, comp]) * comp_noise_std
        # spurious block is intentionally preserved
        return x_t

    def compute_losses(
        self,
        batch,
        *,
        # blocks for modality 0 (X1)
        signal_dims0: int,
        spur_dims0: int,
        # blocks for modality 1 (X2)
        signal_dims1: int,
        spur_dims1: int,
        # weights
        lambda_kl: float = 1.0,
        lambda_uni: float = 0.05,
        comp_noise_std: float = 1.0,
        add_uni_losses: bool = True,
    ):
        """
        Returns:
          total_loss, dict_of_terms

        Base:
          L_base = BCE( f(X1,X2), y )  (+ unimodal BCE probes if add_uni_losses)

        SynIB-lite:
          tilde_X1 = [noise on comp, keep spur]
          tilde_X2 = [noise on comp, keep spur]
          L_cf = KL( f(tilde_X1,X2) || 0.5 ) + KL( f(X1,tilde_X2) || 0.5 )
          L_total = L_base + lambda_kl * L_cf
        """
        y = batch["y"]
        x0 = batch["data"][0]
        x1 = batch["data"][1]

        syn, u0, u1 = self({0: x0, 1: x1})
        bce = F.binary_cross_entropy

        # Base supervised loss
        L_base = bce(syn, y)
        if add_uni_losses:
            L_base = L_base + lambda_uni * (bce(u0, y) + bce(u1, y))

        # Counterfactual inputs: keep spurious, destroy complementary
        x0_t = self.make_tilde_keep_spur_drop_comp(x0, signal_dims0, spur_dims0, comp_noise_std)
        x1_t = self.make_tilde_keep_spur_drop_comp(x1, signal_dims1, spur_dims1, comp_noise_std)

        syn_t0, _, _ = self({0: x0_t, 1: x1})
        syn_t1, _, _ = self({0: x0,   1: x1_t})

        # KL-to-uniform prior on the counterfactual predictions
        L_cf = self.kl_bern_to_uniform(syn_t0) + self.kl_bern_to_uniform(syn_t1)

        L_total = L_base + lambda_kl * L_cf

        terms = {
            "L_total": float(L_total.detach().cpu()),
            "L_base": float(L_base.detach().cpu()),
            "L_cf": float(L_cf.detach().cpu()),
        }
        return L_total, terms


@torch.no_grad()
def acc(p, y):
    return ((p > 0.5).float() == y).float().mean().item()


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    S, U0, U1 = [], [], []
    for batch in loader:
        y = batch["y"]
        syn, u0, u1 = model(batch["data"])
        S.append(acc(syn, y))
        U0.append(acc(u0, y))
        U1.append(acc(u1, y))
    return float(np.mean(S)), float(np.mean(U0)), float(np.mean(U1))


def run_one(
    alpha: float,
    seed: int,
    n_train: int = 2000,     # <<<<<<<<<< larger reduces “bimodal” across seeds
    n_test: int = 5000,
    epochs: int = 120,
    lr: float = 2e-3,
    **kwargs
):
    set_seed(seed)

    train_ds = HighDimXORWithTrainSpurious(n_train, alpha=alpha, mode="train", seed=seed + 101)
    test_ds  = HighDimXORWithTrainSpurious(n_test,  alpha=0.0,   mode="test",  seed=seed + 201)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    model = ResearchModel(hidden=16)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)  # <<<<<<<<<< more regularization
    bce = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            y = batch["y"]
            syn, u0, u1 = model(batch["data"])
            loss = bce(syn, y) + 0.05 * (bce(u0, y) + bce(u1, y))
            opt.zero_grad()
            loss.backward()
            opt.step()

    train_metrics = eval_model(model, train_loader)
    test_metrics = eval_model(model, test_loader)
    return train_metrics, test_metrics

def run_one_synib(
    alpha: float,
    seed: int,
    n_train: int = 2000,     # <<<<<<<<<< larger reduces “bimodal” across seeds
    n_test: int = 5000,
    epochs: int = 120,
    lr: float = 2e-3,
    **kwargs
):
    set_seed(seed)

    train_ds = HighDimXORWithTrainSpurious(n_train, alpha=alpha, mode="train", seed=seed + 101)
    test_ds  = HighDimXORWithTrainSpurious(n_test,  alpha=0.0,   mode="test",  seed=seed + 201)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    model = ResearchModel_synib(hidden=16)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)  # <<<<<<<<<< more regularization

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            y = batch["y"]
            loss, terms = model.compute_losses(
                batch,
                signal_dims0=4, spur_dims0=20,  # modality 0 blocks
                signal_dims1=4, spur_dims1=20,  # modality 1 blocks (set spur_dims1=0 if no spur block there)
                lambda_kl=1.0,
                lambda_uni=0.05,
                comp_noise_std=1.0,
                add_uni_losses=True,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    train_metrics = eval_model(model, train_loader)
    test_metrics = eval_model(model, test_loader)
    return train_metrics, test_metrics


def mean_std(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def run_many(alphas, seeds, **kwargs):
    out = {}
    for a in alphas:
        trS, tr0, tr1 = [], [], []
        teS, te0, te1 = [], [], []
        for s in seeds:
            print(f"Running alpha={a}, seed={s}...")
            if "synib" in kwargs and kwargs["synib"]:
                (tS, t0, t1), (eS, e0, e1) = run_one_synib(a, s, **kwargs)
            else:
                (tS, t0, t1), (eS, e0, e1) = run_one(a, s, **kwargs)
            trS.append(tS); tr0.append(t0); tr1.append(t1)
            teS.append(eS); te0.append(e0); te1.append(e1)

        out[a] = {
            "train_syn": mean_std(trS),
            "train_u0":  mean_std(tr0),
            "train_u1":  mean_std(tr1),
            "test_syn":  mean_std(teS),
            "test_u0":   mean_std(te0),
            "test_u1":   mean_std(te1),
        }
    return out

def make_plot(results_base, results_synib, save_path):
    import matplotlib
    matplotlib.use("Agg")  # MUST be before importing pyplot
    import matplotlib.pyplot as plt
    import numpy as np

    alphas_sorted = sorted(results_base.keys())

    base_mean = np.array([results_base[a]["test_syn"][0] for a in alphas_sorted], dtype=float)
    base_std  = np.array([results_base[a]["test_syn"][1] for a in alphas_sorted], dtype=float)

    syn_mean  = np.array([results_synib[a]["test_syn"][0] for a in alphas_sorted], dtype=float)
    syn_std   = np.array([results_synib[a]["test_syn"][1] for a in alphas_sorted], dtype=float)

    plt.figure()
    plt.plot(alphas_sorted, base_mean, marker="o", label="Baseline")
    plt.fill_between(alphas_sorted, base_mean - base_std, base_mean + base_std, alpha=0.2)

    plt.plot(alphas_sorted, syn_mean, marker="o", label="SynIB")
    plt.fill_between(alphas_sorted, syn_mean - syn_std, syn_mean + syn_std, alpha=0.2)

    plt.xlabel(
        r"$\alpha$ (spurious strength in train): $X_1=[X_1^{\mathrm{comp}}\ \|\ X_1^{\mathrm{spur}}]$"
        "\n"
        r"$\mathrm{corr}(X_1^{\mathrm{spur}},y)=\alpha$ (train), "
        r"$\mathrm{corr}(X_1^{\mathrm{spur}},y)=0$ (test)"
        "\n"
        r"$\alpha=0:\ X_1^{\mathrm{spur}}\perp y,\ \alpha=1:\ X_1^{\mathrm{spur}}=y$"
    )
    plt.ylabel("Multimodal fusion test accuracy")
    plt.ylim(0.4, 1.05)
    plt.title("Baseline vs SynIB (mean ± std over seeds)")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(save_path, dpi=200)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    alphas = [i / 10 for i in range(0, 11)]
    seeds = list(range(10))

    results = run_many(
        alphas, seeds,
        n_train=300, n_test=30000,
        epochs=100, lr=2e-3,
        synib=False
    )
    results_synib = run_many(
        alphas, seeds,
        n_train=300, n_test=30000,
        epochs=100, lr=2e-3,
        synib=True
    )

    def fmt(ms):
        m, s = ms
        return f"{m:6.3f}±{s:5.3f}"

    print("\nBASELINE")
    print(
        f"{'alpha':>5} || "
        f"{'train_syn':>14} {'train_u0':>14} {'train_u1':>14} || "
        f"{'test_syn':>14} {'test_u0':>14} {'test_u1':>14}"
    )
    print("-" * 100)
    for a in alphas:
        r = results[a]
        print(
            f"{a:5.2f} || "
            f"{fmt(r['train_syn']):>14} {fmt(r['train_u0']):>14} {fmt(r['train_u1']):>14} || "
            f"{fmt(r['test_syn']):>14} {fmt(r['test_u0']):>14} {fmt(r['test_u1']):>14}"
        )

    print("\nSYNIB")
    print(
        f"{'alpha':>5} || "
        f"{'train_syn':>14} {'train_u0':>14} {'train_u1':>14} || "
        f"{'test_syn':>14} {'test_u0':>14} {'test_u1':>14}"
    )
    print("-" * 100)
    for a in alphas:
        r = results_synib[a]
        print(
            f"{a:5.2f} || "
            f"{fmt(r['train_syn']):>14} {fmt(r['train_u0']):>14} {fmt(r['train_u1']):>14} || "
            f"{fmt(r['test_syn']):>14} {fmt(r['test_u0']):>14} {fmt(r['test_u1']):>14}"
        )

    print("\nCOMPARISON (test synergy only)")
    print(f"{'alpha':>5} || {'baseline':>14} || {'synib':>14}")
    print("-" * 42)
    for a in alphas:
        b = results[a]["test_syn"]
        s = results_synib[a]["test_syn"]
        print(f"{a:5.2f} || {fmt(b):>14} || {fmt(s):>14}")

    make_plot(results, results_synib, "test_accuracy_vs_alpha_compare.png")
