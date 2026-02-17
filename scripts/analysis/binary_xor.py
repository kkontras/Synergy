import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

def rand_bits(n, d, device=None):
    if d == 0:
        return torch.zeros((n, 0), device=device)
    return torch.randint(0, 2, (n, d), device=device).float()

def xor(a, b):
    return (a + b) % 2.0

def hash_bits(bits, out_d, seed):
    if bits.shape[1] == 0 or out_d == 0:
        return torch.zeros((bits.shape[0], 0), device=bits.device)
    g = torch.Generator(device=bits.device)
    g.manual_seed(seed)
    W = torch.randint(0, 2, (bits.shape[1], out_d), generator=g, device=bits.device).float()
    return (bits @ W) % 2.0

class ThreeViewPIDDatasetEasy(Dataset):
    def __init__(self, n, atom_dims, nuisance_dims=None, device=None):
        super().__init__()
        self.n = n
        self.atom_dims = {k: int(v) for k, v in atom_dims.items()}
        self.keys = ["u1","u2","u3","r12","r13","r23","r123","s12","s13","s23","s123"]
        if nuisance_dims is None:
            nuisance_dims = {"m12": 2, "m13": 2, "m23": 2, "a123": 2, "b123": 2}
        self.nuisance_dims = {k: int(v) for k, v in nuisance_dims.items()}

        self.y = {k: rand_bits(n, self.atom_dims.get(k, 0), device) for k in self.keys}

        m12 = rand_bits(n, self.nuisance_dims["m12"], device)
        m13 = rand_bits(n, self.nuisance_dims["m13"], device)
        m23 = rand_bits(n, self.nuisance_dims["m23"], device)
        a123 = rand_bits(n, self.nuisance_dims["a123"], device)
        b123 = rand_bits(n, self.nuisance_dims["b123"], device)

        s12_h = hash_bits(self.y["s12"], m12.shape[1], seed=12)
        s13_h = hash_bits(self.y["s13"], m13.shape[1], seed=13)
        s23_h = hash_bits(self.y["s23"], m23.shape[1], seed=23)
        s123_h = hash_bits(self.y["s123"], a123.shape[1], seed=123)

        c123 = xor(xor(a123, b123), s123_h)

        x1_fields = [self.y["u1"], self.y["r12"], self.y["r13"], self.y["r123"], m12, m13, a123]
        x2_fields = [self.y["u2"], self.y["r12"], self.y["r23"], self.y["r123"], xor(m12, s12_h), m23, b123]
        x3_fields = [self.y["u3"], self.y["r13"], self.y["r23"], self.y["r123"], xor(m13, s13_h), xor(m23, s23_h), c123]

        self.x1 = torch.cat([t for t in x1_fields if t.numel() > 0], dim=1)
        self.x2 = torch.cat([t for t in x2_fields if t.numel() > 0], dim=1)
        self.x3 = torch.cat([t for t in x3_fields if t.numel() > 0], dim=1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        y_item = {k: self.y[k][idx] for k in self.keys}
        return self.x1[idx], self.x2[idx], self.x3[idx], y_item

class MLP(nn.Module):
    def __init__(self, indim, hiddim, outdim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, hiddim),
            nn.ReLU(),
            nn.Linear(hiddim, outdim),
        )
    def forward(self, x):
        return self.net(x)

class ThreeViewCL(nn.Module):
    def __init__(self, d1, d2, d3, embed_dim=64, hidden=128):
        super().__init__()
        self.e1 = MLP(d1, hidden, hidden)
        self.e2 = MLP(d2, hidden, hidden)
        self.e3 = MLP(d3, hidden, hidden)
        self.p1 = nn.Linear(hidden, embed_dim)
        self.p2 = nn.Linear(hidden, embed_dim)
        self.p3 = nn.Linear(hidden, embed_dim)
        self.f12 = MLP(hidden * 2, hidden, embed_dim)
        self.f13 = MLP(hidden * 2, hidden, embed_dim)
        self.f23 = MLP(hidden * 2, hidden, embed_dim)

    def forward(self, x1, x2, x3):
        z1 = self.e1(x1)
        z2 = self.e2(x2)
        z3 = self.e3(x3)
        p1 = F.normalize(self.p1(z1), dim=-1)
        p2 = F.normalize(self.p2(z2), dim=-1)
        p3 = F.normalize(self.p3(z3), dim=-1)
        f12 = F.normalize(self.f12(torch.cat([z1, z2], dim=-1)), dim=-1)
        f13 = F.normalize(self.f13(torch.cat([z1, z3], dim=-1)), dim=-1)
        f23 = F.normalize(self.f23(torch.cat([z2, z3], dim=-1)), dim=-1)
        return p1, p2, p3, f12, f13, f23

def info_nce(a, b, temp=0.1):
    logits = a @ b.T / temp
    labels = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

def loss_pairwise_only(p1, p2, p3, temp=0.1):
    return (info_nce(p1, p2, temp) + info_nce(p2, p3, temp) + info_nce(p3, p1, temp)) / 3

def loss_fused_only(p1, p2, p3, f12, f13, f23, temp=0.1):
    return (info_nce(f12, p3, temp) + info_nce(f13, p2, temp) + info_nce(f23, p1, temp)) / 3

def loss_pairwise_plus_fused(p1, p2, p3, f12, f13, f23, lambda_=0.5, temp=0.1):
    return lambda_ * loss_fused_only(p1,p2,p3,f12,f13,f23,temp) + (1 - lambda_) * loss_pairwise_only(p1,p2,p3,temp)

def mi_from_bit_error(err):
    err = float(np.clip(err, 1e-9, 1 - 1e-9))
    h2 = -(err * math.log(err, 2) + (1 - err) * math.log(1 - err, 2))
    return 1 - h2

def collect_embeddings(model, loader, device):
    model.eval()
    Z = {k: [] for k in ["z1","z2","z3","z12","z13","z23","z123"]}
    Y = None
    with torch.no_grad():
        for x1, x2, x3, y in loader:
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
            p1, p2, p3, f12, f13, f23 = model(x1, x2, x3)
            Z["z1"].append(p1); Z["z2"].append(p2); Z["z3"].append(p3)
            Z["z12"].append(f12); Z["z13"].append(f13); Z["z23"].append(f23)
            Z["z123"].append(torch.cat([p1, p2, p3], dim=1))
            if Y is None:
                Y = {k: [v.to(device)] for k, v in y.items()}
            else:
                for k, v in y.items():
                    Y[k].append(v.to(device))
    for k in Z: Z[k] = torch.cat(Z[k])
    for k in Y: Y[k] = torch.cat(Y[k])
    return Z, Y

def precompute_ridge_map(Xtr, l2=1e-1):
    Xtrb = torch.cat([Xtr, torch.ones((Xtr.shape[0], 1), device=Xtr.device)], dim=1)
    XtX = Xtrb.T @ Xtrb
    I = torch.eye(XtX.shape[0], device=Xtr.device)
    return torch.linalg.solve(XtX + l2 * I, Xtrb.T)

def ridge_decode_mi(Map, Xte, Ytr, Yte):
    Xteb = torch.cat([Xte, torch.ones((Xte.shape[0], 1), device=Xte.device)], dim=1)
    W = Map @ Ytr
    logits = Xteb @ W
    preds = (torch.sigmoid(logits) > 0.5).float()
    bit_acc = (preds == Yte).float().mean(dim=0)
    mi = sum(mi_from_bit_error(1 - a.item()) for a in bit_acc)
    return float(mi), float(bit_acc.mean().item())

def possible_subset_for_atom(atom):
    if atom == "u1": return "x1"
    if atom == "u2": return "x2"
    if atom == "u3": return "x3"
    if atom == "r12": return "x1 or x2"
    if atom == "r13": return "x1 or x3"
    if atom == "r23": return "x2 or x3"
    if atom == "r123": return "x1 or x2 or x3"
    if atom == "s12": return "x1+x2"
    if atom == "s13": return "x1+x3"
    if atom == "s23": return "x2+x3"
    if atom == "s123": return "x1+x2+x3"
    raise ValueError(atom)

def run_experiment(tr_loader, te_loader, d1, d2, d3, atom_dims, device, mode, lambda_=0.5, epochs=30, lr=1e-4):
    model = ThreeViewCL(d1, d2, d3, embed_dim=64, hidden=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        for x1, x2, x3, _y in tr_loader:
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
            p1, p2, p3, f12, f13, f23 = model(x1, x2, x3)
            if mode == "pairwise_only":
                loss = loss_pairwise_only(p1, p2, p3)
            elif mode == "pairwise_plus_fused":
                loss = loss_pairwise_plus_fused(p1, p2, p3, f12, f13, f23, lambda_=lambda_)
            elif mode == "fused_only":
                loss = loss_fused_only(p1, p2, p3, f12, f13, f23)
            else:
                raise ValueError(mode)
            opt.zero_grad()
            loss.backward()
            opt.step()

    Ztr, Ytr = collect_embeddings(model, tr_loader, device)
    Zte, Yte = collect_embeddings(model, te_loader, device)

    sources = ["z1","z2","z3","z12","z13","z23","z123"]
    maps = {s: precompute_ridge_map(Ztr[s], l2=1e-1) for s in sources}

    rows = []
    for atom in atom_dims.keys():
        mi_scores = {}
        acc_scores = {}
        for s in sources:
            mi, acc = ridge_decode_mi(maps[s], Zte[s], Ytr[atom], Yte[atom])
            mi_scores[s] = mi
            acc_scores[s] = acc
        best = max(mi_scores, key=mi_scores.get)
        mi_possible = float(atom_dims[atom])
        mi_rec = mi_scores[best]
        frac = mi_rec / mi_possible if mi_possible > 0 else float("nan")
        rows.append({
            "mode": mode if mode != "pairwise_plus_fused" else f"{mode}(lambda={lambda_})",
            "atom": atom,
            "possible_subset": possible_subset_for_atom(atom),
            "MI_possible": mi_possible,
            "best_embedding": best,
            "MI_recovered": mi_rec,
            "fraction_recovered": frac,
            "best_bit_acc": acc_scores[best],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    atom_dims = {k: 8 for k in ["u1","u2","u3","r12","r13","r23","s12","s13","s23","r123","s123"]}
    nuisance_dims = {"m12": 2, "m13": 2, "m23": 2, "a123": 2, "b123": 2}

    ds = ThreeViewPIDDatasetEasy(n=20000, atom_dims=atom_dims, nuisance_dims=nuisance_dims)
    d1, d2, d3 = ds.x1.shape[1], ds.x2.shape[1], ds.x3.shape[1]

    tr, te = random_split(ds, [16000, 4000], generator=torch.Generator().manual_seed(0))
    tr_loader = DataLoader(tr, batch_size=512, shuffle=True, num_workers=0)
    te_loader = DataLoader(te, batch_size=512, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_a = run_experiment(tr_loader, te_loader, d1, d2, d3, atom_dims, device, mode="pairwise_only", epochs=30, lr=1e-4)
    df_b = run_experiment(tr_loader, te_loader, d1, d2, d3, atom_dims, device, mode="pairwise_plus_fused", lambda_=0.5, epochs=30, lr=1e-4)

    df = pd.concat([df_a, df_b], axis=0)

    view = df.pivot_table(index=["atom","possible_subset","MI_possible"],
                          columns="mode",
                          values=["MI_recovered","fraction_recovered","best_bit_acc","best_embedding"],
                          aggfunc="first")

    pd.set_option("display.max_columns", None)
    print(view.sort_index().to_string())
