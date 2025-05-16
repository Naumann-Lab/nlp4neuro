#!/usr/bin/env python3
import os, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, GPT2Model, BertModel, BitsAndBytesConfig
from torch.optim import AdamW
from scipy.stats import wilcoxon
from tqdm import tqdm

RESULTS_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment3b_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("▶ Device:", device)

fish_num   = 9
DATA_DIR   = "/hpc/group/naumannlab/jjm132/data_prepped_for_models"
X_full_raw = np.load(f"{DATA_DIR}/fish{fish_num}_neural_data_matched.npy", allow_pickle=True)[:, :-2]
Y_full_raw = np.load(f"{DATA_DIR}/fish{fish_num}_tail_data_matched.npy", allow_pickle=True)

X_full_raw = X_full_raw.T
assert X_full_raw.shape[0] == Y_full_raw.shape[0]

T      = X_full_raw.shape[0]
tr_end = int(0.70 * T)
va_end = int(0.80 * T)

X_tr, X_va, X_te = X_full_raw[:tr_end], X_full_raw[tr_end:va_end], X_full_raw[va_end:]
Y_tr, Y_va, Y_te = Y_full_raw[:tr_end], Y_full_raw[tr_end:va_end], Y_full_raw[va_end:]
out_dim          = Y_tr.shape[1]

np.save(os.path.join(RESULTS_DIR, "groundtruth_val.npy"),  Y_va)
np.save(os.path.join(RESULTS_DIR, "groundtruth_test.npy"), Y_te)
print("▶ Ground truths saved.")

SALIENCY_ROOT = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/deepseek_only"
imp_path      = f"{SALIENCY_ROOT}/fish{fish_num}/fish{fish_num}_importance.npy"
precomp_imp   = np.load(imp_path)
SALIENT50_IDX = precomp_imp.argsort()[-50:][::-1]
print("▶ Loaded top-50 salient neuron list.")

def create_sequences(x, y, L):
    xs, ys = [], []
    for i in range(len(x) - L + 1):
        xs.append(x[i:i+L])
        ys.append(y[i:i+L])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

def overlap_mean(preds, L, total_len):
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    out = np.zeros((total_len, preds.shape[2]))
    cnt = np.zeros(total_len)
    for i in range(preds.shape[0]):
        out[i:i+L] += preds[i]
        cnt[i:i+L] += 1
    return out / cnt[:, None]

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def feature_importance(model, loader, in_dim):
    model.eval()
    imp = torch.zeros(in_dim, device=device)
    n = 0
    for xb, _ in loader:
        xb = xb.to(device).requires_grad_(True)
        model(xb).sum().backward()
        imp += (xb.grad * xb).abs().mean((0,1)).detach()
        n += 1
    return (imp / n).cpu().numpy()

def train(model, opt, tr_loader, va_loader, epochs=3, use_pos_ids=False):
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        tloss = 0.0
        for xb, yb in tqdm(tr_loader, desc=f"Epoch {ep+1}/{epochs}[train]", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pos_ids = None
            if use_pos_ids:
                B,L,_ = xb.shape
                pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B,L)
            yhat = model(xb, position_ids=pos_ids) if pos_ids is not None else model(xb)
            loss = crit(yhat, yb)
            loss.backward()
            opt.step()
            tloss += loss.item()
        vloss = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in tqdm(va_loader, desc=f"Epoch {ep+1}/{epochs}[val]", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                pos_ids = None
                if use_pos_ids:
                    B,L,_ = xb.shape
                    pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B,L)
                yhat = model(xb, position_ids=pos_ids) if pos_ids is not None else model(xb)
                vloss += crit(yhat, yb).item()
        print(f"    ↳ train RMSE {math.sqrt(tloss/len(tr_loader)):.4f} | val RMSE {math.sqrt(vloss/len(va_loader)):.4f}")

def predict(model, loader, use_pos_ids=False):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc="Predicting", leave=False):
            xb = xb.to(device)
            pos_ids = None
            if use_pos_ids:
                B,L,_ = xb.shape
                pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B,L)
            yhat = model(xb, position_ids=pos_ids) if pos_ids is not None else model(xb)
            preds.append(yhat.cpu())
    return torch.cat(preds, 0)

class VanillaLinear(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__()
        self.l = nn.Linear(in_dim, h)
    def forward(self, x):
        return self.l(x)

class PositionalLinear(nn.Module):
    def __init__(self, in_dim, h, max_len=1024):
        super().__init__()
        self.l = nn.Linear(in_dim, h)
        pe = torch.zeros(max_len, h)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, h, 2) * (-math.log(10000.0) / h))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        z = self.l(x)
        return z + self.pe[:, :z.size(1)]

class RelPosLinear(nn.Module):
    def __init__(self, in_dim, h, max_dist=32):
        super().__init__()
        self.l = nn.Linear(in_dim, h)
        self.md = max_dist
        self.rel = nn.Embedding(2*max_dist+1, h)
    def forward(self, x):
        z = self.l(x)
        B,L,_ = z.shape
        offs = torch.arange(L, device=z.device) - torch.arange(L, device=z.device)[0]
        offs = torch.clamp(offs, -self.md, self.md) + self.md
        return z + self.rel(offs).unsqueeze(0).expand(B,L,-1)

class SparseAE(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__()
        self.enc = nn.Linear(in_dim, h)
    def forward(self, x):
        B,L,_ = x.shape
        return torch.relu(self.enc(x.view(B*L, -1))).view(B,L,-1)

class Spectral(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__()
        self.l = nn.Linear(in_dim, h)
        nn.init.orthogonal_(self.l.weight)
    def forward(self, x):
        return torch.tanh(self.l(x))

embed_dict = {
    "Vanilla": VanillaLinear,
    "Positional": PositionalLinear,
    "RelativePos": RelPosLinear,
    "Sparse": SparseAE,
    "Spectral": Spectral,
}

quant_cfg = BitsAndBytesConfig(load_in_4bit=True, llm_int8_threshold=6.0)
H_DS, H_HF = 4096, 768

class DeepSeek(nn.Module):
    def __init__(self, embed, out_dim):
        super().__init__()
        self.e = embed
        self.m = AutoModel.from_pretrained("deepseek-ai/deepseek-coder-7b",
                                           trust_remote_code=True,
                                           device_map="auto",
                                           quantization_config=quant_cfg)
        for p in self.m.parameters():
            p.requires_grad = False
        self.out = nn.Linear(H_DS, out_dim)
    def forward(self, x, **kw):
        return self.out(self.m(inputs_embeds=self.e(x)).last_hidden_state)

class GPT2(nn.Module):
    def __init__(self, embed, out_dim):
        super().__init__()
        self.e = embed
        self.t = GPT2Model.from_pretrained("gpt2")
        self.out = nn.Linear(H_HF, out_dim)
        for p in self.t.parameters():
            p.requires_grad = False
    def forward(self, x, position_ids=None, **kw):
        return self.out(self.t(inputs_embeds=self.e(x), position_ids=position_ids).last_hidden_state)

class BERT(nn.Module):
    def __init__(self, embed, out_dim):
        super().__init__()
        self.e = embed
        self.t = BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(H_HF, out_dim)
        for p in self.t.parameters():
            p.requires_grad = False
    def forward(self, x, **kw):
        return self.out(self.t(inputs_embeds=self.e(x)).last_hidden_state)

families = {
    "DeepSeek": {"cls": DeepSeek, "h": H_DS, "use_pos": False},
    "GPT2":     {"cls": GPT2,     "h": H_HF, "use_pos": True},
    "BERT":     {"cls": BERT,     "h": H_HF, "use_pos": False},
}

SEQ   = 20
EPOCH = 10
BS    = 32
LR    = 1e-4
RUNS  = 10

results_sal = {f:{e:[] for e in embed_dict} for f in families}
results_rnd = {f:{e:[] for e in embed_dict} for f in families}

for scenario in ("salient50_removed", "random50_removed"):
    scen_dir = os.path.join(RESULTS_DIR, scenario)
    os.makedirs(scen_dir, exist_ok=True)

    for run in range(1, RUNS+1):
        print(f"\n━━ {scenario} | run {run}/{RUNS} ━━")

        if scenario == "salient50_removed":
            rem_idx = SALIENT50_IDX
        else:
            np.random.seed(1000+run)
            rem_idx = np.random.choice(X_full_raw.shape[1], 50, replace=False)

        keep_idx = np.setdiff1d(np.arange(X_full_raw.shape[1]), rem_idx)
        def sub(x): return x[:, keep_idx]

        X_tr_s = sub(X_tr)
        X_va_s = sub(X_va)
        X_te_s = sub(X_te)
        in_dim = X_tr_s.shape[1]

        def make_loader(x, y, shuf=False):
            xs, ys = create_sequences(x, y, SEQ)
            return DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=shuf)

        tr_loader = make_loader(X_tr_s, Y_tr, True)
        va_loader = make_loader(X_va_s, Y_va)
        te_loader = make_loader(X_te_s, Y_te)

        for fam, fp in families.items():
            fam_dir = os.path.join(scen_dir, f"run_{run}", fam.lower()+"_embedding_comparisons")
            os.makedirs(fam_dir, exist_ok=True)
            for emb_name, emb_cls in embed_dict.items():
                print(f"→ {fam} + {emb_name}")
                emb = emb_cls(in_dim, fp["h"]).to(device)
                net = fp["cls"](emb, out_dim).to(device)
                opt = AdamW(net.parameters(), lr=LR)

                train(net, opt, tr_loader, va_loader, EPOCH, use_pos_ids=fp["use_pos"])
                preds = predict(net, te_loader, use_pos_ids=fp["use_pos"])
                predsF = overlap_mean(preds, SEQ, len(X_te_s))
                e_rmse = rmse(predsF, Y_te)

                target = results_sal if scenario=="salient50_removed" else results_rnd
                target[fam][emb_name].append(e_rmse)
                print(f"   RMSE {e_rmse:.5f}")

                emb_dir = os.path.join(fam_dir, emb_name.lower())
                os.makedirs(emb_dir, exist_ok=True)
                np.save(os.path.join(emb_dir, f"{fam.lower()}_{emb_name.lower()}_preds_run{run}.npy"), predsF)

                gt_path = os.path.join(emb_dir, f"{fam.lower()}_{emb_name.lower()}_groundtruth.npy")
                if not os.path.exists(gt_path):
                    np.save(gt_path, Y_te)

                imp = feature_importance(net, va_loader, in_dim)
                top50 = imp.argsort()[-50:][::-1]
                np.savez(os.path.join(emb_dir, f"{fam.lower()}_{emb_name.lower()}_top50_salient.npz"),
                         idx=top50, score=imp[top50])

PLOT_DIR = os.path.join(RESULTS_DIR, "final_plots_and_stats")
os.makedirs(PLOT_DIR, exist_ok=True)

for fam in families:
    emb_names = list(embed_dict.keys())
    means_sal = [np.mean(results_sal[fam][e]) for e in emb_names]
    sems_sal  = [np.std(results_sal[fam][e])/np.sqrt(RUNS) for e in emb_names]
    means_rnd = [np.mean(results_rnd[fam][e]) for e in emb_names]
    sems_rnd  = [np.std(results_rnd[fam][e])/np.sqrt(RUNS) for e in emb_names]

    x = np.arange(len(emb_names))
    w = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x-w/2, means_sal, w, yerr=sems_sal, capsize=3, label="Salient-50 Removed")
    plt.bar(x+w/2, means_rnd, w, yerr=sems_rnd, capsize=3, label="Random-50 Removed")
    plt.xticks(x, emb_names, rotation=20)
    plt.ylabel("RMSE")
    plt.title(f"{fam}: salient vs random neuron removal (mean ± SEM, n={RUNS})")
    plt.legend()
    plt.tight_layout()
    pth = f"{PLOT_DIR}/{fam.lower()}_salient_vs_random_barplot.png"
    plt.savefig(pth)
    plt.close()
    print("Saved", pth)

    sig_p = f"{PLOT_DIR}/{fam.lower()}_salient_vs_random_significance.txt"
    with open(sig_p, "w") as f:
        f.write(f"Wilcoxon (salient vs random) ― {fam}\n")
        for e in emb_names:
            stat, p = wilcoxon(results_sal[fam][e], results_rnd[fam][e])
            f.write(f"{e}: p={p:.4e}\n")
    print("Stats →", sig_p)

print("\n Experiment 3b completed.")
