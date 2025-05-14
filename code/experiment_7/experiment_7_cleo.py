#!/usr/bin/env python3
# run_deepseek_exp1D_with_saliency.py
# Fine-tune DeepSeek-coder-7B on neural→tail & neural→tail_sum
# Saves checkpoints, resumes automatically, and produces predictions + saliency
# J. May 2025

import os, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, BitsAndBytesConfig
from torch.optim import AdamW
from datetime import datetime

# ───────────────────────── CONFIG ────────────────────────────────────────────
BASE = "/hpc/group/naumannlab/jjm132/nlp4neuro/experiment_7_cleo"
DATA_DIR   = f"{BASE}/data"
RESULT_DIR = f"{BASE}/results"
os.makedirs(RESULT_DIR, exist_ok=True)

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_length  = 20           # preferred window length
num_epochs  = 100
batch_size  = 32
lr          = 1e-4
quant_cfg   = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

def banner():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"▶ {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("▶ Device          :", device)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("▶ CUDA name       :", props.name)
        print("▶ CUDA memory     :", f"{props.total_memory/1e9:.1f} GB")
    print("▶ Epochs          :", num_epochs)
    print("▶ Batch size      :", batch_size)
    print("▶ Preferred L     :", seq_length)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ──────────────────────── HELPERS ────────────────────────────────────────────
def create_sequences(x: np.ndarray, y: np.ndarray, L: int):
    xs = sliding_window_view(x, window_shape=(L), axis=0)
    ys = sliding_window_view(y, window_shape=(L), axis=0)
    return (torch.tensor(xs, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.float32))

def rmse_from_mse(mses): return [float(np.sqrt(v)) for v in mses]

def overlap_mean(preds, total_len):
    L = preds.shape[1]
    out, cnt = np.zeros((total_len, preds.shape[2])), np.zeros(total_len)
    for i in range(preds.shape[0]):
        out[i:i+L] += preds[i]; cnt[i:i+L] += 1
    return out / cnt[:, None]

def train(model, opt, tr_loader, va_loader, model_path):
    crit, tr_hist, va_hist = nn.MSELoss(), [], []
    for ep in range(num_epochs):
        print(f"    ── Epoch {ep+1:3d}/{num_epochs} ──")
        model.train(); tot = 0.
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item()
        tr_hist.append(tot / len(tr_loader))

        model.eval(); tot = 0.
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                tot += crit(model(xb), yb).item()
        va_hist.append(tot / len(va_loader))

        if (ep+1) % 10 == 0 or ep == 0:
            print(f"       RMSE train {np.sqrt(tr_hist[-1]):.4f} | "
                  f"val {np.sqrt(va_hist[-1]):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"    ✓ checkpoint saved → {os.path.basename(model_path)}")
    return tr_hist, va_hist

@torch.no_grad()
def predict(model, loader):
    model.eval(); preds, gts = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu())
        gts.append(yb)
    return torch.cat(preds), torch.cat(gts)

def feature_importance(model, loader, input_dim):
    model.eval(); imp = torch.zeros(input_dim, device=device); n = 0
    for xb, _ in loader:
        xb = xb.to(device).requires_grad_(True)
        model(xb).sum().backward()
        imp += (xb.grad * xb).abs().mean(dim=(0,1)).detach(); n += 1
    return (imp / n).cpu().numpy()

# ───────────────────────── MODEL ─────────────────────────────────────────────
class DeepSeekMoE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=4096, n_exp=8, top_k=2):
        super().__init__()
        self.in_proj  = nn.Linear(in_dim, hidden)
        self.backbone = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True, device_map="auto",
            quantization_config=quant_cfg
        )
        for p in self.backbone.parameters(): p.requires_grad = False
        self.router  = nn.Linear(hidden, n_exp);  self.softmax = nn.Softmax(-1)
        self.top_k   = top_k;                     self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        h = self.backbone(inputs_embeds=x,
                          output_hidden_states=True).hidden_states[-1]
        w = self.softmax(self.router(h))
        idx = torch.topk(w, self.top_k, -1).indices
        agg = torch.zeros_like(h)
        for k in range(self.top_k):
            gather_idx = idx[..., k].unsqueeze(-1).expand_as(h)
            agg += torch.gather(h, -1, gather_idx) / self.top_k
        return self.out_proj(agg)

# ──────────────────── PIPELINE ───────────────────────────────────────────────
def run_pipeline(Xtr, Xva, Xte,
                 Ytr, Yva, Yte,
                 label: str, L: int):
    print(f"\n==== {label.upper()} ====")
    print(f"  • window length L = {L}")

    # paths
    model_path = f"{RESULT_DIR}/{label}_model.pt"

    # windows
    Xtr_t, Ytr_t = create_sequences(Xtr, Ytr, L)
    Xva_t, Yva_t = create_sequences(Xva, Yva, L)
    Xte_t, Yte_t = create_sequences(Xte, Yte, L)

    tr_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t),
                           batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva_t, Yva_t), batch_size=batch_size)
    te_loader = DataLoader(TensorDataset(Xte_t, Yte_t), batch_size=batch_size)

    # model
    model = DeepSeekMoE(Xtr.shape[1], Ytr.shape[1]).to(device)
    opt   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # ───── checkpoint logic ─────
    if os.path.exists(model_path):
        print(f"  • checkpoint found — loading {os.path.basename(model_path)}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        tr_hist = va_hist = None  # we won't have losses
    else:
        print("  • no checkpoint → training model …")
        tr_hist, va_hist = train(model, opt, tr_loader, va_loader, model_path)

    # ───── plots (train curve only if we just trained) ─────
    if tr_hist is not None:
        epochs = np.arange(1, num_epochs+1)
        plt.figure(figsize=(6,4))
        plt.plot(epochs, rmse_from_mse(tr_hist), label="train")
        plt.plot(epochs, rmse_from_mse(va_hist), label="val")
        plt.xlabel("epoch"); plt.ylabel("RMSE")
        plt.title(f"{label} RMSE (L={L})"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/{label}_rmse_curve.png", dpi=300); plt.close()

    # ───── predictions ─────
    pred_seq, gt_seq = predict(model, te_loader)
    np.save(f"{RESULT_DIR}/{label}_pred_sequences.npy", pred_seq.numpy())
    np.save(f"{RESULT_DIR}/{label}_gt_sequences.npy",   gt_seq.numpy())
    np.save(f"{RESULT_DIR}/{label}_groundtruth_full.npy", Yte)

    pred_full = overlap_mean(pred_seq.numpy(), len(Xte))
    np.save(f"{RESULT_DIR}/{label}_pred_full.npy", pred_full)
    final_rmse = np.sqrt(np.mean((pred_full - Yte)**2))
    print(f"  • final test RMSE : {final_rmse:.4f}")

    # ───── global saliency ─────
    print("  • computing global saliency …")
    imp = feature_importance(model, va_loader, Xtr.shape[1])
    np.save(f"{RESULT_DIR}/{label}_importance.npy", imp)

    # bar plot of top-20
    topk = imp.argsort()[-20:][::-1]
    plt.figure(figsize=(8,4))
    plt.bar(range(20), imp[topk]); plt.xticks(range(20), topk, rotation=45)
    plt.ylabel("|grad×input|"); plt.title(f"{label} – top-20 saliency")
    plt.tight_layout(); plt.savefig(f"{RESULT_DIR}/{label}_top20_saliency.png"); plt.close()

    # ───── per-frame saliency ─────
    print("  • computing per-frame saliency map …")
    frame_loader = DataLoader(TensorDataset(Xte_t, Yte_t),
                              batch_size=batch_size, shuffle=False)
    n_frames, n_neurons = Xte.shape
    frame_sal = np.zeros((n_frames, n_neurons)); frame_cnt = np.zeros(n_frames)

    model.eval(); win_start = 0
    for xb, _ in frame_loader:
        xb = xb.to(device).requires_grad_(True)
        model(xb).sum().backward()
        wi = (xb.grad * xb).abs().cpu().numpy()   # (B,L,N)
        B,Lf,_ = wi.shape
        for b in range(B):
            abs_win = win_start + b
            frame_sal[abs_win:abs_win+Lf] += wi[b]
            frame_cnt[abs_win:abs_win+Lf] += 1
        win_start += B
    frame_sal /= frame_cnt[:,None]
    np.save(f"{RESULT_DIR}/{label}_frame_saliency.npy", frame_sal)
    print("    ✓ done")

# ─────────────────────────── MAIN ────────────────────────────────────────────
if __name__ == "__main__":
    banner()

    # ----- load & tidy -----
    neural = np.load(f"{DATA_DIR}/neural_data_groundtruth_matched.npy",
                     allow_pickle=True).astype(np.float32)
    neural = neural[:, :-2].T          # drop last 2 metadata cols, then (T,N)
    tail   = np.load(f"{DATA_DIR}/tail_data_groundtruth_matched.npy",
                     allow_pickle=True).astype(np.float32)
    tail_sum = np.load(f"{DATA_DIR}/tail_data_sum_groundtruth_matched.npy",
                       allow_pickle=True).astype(np.float32)
    if tail_sum.ndim == 1: tail_sum = tail_sum[:, None]

    T = min(len(neural), len(tail), len(tail_sum))
    neural, tail, tail_sum = neural[:T], tail[:T], tail_sum[:T]

    # ----- choose an L that fits all splits -----
    L_pref = seq_length
    if T >= 3*L_pref:
        L = L_pref
    else:
        L = max(2, int(T*0.1))
        while (T*0.2) < L: L -= 1
        print(f"⚠ Short series (T={T}) → using L = {L}")

    # splits
    tr_end, va_end = int(.70*T), int(.80*T)
    Xtr, Xva, Xte = neural[:tr_end], neural[tr_end:va_end], neural[va_end:]

    # ----- run both targets -----
    run_pipeline(Xtr, Xva, Xte,
                 tail[:tr_end], tail[tr_end:va_end], tail[va_end:],
                 label="tail", L=L)

    run_pipeline(Xtr, Xva, Xte,
                 tail_sum[:tr_end], tail_sum[tr_end:va_end], tail_sum[va_end:],
                 label="tail_sum", L=L)

    print("\n✓ All done (checkpoints, predictions & saliency in results folder).")
