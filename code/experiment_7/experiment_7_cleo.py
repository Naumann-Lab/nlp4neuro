#!/usr/bin/env python3
# experiment_7_cleo.py
# Fine-tune DeepSeek-coder-7B heads on neural→tail & neural→tail_sum
# Checkpoint-aware; global + per-frame saliency
# May 2025

import os, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
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
seq_length  = 20
num_epochs  = 100
batch_size  = 32
lr          = 1e-4
quant_cfg   = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

def banner():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"▶ {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("▶ Device          :", device)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("▶ GPU             :", p.name,
              "| mem", f"{p.total_memory/1e9:.1f} GB")
    print("▶ Epochs          :", num_epochs,
          "| Batch", batch_size, "| seq_length", seq_length)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ──────────────────────── HELPERS ────────────────────────────────────────────
def create_sequences(x: torch.Tensor, y: torch.Tensor, L: int):
    xs, ys = [], []
    for i in range(len(x) - L + 1):
        xs.append(x[i:i+L]); ys.append(y[i:i+L])
    return torch.stack(xs), torch.stack(ys)

def rmse_from_mse(mses): return [float(np.sqrt(v)) for v in mses]

def overlap_mean(preds, total_len):
    L = preds.shape[1]
    out, cnt = np.zeros((total_len, preds.shape[2])), np.zeros(total_len)
    for i in range(preds.shape[0]):
        out[i:i+L] += preds[i]; cnt[i:i+L] += 1
    return out / cnt[:, None]

def orient_t_first(arr, T_ref=None):
    """
    Guarantee arr has time in dim-0, matching T_ref if supplied.
    """
    if arr.ndim == 1:                         # (T,) → (T,1)
        arr = arr[:, None]
    if T_ref is None:                         # decide by longer dim
        return arr if arr.shape[0] >= arr.shape[1] else arr.T
    # with reference → match it
    if arr.shape[0] == T_ref:     return arr
    if arr.shape[1] == T_ref:     return arr.T
    raise ValueError("Cannot align array with time dimension")

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
        self.top_k   = top_k;                    self.out_proj = nn.Linear(hidden, out_dim)

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

# ──────────────────── TRAIN / INFER ──────────────────────────────────────────
def train_or_load(model, tr_loader, va_loader, path):
    if os.path.exists(path):
        print("  • checkpoint found – loading and skipping training")
        model.load_state_dict(torch.load(path, map_location=device))
        return None, None     # no loss curves
    crit = nn.MSELoss(); opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    tr_hist, va_hist = [], []
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
        if (ep == 0) or ((ep+1) % 10 == 0):
            print(f"       RMSE train {np.sqrt(tr_hist[-1]):.4f} | "
                  f"val {np.sqrt(va_hist[-1]):.4f}")
    torch.save(model.state_dict(), path)
    print("    ✓ checkpoint saved")
    return tr_hist, va_hist

@torch.no_grad()
def predict(model, loader):
    model.eval(); preds, gts = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu())
        gts.append(yb)
    return torch.cat(preds), torch.cat(gts)

def feature_importance(model, loader, N_in):
    """
    Mean |grad × input| across all windows & time-steps.
    Returns a NumPy array of length N_in.
    """
    model.eval()
    imp = torch.zeros(N_in, device=device)          # accumulation buffer
    n = 0
    for xb, _ in loader:
        xb = xb.to(device).requires_grad_(True)
        model(xb).sum().backward()                  # single scalar backward
        imp += (xb.grad * xb).abs().mean(dim=(0, 1)).detach()
        n += 1
    if n == 0:                                      # should never happen, but safe-guard
        return imp.cpu().numpy()
    return (imp / n).detach().cpu().numpy()         # ← detach before .numpy()


# ──────────────────── PIPELINE ───────────────────────────────────────────────
def run_pipeline(Xtr, Xva, Xte,
                 Ytr, Yva, Yte,
                 label):
    print(f"\n==== {label.upper()} ====")
    model_path = f"{RESULT_DIR}/{label}_model.pt"

    Xtr_t, Ytr_t = create_sequences(torch.tensor(Xtr), torch.tensor(Ytr), seq_length)
    Xva_t, Yva_t = create_sequences(torch.tensor(Xva), torch.tensor(Yva), seq_length)
    Xte_t, Yte_t = create_sequences(torch.tensor(Xte), torch.tensor(Yte), seq_length)

    tr_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva_t, Yva_t), batch_size=batch_size)
    te_loader = DataLoader(TensorDataset(Xte_t, Yte_t), batch_size=batch_size)

    model = DeepSeekMoE(Xtr.shape[1], Ytr.shape[1]).to(device)
    tr_loss, va_loss = train_or_load(model, tr_loader, va_loader, model_path)

    if tr_loss is not None:
        epochs = np.arange(1, num_epochs+1)
        plt.figure(figsize=(6,4))
        plt.plot(epochs, rmse_from_mse(tr_loss), label="train")
        plt.plot(epochs, rmse_from_mse(va_loss), label="val")
        plt.xlabel("epoch"); plt.ylabel("RMSE")
        plt.title(label); plt.legend(); plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/{label}_rmse_curve.png", dpi=300); plt.close()

    pred_seq, gt_seq = predict(model, te_loader)
    np.save(f"{RESULT_DIR}/{label}_pred_sequences.npy", pred_seq.numpy())
    np.save(f"{RESULT_DIR}/{label}_gt_sequences.npy",   gt_seq.numpy())
    np.save(f"{RESULT_DIR}/{label}_groundtruth_full.npy", Yte)

    pred_full = overlap_mean(pred_seq.numpy(), len(Xte))
    np.save(f"{RESULT_DIR}/{label}_pred_full.npy", pred_full)
    print(f"  • final test RMSE : {np.sqrt(np.mean((pred_full-Yte)**2)):.4f}")

    # global saliency
    print("  • global saliency …")
    imp = feature_importance(model, va_loader, Xtr.shape[1])
    np.save(f"{RESULT_DIR}/{label}_importance.npy", imp)

    # per-frame saliency
    print("  • per-frame saliency map …")
    frame_loader = DataLoader(TensorDataset(Xte_t, Yte_t),
                              batch_size=batch_size, shuffle=False)
    n_frames, N_in = Xte.shape; frame_sal = np.zeros((n_frames, N_in))
    frame_cnt = np.zeros(n_frames, dtype=np.int32)
    model.eval(); win_start = 0
    for xb, _ in frame_loader:
        xb = xb.to(device).requires_grad_(True)
        model(xb).sum().backward()
        wi = (xb.grad * xb).abs().detach().cpu().numpy()   # (B,L,N)
        B,L,_ = wi.shape
        for b in range(B):
            frame_sal[win_start+b : win_start+b+L] += wi[b]
            frame_cnt[win_start+b : win_start+b+L] += 1
        win_start += B
    frame_sal /= frame_cnt[:,None]
    np.save(f"{RESULT_DIR}/{label}_frame_saliency.npy", frame_sal)
    print("    ✓ done")

# ───────────────────────── MAIN ────────────────────────────────────────────
if __name__ == "__main__":
    banner()

    # ---------- load raw arrays ----------
    neural    = np.load(f"{DATA_DIR}/neural_data_groundtruth_matched.npy",
                        allow_pickle=True).astype(np.float32)
    tail      = np.load(f"{DATA_DIR}/tail_data_groundtruth_matched.npy",
                        allow_pickle=True).astype(np.float32)
    tail_sum  = np.load(f"{DATA_DIR}/tail_data_sum_groundtruth_matched.npy",
                        allow_pickle=True).astype(np.float32)

    # ---------- orient so time is dim-0 ----------
    neural   = orient_t_first(neural)
    T        = neural.shape[0]                   # reference length
    tail     = orient_t_first(tail,     T_ref=T)[:T]
    tail_sum = orient_t_first(tail_sum, T_ref=T)[:T]

    # If neural has extra metadata cols (e.g. last 2), drop them *after* aligning
    if neural.shape[1] > tail.shape[1]:
        neural = neural[:, :tail.shape[1]]

    # ---------- train/val/test split ----------
    tr_end, va_end = int(.70*T), int(.80*T)
    Xtr, Xva, Xte = neural[:tr_end], neural[tr_end:va_end], neural[va_end:]

    run_pipeline(Xtr, Xva, Xte,
                 tail[:tr_end], tail[tr_end:va_end], tail[va_end:],
                 label="tail")

    run_pipeline(Xtr, Xva, Xte,
                 tail_sum[:tr_end], tail_sum[tr_end:va_end], tail_sum[va_end:],
                 label="tail_sum")

    print("\n✓ Finished (checkpoints & outputs in results folder)")
