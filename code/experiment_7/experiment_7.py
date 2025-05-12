#!/usr/bin/env python3
# run_deepseek_exp1D.py — fine-tune DeepSeek-coder-7B only
# J. May 2025  (verbose version)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, BitsAndBytesConfig
from torch.optim import AdamW
from datetime import datetime

# ───────────────────────── CONFIG ────────────────────────────────────────────
BASE_SAVE_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/deepseek_only"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fish_list     = [9, 10, 11, 12, 13]
seq_length    = 20            # fixed
num_epochs    = 100
batch_size    = 32
lr            = 1e-4

quant_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"▶ Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("▶ Using device :", device)
if torch.cuda.is_available():
    print("▶ CUDA name   :", torch.cuda.get_device_name(0))
    print("▶ CUDA memory :", f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("▶ Epochs      :", num_epochs)
print("▶ Batch size  :", batch_size)
print("▶ Seq length  :", seq_length)
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ──────────────────────── HELPERS ────────────────────────────────────────────
def create_sequences(x: torch.Tensor, y: torch.Tensor, L: int):
    """Return rolling windows of length L from x and y (T × D)."""
    xseq, yseq = [], []
    for i in range(len(x) - L + 1):
        xseq.append(x[i : i + L])
        yseq.append(y[i : i + L])
    return torch.stack(xseq), torch.stack(yseq)

def rmse_from_mse(mse_vals):
    """Convert a list/array of MSE values → RMSE values."""
    return [float(np.sqrt(v)) for v in mse_vals]

def train(model, opt, tr_loader, va_loader):
    crit = nn.MSELoss()
    tr_hist, va_hist = [], []
    for ep in range(num_epochs):
        print(f"    ── Epoch {ep+1:3d}/{num_epochs} ──")
        # ---- train
        model.train(); tot = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        tr_hist.append(tot / len(tr_loader))

        # ---- validate
        model.eval(); tot = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                tot += crit(out, yb).item()
        va_hist.append(tot / len(va_loader))

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"       RMSE train {np.sqrt(tr_hist[-1]):.4f} | "
                  f"val {np.sqrt(va_hist[-1]):.4f}")

    return tr_hist, va_hist

@torch.no_grad()
def predict(model, loader):
    print("    ▶ Predicting on test set …")
    model.eval(); preds, gts = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu())
        gts.append(yb)
    return torch.cat(preds), torch.cat(gts)

# ───────────────────────── MODEL ─────────────────────────────────────────────
class DeepSeekMoE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=4096, n_exp=8, top_k=2):
        super().__init__()
        self.in_proj  = nn.Linear(in_dim, hidden)
        self.backbone = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_cfg
        )
        self.router   = nn.Linear(hidden, n_exp)
        self.out_proj = nn.Linear(hidden, out_dim)
        self.softmax  = nn.Softmax(-1)
        self.top_k    = top_k

        # freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.in_proj(x)
        h = self.backbone(inputs_embeds=x,
                          output_hidden_states=True).hidden_states[-1]
        w = self.softmax(self.router(h))
        topk_idx = torch.topk(w, self.top_k, -1).indices
        agg = torch.zeros_like(h)
        for k in range(self.top_k):
            idx = topk_idx[..., k].unsqueeze(-1).expand_as(h)
            agg += torch.gather(h, -1, idx) / self.top_k
        return self.out_proj(agg)

# ───────────────────────── MAIN LOOP ─────────────────────────────────────────
for fish in fish_list:
    print(f"\n================= Fish {fish} =================")
    fish_dir = os.path.join(BASE_SAVE_DIR, f"fish{fish}")
    os.makedirs(fish_dir, exist_ok=True)

    # --- load data
    neural = np.load(
        f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/"
        f"fish{fish}_neural_data_matched.npy", allow_pickle=True
    )[:, :-2].T
    tail = np.load(
        f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/"
        f"fish{fish}_tail_data_matched.npy", allow_pickle=True
    )

    print("  • neural data shape :", neural.shape)
    print("  • tail   data shape :", tail.shape)

    # --- split
    n_frames = neural.shape[0]
    tr_end, va_end = int(0.7 * n_frames), int(0.8 * n_frames)
    Xtr, Xva, Xte = neural[:tr_end], neural[tr_end:va_end], neural[va_end:]
    Ytr, Yva, Yte = tail[:tr_end], tail[tr_end:va_end], tail[va_end:]
    print(f"  • frames -> train {tr_end}, val {va_end - tr_end}, "
          f"test {n_frames - va_end}")

    # --- sequences
    Xtr_t, Ytr_t = create_sequences(torch.tensor(Xtr), torch.tensor(Ytr), seq_length)
    Xva_t, Yva_t = create_sequences(torch.tensor(Xva), torch.tensor(Yva), seq_length)
    Xte_t, Yte_t = create_sequences(torch.tensor(Xte), torch.tensor(Yte), seq_length)
    print(f"  • windowed Xtr size  :", Xtr_t.shape)
    print(f"  • batches/epoch      :", len(Xtr_t) // batch_size)

    tr_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t),
                           batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva_t, Yva_t),
                           batch_size=batch_size)
    te_loader = DataLoader(TensorDataset(Xte_t, Yte_t),
                           batch_size=batch_size)

    # --- model
    model = DeepSeekMoE(Xtr.shape[1], Ytr.shape[1]).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • model dims         : {Xtr.shape[1]} → {Ytr.shape[1]}")
    print(f"  • trainable params   : {trainable:,}")

    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # --- train
    tr_loss, va_loss = train(model, opt, tr_loader, va_loader)

    # --- plot RMSE vs epoch
    epochs = np.arange(1, num_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, rmse_from_mse(tr_loss), label="train")
    plt.plot(epochs, rmse_from_mse(va_loss), label="val")
    plt.xlabel("epoch"); plt.ylabel("RMSE"); plt.title(f"Fish {fish}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fish_dir, f"fish{fish}_rmse_curve.png"), dpi=300)
    plt.close()
    print("  • saved RMSE curve   ✔")

    # --- predictions
    pred_seq, gt_seq = predict(model, te_loader)     # (Nwin, L, out_dim)
    np.save(os.path.join(fish_dir, f"fish{fish}_pred_sequences.npy"),
            pred_seq.numpy())
    np.save(os.path.join(fish_dir, f"fish{fish}_gt_sequences.npy"),
            gt_seq.numpy())
    np.save(os.path.join(fish_dir, f"fish{fish}_groundtruth_full.npy"), Yte)
    print("  • saved sequences    ✔")

    # --- overlap-mean to per-frame preds
    def overlap_mean(preds, total_len):
        L = preds.shape[1]
        out = np.zeros((total_len, preds.shape[2]))
        cnt = np.zeros(total_len)
        for i in range(preds.shape[0]):
            out[i : i + L] += preds[i]
            cnt[i : i + L] += 1
        return out / cnt[:, None]

    pred_full = overlap_mean(pred_seq.numpy(), len(Xte))
    np.save(os.path.join(fish_dir, f"fish{fish}_pred_full.npy"), pred_full)

    # --- final RMSE
    final_rmse = np.sqrt(np.mean((pred_full - Yte) ** 2))
    print(f"  • final test RMSE    : {final_rmse:.4f}")

print("\n✓ DeepSeek-only experiment finished for all fish.")
