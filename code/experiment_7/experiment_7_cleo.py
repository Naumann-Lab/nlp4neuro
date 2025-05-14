#!/usr/bin/env python3
# run_deepseek_exp1D_with_saliency.py
# Fine-tune DeepSeek-coder-7B on neural→tail and neural→tail_sum
# and store global + per-frame saliency maps
# J. May 2025  (verbose version)

import os, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, BitsAndBytesConfig
from torch.optim import AdamW
from datetime import datetime

# ───────────────────────── CONFIG ────────────────────────────────────────────
BASE_SAVE_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/experiment_7_cleo/results"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_length  = 20
num_epochs  = 100
batch_size  = 32
lr          = 1e-4

quant_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"▶ Script started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("▶ Using device    :", device)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("▶ CUDA name       :", props.name)
    print("▶ CUDA memory     :", f"{props.total_memory/1e9:.1f} GB")   # ← fixed
print("▶ Epochs          :", num_epochs)
print("▶ Batch size      :", batch_size)
print("▶ Sequence length :", seq_length)
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ──────────────────────── HELPERS ────────────────────────────────────────────
def create_sequences(x: torch.Tensor, y: torch.Tensor, L: int):
    xs, ys = [], []
    for i in range(len(x) - L + 1):
        xs.append(x[i:i + L])
        ys.append(y[i:i + L])
    return torch.stack(xs), torch.stack(ys)

def rmse_from_mse(mses):
    return [float(np.sqrt(v)) for v in mses]

def overlap_mean(preds, total_len):
    L = preds.shape[1]
    out, cnt = np.zeros((total_len, preds.shape[2])), np.zeros(total_len)
    for i in range(preds.shape[0]):
        out[i:i + L] += preds[i]
        cnt[i:i + L] += 1
    return out / cnt[:, None]

def train(model, opt, tr_loader, va_loader):
    crit, tr_hist, va_hist = nn.MSELoss(), [], []
    for ep in range(num_epochs):
        print(f"    ── Epoch {ep + 1:3d}/{num_epochs} ──")
        model.train(); tot = 0.
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item()
        tr_hist.append(tot / len(tr_loader))
        model.eval(); tot = 0.
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                tot += crit(model(xb), yb).item()
        va_hist.append(tot / len(va_loader))
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"       RMSE  train {np.sqrt(tr_hist[-1]):.4f} | "
                  f"val {np.sqrt(va_hist[-1]):.4f}")
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
        out = model(xb).sum(); out.backward()
        imp += (xb.grad * xb).abs().mean(dim=(0, 1)).detach()
        n += 1
    return (imp / n).cpu().numpy()

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
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.router   = nn.Linear(hidden, n_exp)
        self.softmax  = nn.Softmax(-1)
        self.top_k    = top_k
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        h = self.backbone(inputs_embeds=x,
                          output_hidden_states=True).hidden_states[-1]
        w = self.softmax(self.router(h))
        top_idx = torch.topk(w, self.top_k, -1).indices
        agg = torch.zeros_like(h)
        for k in range(self.top_k):
            idx = top_idx[..., k].unsqueeze(-1).expand_as(h)
            agg += torch.gather(h, -1, idx) / self.top_k
        return self.out_proj(agg)

# ──────────────────── PIPELINE ───────────────────────────────────────────────
def run_pipeline(Xtr, Xva, Xte,
                 Ytr, Yva, Yte,
                 label: str):
    Xtr_t, Ytr_t = create_sequences(torch.tensor(Xtr), torch.tensor(Ytr), seq_length)
    Xva_t, Yva_t = create_sequences(torch.tensor(Xva), torch.tensor(Yva), seq_length)
    Xte_t, Yte_t = create_sequences(torch.tensor(Xte), torch.tensor(Yte), seq_length)

    tr_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva_t, Yva_t), batch_size=batch_size)
    te_loader = DataLoader(TensorDataset(Xte_t, Yte_t), batch_size=batch_size)

    model = DeepSeekMoE(Xtr.shape[1], Ytr.shape[1]).to(device)
    opt   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    tr_loss, va_loss = train(model, opt, tr_loader, va_loader)

    epochs = np.arange(1, num_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, rmse_from_mse(tr_loss), label="train")
    plt.plot(epochs, rmse_from_mse(va_loss), label="val")
    plt.xlabel("epoch"); plt.ylabel("RMSE"); plt.title(f"{label} RMSE")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(BASE_SAVE_DIR, f"{label}_rmse_curve.png"), dpi=300)
    plt.close()

    pred_seq, gt_seq = predict(model, te_loader)
    np.save(os.path.join(BASE_SAVE_DIR, f"{label}_pred_sequences.npy"), pred_seq.numpy())
    np.save(os.path.join(BASE_SAVE_DIR, f"{label}_gt_sequences.npy"),   gt_seq.numpy())
    np.save(os.path.join(BASE_SAVE_DIR, f"{label}_groundtruth_full.npy"), Yte)

    pred_full = overlap_mean(pred_seq.numpy(), len(Xte))
    np.save(os.path.join(BASE_SAVE_DIR, f"{label}_pred_full.npy"), pred_full)

    final_rmse = np.sqrt(np.mean((pred_full - Yte) ** 2))
    print(f"  • {label} final test RMSE : {final_rmse:.4f}")

    print(f"  • computing {label} global saliency …")
    imp = feature_importance(model, va_loader, Xtr.shape[1])
    np.save(os.path.join(BASE_SAVE_DIR, f"{label}_importance.npy"), imp)

    top200_idx = imp.argsort()[-200:][::-1]
    np.savez(os.path.join(BASE_SAVE_DIR, f"{label}_top200_neurons.npz"),
             idx=top200_idx, score=imp[top200_idx])

    k = 20
    topk = imp.argsort()[-k:][::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(k), imp[topk])
    plt.xticks(range(k), topk, rotation=45)
    plt.ylabel("|grad × input|"); plt.title(f"{label} – top {k} saliency")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_SAVE_DIR, f"{label}_top{k}_saliency.png"))
    plt.close()

    print(f"  • computing {label} per-frame saliency map …")
    frame_loader = DataLoader(TensorDataset(Xte_t, Yte_t),
                              batch_size=batch_size, shuffle=False)

    n_test_frames, n_neurons = Xte.shape[0], Xte.shape[1]
    frame_sal = np.zeros((n_test_frames, n_neurons), dtype=np.float64)
    frame_cnt = np.zeros(n_test_frames, dtype=np.int32)

    model.eval(); window_start = 0
    for xb, _ in frame_loader:
        xb = xb.to(device).requires_grad_(True)
        (model(xb).sum()).backward()
        wi = (xb.grad * xb).abs().cpu().numpy()
        B, L, _ = wi.shape
        for b in range(B):
            absolute_win = window_start + b
            for t in range(L):
                frame_sal[absolute_win + t] += wi[b, t]
                frame_cnt[absolute_win + t] += 1
        window_start += B
    frame_sal /= frame_cnt[:, None]
    np.save(os.path.join(BASE_SAVE_DIR, f"{label}_frame_saliency.npy"), frame_sal)
    print("    saved ✔")

# ─────────────────────────── MAIN ────────────────────────────────────────────
neural = np.load("/hpc/group/naumannlab/jjm132/nlp4neuro/experiment_7_cleo/data/"
                 "neural_data_groundtruth_matched.npy", allow_pickle=True)[:, :-2].T
tail       = np.load("/hpc/group/naumannlab/jjm132/nlp4neuro/experiment_7_cleo/data/"
                     "tail_data_groundtruth_matched.npy", allow_pickle=True)
tail_sum   = np.load("/hpc/group/naumannlab/jjm132/nlp4neuro/experiment_7_cleo/data/"
                     "tail_data_sum_groundtruth_matched.npy", allow_pickle=True)

n_frames = neural.shape[0]
tr_end, va_end = int(.70 * n_frames), int(.80 * n_frames)
Xtr, Xva, Xte = neural[:tr_end], neural[tr_end:va_end], neural[va_end:]

Ytr, Yva, Yte = tail[:tr_end], tail[tr_end:va_end], tail[va_end:]
run_pipeline(Xtr, Xva, Xte, Ytr, Yva, Yte, label="tail")

Ytr, Yva, Yte = tail_sum[:tr_end], tail_sum[tr_end:va_end], tail_sum[va_end:]
run_pipeline(Xtr, Xva, Xte, Ytr, Yva, Yte, label="tail_sum")

print("\n✓ DeepSeek experiment finished (neural→tail and neural→tail_sum).")
