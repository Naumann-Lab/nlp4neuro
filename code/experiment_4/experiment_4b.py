#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, BitsAndBytesConfig
from torch.optim import AdamW
from datetime import datetime
from pathlib import Path
from config import DATA_DIR as CONFIG_DATA_DIR, RESULTS_DIR as CONFIG_RESULTS_DIR

quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

# if needed, change to where you would like model results to be saved
BASE_SAVE_DIR = (CONFIG_RESULTS_DIR / "experiment_4b").resolve()
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# this should point to where the exp1-4_data folder and subfolders are...
DATA_DIR = (CONFIG_DATA_DIR / "exp1-4_data" / "data_prepped_for_models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fish_list = [9, 10, 11, 12, 13]
seq_length = 20
num_epochs = 100
batch_size = 32
lr = 1e-4
quant_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

def create_sequences(x, y, L):
    xs, ys = [], []
    for i in range(len(x) - L + 1):
        xs.append(x[i:i+L])
        ys.append(y[i:i+L])
    return torch.stack(xs), torch.stack(ys)

def rmse_from_mse(mses):
    return [float(np.sqrt(v)) for v in mses]

def train(model, opt, tr_loader, va_loader):
    crit = nn.MSELoss()
    tr_hist, va_hist = [], []
    for ep in range(num_epochs):
        model.train()
        tot = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        tr_hist.append(tot / len(tr_loader))
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                tot += crit(model(xb), yb).item()
        va_hist.append(tot / len(va_loader))
    return tr_hist, va_hist

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu())
        gts.append(yb)
    return torch.cat(preds), torch.cat(gts)

def feature_importance(model, loader, input_dim):
    model.eval()
    imp = torch.zeros(input_dim, device=device)
    n = 0
    for xb, _ in loader:
        xb = xb.to(device).requires_grad_(True)
        model(xb).sum().backward()
        imp += (xb.grad * xb).abs().mean(dim=(0,1)).detach()
        n += 1
    return (imp / n).cpu().numpy()

class DeepSeekMoE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=4096, n_exp=8, top_k=2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.backbone = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_cfg
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.router = nn.Linear(hidden, n_exp)
        self.softmax = nn.Softmax(-1)
        self.top_k = top_k
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        h = self.backbone(inputs_embeds=x, output_hidden_states=True).hidden_states[-1]
        w = self.softmax(self.router(h))
        top_idx = torch.topk(w, self.top_k, -1).indices
        agg = torch.zeros_like(h)
        for k in range(self.top_k):
            idx = top_idx[..., k].unsqueeze(-1).expand_as(h)
            agg += torch.gather(h, -1, idx) / self.top_k
        return self.out_proj(agg)

for fish in fish_list:
    fish_dir = BASE_SAVE_DIR / f"fish{fish}"
    os.makedirs(fish_dir, exist_ok=True)
    ckpt_path = fish_dir / "finetuned_heads.pt"

    neural = np.load(DATA_DIR / f"fish{fish}_neural_data_matched.npy", allow_pickle=True)[:, :-2].T
    tail = np.load(DATA_DIR / f"fish{fish}_tail_data_matched.npy", allow_pickle=True)

    n_frames = neural.shape[0]
    tr_end, va_end = int(.70*n_frames), int(.80*n_frames)
    Xtr, Xva, Xte = neural[:tr_end], neural[tr_end:va_end], neural[va_end:]
    Ytr, Yva, Yte = tail[:tr_end], tail[tr_end:va_end], tail[va_end:]

    Xtr_t, Ytr_t = create_sequences(torch.tensor(Xtr), torch.tensor(Ytr), seq_length)
    Xva_t, Yva_t = create_sequences(torch.tensor(Xva), torch.tensor(Yva), seq_length)
    Xte_t, Yte_t = create_sequences(torch.tensor(Xte), torch.tensor(Yte), seq_length)

    tr_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva_t, Yva_t), batch_size=batch_size)
    te_loader = DataLoader(TensorDataset(Xte_t, Yte_t), batch_size=batch_size)

    model = DeepSeekMoE(Xtr.shape[1], Ytr.shape[1]).to(device)
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if os.path.exists(ckpt_path):
        heads_state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(heads_state, strict=False)
        tr_loss = va_loss = []
    else:
        xb_smoke, _ = next(iter(va_loader))
        xb_smoke = xb_smoke.to(device).requires_grad_(True)
        (model(xb_smoke).sum()).backward()
        tr_loss, va_loss = train(model, opt, tr_loader, va_loader)
        trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        heads_state = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_keys}
        torch.save(heads_state, ckpt_path)
        epochs = np.arange(1, num_epochs+1)
        plt.figure(figsize=(6,4))
        plt.plot(epochs, rmse_from_mse(tr_loss), label="train")
        plt.plot(epochs, rmse_from_mse(va_loss), label="val")
        plt.xlabel("epoch")
        plt.ylabel("RMSE")
        plt.title(f"Fish {fish}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fish_dir / f"fish{fish}_rmse_curve.png", dpi=300)
        plt.close()

    pred_seq, gt_seq = predict(model, te_loader)
    np.save(fish_dir / f"fish{fish}_pred_sequences.npy", pred_seq.numpy())
    np.save(fish_dir / f"fish{fish}_gt_sequences.npy",   gt_seq.numpy())
    np.save(fish_dir / f"fish{fish}_groundtruth_full.npy", Yte)

    def overlap_mean(preds, total_len):
        L = preds.shape[1]
        out = np.zeros((total_len, preds.shape[2]))
        cnt = np.zeros(total_len)
        for i in range(preds.shape[0]):
            out[i:i+L] += preds[i]
            cnt[i:i+L] += 1
        return out / cnt[:, None]

    pred_full = overlap_mean(pred_seq.numpy(), len(Xte))
    np.save(fish_dir / f"fish{fish}_pred_full.npy", pred_full)

    imp = feature_importance(model, va_loader, Xtr.shape[1])
    np.save(fish_dir / f"fish{fish}_importance.npy", imp)

    top200 = imp.argsort()[-200:][::-1]
    np.savez(fish_dir / f"fish{fish}_top200_neurons.npz", idx=top200, score=imp[top200])

    k = 20
    topk = imp.argsort()[-k:][::-1]
    plt.figure(figsize=(8,4))
    plt.bar(range(k), imp[topk])
    plt.xticks(range(k), topk, rotation=45)
    plt.ylabel("|grad × input|")
    plt.title(f"Fish {fish} – top {k} saliency")
    plt.tight_layout()
    plt.savefig(fish_dir / f"fish{fish}_top{k}_saliency.png")
    plt.close()

    frame_loader = DataLoader(TensorDataset(Xte_t, Yte_t), batch_size=batch_size, shuffle=False)

    n_frames_test, n_neurons = Xte.shape
    frame_sal = np.zeros((n_frames_test, n_neurons))
    frame_cnt = np.zeros(n_frames_test, dtype=np.int32)

    win_start = 0
    for xb, _ in frame_loader:
        xb = xb.to(device).requires_grad_(True)
        model(xb).sum().backward()
        wi = (xb.grad * xb).abs().detach().cpu().numpy()
        B, L, _ = wi.shape
        for b in range(B):
            for t in range(L):
                frame_sal[win_start + b + t] += wi[b, t]
                frame_cnt[win_start + b + t] += 1
        win_start += B

    frame_sal /= frame_cnt[:, None]
    np.save(fish_dir / f"fish{fish}_frame_saliency.npy", frame_sal)
