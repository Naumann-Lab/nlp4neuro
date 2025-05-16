#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AdamW, BitsAndBytesConfig
from scipy.stats import wilcoxon
from tqdm import tqdm

quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

BASE_SAVE_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment_4a"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

fish_list   = [9, 10, 11, 12, 13]
seq_lengths = [10]
num_epochs  = 10
batch_size  = 32
lr_default  = 1e-4
num_runs    = 5

def create_sequences(inputs, targets, seq_length):
    xs, ys = [], []
    for i in range(len(inputs) - seq_length + 1):
        xs.append(inputs[i:i+seq_length])
        ys.append(targets[i:i+seq_length])
    return torch.stack(xs), torch.stack(ys)

def average_sliding_window_predictions(pred, seq_len, total_len):
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    n_win, _, out_dim = pred.shape
    out = np.zeros((total_len, out_dim))
    cnt = np.zeros(total_len)
    for i in range(n_win):
        out[i:i+seq_len] += pred[i]
        cnt[i:i+seq_len] += 1
    return out / cnt[:, None]

def compute_rmse(a, b):
    return float(np.sqrt(np.mean((a - b)**2)))

def train_model(model, optimizer, dl_train, dl_val, n_epochs):
    crit = nn.MSELoss()
    for ep in range(1, n_epochs+1):
        model.train()
        tr = 0.0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            optimizer.step()
            tr += loss.item()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                vl += crit(model(x), y).item()
        print(f"Epoch {ep}/{n_epochs} Train {tr/len(dl_train):.4f} Val {vl/len(dl_val):.4f}")

def get_predictions(model, dl):
    model.eval()
    preds, ts = [], []
    with torch.no_grad():
        for x, y in dl:
            preds.append(model(x.to(device)).cpu())
            ts.append(y)
    return torch.cat(preds), torch.cat(ts)

def feature_importance(model, dl, input_dim):
    model.eval()
    imp = torch.zeros(input_dim, device=device)
    n = 0
    for x, _ in dl:
        x = x.to(device).requires_grad_(True)
        model(x).sum().backward()
        imp += (x.grad * x).abs().mean(dim=(0,1)).detach()
        n += 1
    return (imp / n).cpu().numpy()

hidden_size  = 4096
num_experts  = 8
top_k        = 2

class DeepSeekV3MoE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_size)
        self.model       = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config
        )
        self.output_proj = nn.Linear(hidden_size, output_dim)
        self.router      = nn.Linear(hidden_size, num_experts)
        self.softmax     = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.input_proj(x)
        hidden = self.model(inputs_embeds=x, output_hidden_states=True).hidden_states[-1]
        logits = self.router(hidden)
        weights = self.softmax(logits)
        idx_topk = torch.topk(weights, top_k, dim=-1).indices
        agg = torch.zeros_like(hidden)
        for k in range(top_k):
            idx = idx_topk[..., k].unsqueeze(-1).expand_as(hidden)
            agg += torch.gather(hidden, -1, idx) / top_k
        return self.output_proj(agg)

for fish in fish_list:
    fish_dir = os.path.join(BASE_SAVE_DIR, f"fish{fish}")
    os.makedirs(fish_dir, exist_ok=True)
    neural = np.load(
        f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish}_neural_data_matched.npy",
        allow_pickle=True
    )[:, :-2].T
    tail = np.load(
        f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish}_tail_data_matched.npy",
        allow_pickle=True
    )
    n  = neural.shape[0]
    tr = int(0.7 * n)
    vl = int(0.8 * n)
    splits = {
        "X_train": torch.tensor(neural[:tr], dtype=torch.float32),
        "Y_train": torch.tensor(tail[:tr], dtype=torch.float32),
        "X_val":   torch.tensor(neural[tr:vl], dtype=torch.float32),
        "Y_val":   torch.tensor(tail[tr:vl], dtype=torch.float32),
        "X_test":  torch.tensor(neural[vl:], dtype=torch.float32),
        "Y_test":  torch.tensor(tail[vl:], dtype=torch.float32)
    }
    inp_dim = splits["X_train"].shape[-1]
    out_dim = splits["Y_train"].shape[-1]
    np.save(os.path.join(fish_dir, "val_gt.npy"), splits["Y_val"].numpy())
    np.save(os.path.join(fish_dir, "test_gt.npy"), splits["Y_test"].numpy())
    for run in range(1, num_runs+1):
        run_dir = os.path.join(fish_dir, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)
        for seq_len in seq_lengths:
            seq_dir = os.path.join(run_dir, f"seq_{seq_len}")
            os.makedirs(seq_dir, exist_ok=True)
            X_tr_s, Y_tr_s = create_sequences(splits["X_train"], splits["Y_train"], seq_len)
            X_vl_s, Y_vl_s = create_sequences(splits["X_val"],   splits["Y_val"],   seq_len)
            X_ts_s, Y_ts_s = create_sequences(splits["X_test"],  splits["Y_test"],  seq_len)
            dl_tr = DataLoader(TensorDataset(X_tr_s, Y_tr_s), batch_size=batch_size, shuffle=True)
            dl_vl = DataLoader(TensorDataset(X_vl_s, Y_vl_s), batch_size=batch_size)
            dl_ts = DataLoader(TensorDataset(X_ts_s, Y_ts_s), batch_size=batch_size)
            model = DeepSeekV3MoE(inp_dim, out_dim).to(device)
            for p in model.model.parameters():
                p.requires_grad = False
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
            train_model(model, optimizer, dl_tr, dl_vl, num_epochs)
            preds, _ = get_predictions(model, dl_ts)
            preds = average_sliding_window_predictions(preds, seq_len, len(splits["X_test"]))
            np.save(os.path.join(seq_dir, "preds.npy"), preds)
            err = compute_rmse(preds, splits["Y_test"].numpy())
            imp = feature_importance(model, dl_vl, inp_dim)
            np.save(os.path.join(seq_dir, "importance.npy"), imp)
            top10 = imp.argsort()[-10:][::-1]
            k = 20
            plt.figure(figsize=(8,4))
            topk = imp.argsort()[-k:][::-1]
            plt.bar(range(k), imp[topk])
            plt.xticks(range(k), topk, rotation=45)
            plt.ylabel("|grad × input|")
            plt.title(f"Fish {fish} run {run} seq {seq_len}")
            plt.tight_layout()
            plt.savefig(os.path.join(seq_dir, f"top{k}_saliency.png"))
            plt.close()
