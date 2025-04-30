# experiment_6_deepseek_only.py
# ─────────────────────────────────────────────────────────────────────────────
import os, numpy as np, matplotlib.pyplot as plt, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AdamW, BitsAndBytesConfig
# ─────────────────────────────────────────────────────────────────────────────
# 0) GLOBAL SET-UP
# ─────────────────────────────────────────────────────────────────────────────
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,              # use 4-bit if GPU RAM is tight
    llm_int8_threshold=6.0
)

BASE_SAVE_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment_6"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

fish_list    = [9, 10, 11, 12, 13]
seq_lengths  = [10]                 # pick one or expand [5,10,15,20]
num_epochs   = 10
batch_size   = 32
lr_default   = 1e-4
num_runs     = 5                    # reduce if memory/time limited
# ─────────────────────────────────────────────────────────────────────────────
# 1) HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def create_sequences(inputs, targets, seq_length):
    xs, ys = [], []
    for i in range(len(inputs) - seq_length + 1):
        xs.append(inputs[i:i+seq_length])
        ys.append(targets[i:i+seq_length])
    return torch.stack(xs), torch.stack(ys)

def average_sliding_window_predictions(pred, seq_len, total_len):
    if torch.is_tensor(pred): pred = pred.cpu().numpy()
    n_win, _, out_dim = pred.shape
    out, cnt = np.zeros((total_len, out_dim)), np.zeros(total_len)
    for i in range(n_win):
        out[i:i+seq_len] += pred[i]
        cnt[i:i+seq_len] += 1
    return out / cnt[:, None]

def compute_rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))

def train_model(model, optim, dl_train, dl_val, n_epochs):
    crit = nn.MSELoss()
    for ep in range(1, n_epochs+1):
        model.train(); tr = 0.0
        for x,y in dl_train:
            x,y = x.to(device), y.to(device)
            optim.zero_grad(); loss = crit(model(x), y); loss.backward(); optim.step()
            tr += loss.item()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x,y in dl_val:
                x,y = x.to(device), y.to(device)
                vl += crit(model(x), y).item()
        print(f"⎡Epoch {ep:02d}/{n_epochs} ⎣ Train {tr/len(dl_train):.4f} | Val {vl/len(dl_val):.4f}")

def get_predictions(model, dl):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for x,y in dl:
            ps.append(model(x.to(device)).cpu()); ts.append(y)
    return torch.cat(ps), torch.cat(ts)

def feature_importance(model, dl, input_dim):
    model.eval(); imp = torch.zeros(input_dim, device=device); n = 0
    for x,_ in dl:
        x = x.to(device).requires_grad_(True)
        model(x).sum().backward()
        imp += (x.grad * x).abs().mean(dim=(0,1)).detach()
        n += 1
    return (imp / n).cpu().numpy()
# ─────────────────────────────────────────────────────────────────────────────
# 2) MODEL – DeepSeek-V3 MoE
# ─────────────────────────────────────────────────────────────────────────────
hidden_size, num_experts, top_k = 4096, 8, 2

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

    def forward(self, x):                     # x: (B,T,inp)
        x = self.input_proj(x)                # (B,T,H)
        hidden = self.model(inputs_embeds=x,
                            output_hidden_states=True
                           ).hidden_states[-1]# (B,T,H)
        logits   = self.router(hidden)        # (B,T,E)
        weights  = self.softmax(logits)
        idx_topk = torch.topk(weights, top_k, dim=-1).indices
        agg = torch.zeros_like(hidden)
        for k in range(top_k):
            idx = idx_topk[..., k].unsqueeze(-1).expand_as(hidden)
            agg += torch.gather(hidden, -1, idx) / top_k
        return self.output_proj(agg)          # (B,T,out)
# ─────────────────────────────────────────────────────────────────────────────
# 3) MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────
for fish in fish_list:
    print("\n", "═"*30, f"Fish {fish}", "═"*30)
    fish_dir = os.path.join(BASE_SAVE_DIR, f"fish{fish}"); os.makedirs(fish_dir, exist_ok=True)

    # load & split data ---------------------------------------------------------
    neural = np.load(f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/"
                     f"fish{fish}_neural_data_matched.npy", allow_pickle=True)[:, :-2].T
    tail   = np.load(f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/"
                     f"fish{fish}_tail_data_matched.npy",   allow_pickle=True)
    assert neural.shape[0] == tail.shape[0]
    n = neural.shape[0]; tr, vl = int(0.7*n), int(0.8*n)
    splits = dict(
        X_train = neural[:tr],  Y_train = tail[:tr],
        X_val   = neural[tr:vl],Y_val   = tail[tr:vl],
        X_test  = neural[vl:],  Y_test  = tail[vl:]
    )
    for k in splits: splits[k] = torch.tensor(splits[k], dtype=torch.float32)
    inp_dim, out_dim = splits["X_train"].shape[-1], splits["Y_train"].shape[-1]

    # save ground-truth once
    np.save(os.path.join(fish_dir, "val_gt.npy"), splits["Y_val"].numpy())
    np.save(os.path.join(fish_dir, "test_gt.npy"), splits["Y_test"].numpy())

    for run in range(1, num_runs+1):
        print(f"\n— Run {run}/{num_runs} —")
        run_dir = os.path.join(fish_dir, f"run_{run}"); os.makedirs(run_dir, exist_ok=True)

        for seq_len in seq_lengths:
            print(f"SeqLen = {seq_len}")
            seq_dir = os.path.join(run_dir, f"seq_{seq_len}"); os.makedirs(seq_dir, exist_ok=True)

            # build DataLoaders --------------------------------------------------
            X_tr, Y_tr = create_sequences(splits["X_train"], splits["Y_train"], seq_len)
            X_vl, Y_vl = create_sequences(splits["X_val"],   splits["Y_val"],   seq_len)
            X_ts, Y_ts = create_sequences(splits["X_test"],  splits["Y_test"],  seq_len)

            dl_tr = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
            dl_vl = DataLoader(TensorDataset(X_vl, Y_vl), batch_size=batch_size)
            dl_ts = DataLoader(TensorDataset(X_ts, Y_ts), batch_size=batch_size)

            # initialise & freeze backbone --------------------------------------
            model = DeepSeekV3MoE(inp_dim, out_dim).to(device)
            for p in model.model.parameters(): p.requires_grad = False

            optim_ds = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr_default)
            train_model(model, optim_ds, dl_tr, dl_vl, num_epochs)

            # predictions & RMSE ------------------------------------------------
            preds, _ = get_predictions(model, dl_ts)
            preds = average_sliding_window_predictions(preds, seq_len, len(splits["X_test"]))
            np.save(os.path.join(seq_dir, "preds.npy"), preds)
            rmse = compute_rmse(preds, splits["Y_test"].numpy())
            print(f"RMSE = {rmse:.4f}")

            # saliency analysis --------------------------------------------------
            imp = feature_importance(model, dl_vl, inp_dim)
            np.save(os.path.join(seq_dir, "importance.npy"), imp)
            top10 = imp.argsort()[-10:][::-1]
            print("Top-10 neurons:", top10.tolist(), "scores:", imp[top10])

            # quick bar plot -----------------------------------------------------
            k = 20
            plt.figure(figsize=(8,4))
            topk = imp.argsort()[-k:][::-1]
            plt.bar(range(k), imp[topk]); plt.xticks(range(k), topk, rotation=45)
            plt.ylabel("|grad × input|"); plt.title(f"Fish {fish} run {run} seq {seq_len}")
            plt.tight_layout()
            plt.savefig(os.path.join(seq_dir, f"top{k}_saliency.png"))
            plt.close()

print("\nAll fish completed.")
