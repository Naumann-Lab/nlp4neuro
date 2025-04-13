#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import BitsAndBytesConfig for quantization support
from transformers import (
    GPT2Config,
    GPT2Model,
    AdamW,
    AutoModel,
    AutoConfig,
    BertModel,
    BertConfig,
    BitsAndBytesConfig
)
from scipy.stats import wilcoxon

# Define a quantization configuration for DeepSeek
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,          # or load_in_4bit=True if desired
    llm_int8_threshold=6.0
)

# =============================================================================
# 0) Setup & Data Loading
# =============================================================================

RESULTS_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment2_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Choose which fish to run (example: only fish 9, or a list)
fish_num = 9

data_dir = "/hpc/group/naumannlab/jjm132/data_prepped_for_models"
neural_data = np.load(
    os.path.join(data_dir, f"fish{fish_num}_neural_data_matched.npy"),
    allow_pickle=True
)[:, :-2]
tail_data = np.load(
    os.path.join(data_dir, f"fish{fish_num}_tail_data_matched.npy"),
    allow_pickle=True
)

# Transpose neural data => shape: (num_frames, num_neurons)
neural_data = neural_data.T
print("Neural data shape:", neural_data.shape)
print("Tail data shape:", tail_data.shape)
assert neural_data.shape[0] == tail_data.shape[0], "Mismatch in data length"

# Train/Val/Test split
total_frames = neural_data.shape[0]
train_end = int(0.7 * total_frames)  # 70% train
val_end   = int(0.8 * total_frames)  # 10% val, 20% test

X_train = neural_data[:train_end]
X_val   = neural_data[train_end:val_end]
X_test  = neural_data[val_end:]
Y_train = tail_data[:train_end]
Y_val   = tail_data[train_end:val_end]
Y_test  = tail_data[val_end:]

# Save ground-truth for reference, in base results directory
np.save(os.path.join(RESULTS_DIR, "groundtruth_val.npy"), Y_val)
np.save(os.path.join(RESULTS_DIR, "groundtruth_test.npy"), Y_test)
print("Ground truth for val/test saved in base directory.")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)

# Input/output dimensions
input_dim  = X_train_t.size(-1)  # number of neurons
output_dim = Y_train_t.size(-1)  # number of tail features
print(f"Input_dim={input_dim}, Output_dim={output_dim}")

# =============================================================================
# 1) Helper Functions
# =============================================================================

def create_sequences(inputs, targets, seq_length):
    """
    Create overlapping sequences of length seq_length from continuous data.
    Returns: (seq_inputs, seq_targets) with shape (num_windows, seq_length, input_dim).
    """
    seq_in_list = []
    seq_out_list = []
    for i in range(len(inputs) - seq_length + 1):
        seq_in_list.append(inputs[i : i + seq_length])
        seq_out_list.append(targets[i : i + seq_length])
    return torch.stack(seq_in_list), torch.stack(seq_out_list)

def average_sliding_window_predictions(predictions, seq_length, total_length):
    """
    Averages predictions to get a single series of length total_length.
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    num_windows, _, out_dim = predictions.shape
    out = np.zeros((total_length, out_dim))
    counts = np.zeros(total_length)
    for i in range(num_windows):
        out[i : i + seq_length] += predictions[i]
        counts[i : i + seq_length] += 1
    out /= counts[:, None]
    return out

def compute_rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))

def train_model(model, optimizer, train_loader, val_loader, device, num_epochs):
    """
    Simple training loop with MSE loss.
    """
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

def get_predictions(model, data_loader, device):
    """
    Return predictions for entire data_loader.
    """
    model.eval()
    preds_list = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_list.append(outputs.cpu())
    return torch.cat(preds_list, dim=0)

# =============================================================================
# 2) Model Definitions
# =============================================================================

hidden_size_deepseek = 4096
num_experts = 8
top_k = 2

class DeepSeekV3MoEPretrained(nn.Module):
    """
    DeepSeek Pretrained
    UPDATED: This version now mimics the memory-saving technique from your second code.
    The transformer is loaded in quantized mode and its parameters are frozen.
    We call the transformer normally (as in code 2) and then freeze its parameters.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        from transformers import AutoModel
        self.input_proj = nn.Linear(input_dim, hidden_size_deepseek)
        self.model = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config
        )
        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)
        self.router = nn.Linear(hidden_size_deepseek, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.input_proj(x)
        # Call the transformer normally
        out = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = out.hidden_states[-1]
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, top_k, dim=-1)[1]
        aggregated = torch.zeros_like(hidden_states)
        for i in range(top_k):
            expert_idx = top_k_experts[:, :, i].unsqueeze(-1).expand_as(hidden_states)
            expert_out = torch.gather(hidden_states, dim=-1, index=expert_idx)
            aggregated += expert_out / top_k
        logits = self.output_proj(aggregated)
        return logits

class DeepSeekV3MoEVanilla(nn.Module):
    """
    DeepSeek Untrained (Vanilla) - same architecture, random init
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        self.input_proj = nn.Linear(input_dim, hidden_size_deepseek)
        config = AutoConfig.from_pretrained("deepseek-ai/deepseek-coder-7b", trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)
        self.router = nn.Linear(hidden_size_deepseek, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = out.hidden_states[-1]
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, top_k, dim=-1)[1]
        aggregated = torch.zeros_like(hidden_states)
        for i in range(top_k):
            expert_idx = top_k_experts[:, :, i].unsqueeze(-1).expand_as(hidden_states)
            expert_out = torch.gather(hidden_states, dim=-1, index=expert_idx)
            aggregated += expert_out / top_k
        logits = self.output_proj(aggregated)
        return logits

# ---- GPT-2 Models (Pretrained vs Vanilla) ----
class GPT2Pretrained(nn.Module):
    """
    GPT-2 with pretrained weights
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained("gpt2")
        hidden_size = self.transformer.config.n_embd
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(inputs_embeds=x)
        hidden_states = out.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

class GPT2Vanilla(nn.Module):
    """
    GPT-2 Untrained (Vanilla) - random init
    Example smaller config
    """
    def __init__(self, input_dim, hidden_size, output_dim, n_head, n_layer, n_positions):
        super().__init__()
        config = GPT2Config(
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=n_positions,
            vocab_size=1
        )
        self.transformer = GPT2Model(config)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(inputs_embeds=x)
        hidden_states = out.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# ---- BERT Models (Pretrained vs Vanilla) ----
class BERTPretrained(nn.Module):
    """
    BERT (bert-base-uncased) pretrained
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.bert(inputs_embeds=x)
        hidden_states = out.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

class BERTVanilla(nn.Module):
    """
    BERT Untrained (Vanilla) - random init
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel(config)  # random init
        hidden_size = config.hidden_size
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.bert(inputs_embeds=x)
        hidden_states = out.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# =============================================================================
# 3) Main Loop: 10 Runs, Sequence Lengths [5, 20]
# =============================================================================
model_names = [
    "DeepSeek Pretrained",
    "DeepSeek Vanilla",
    "GPT2 Pretrained",
    "GPT2 Vanilla",
    "BERT Pretrained",
    "BERT Vanilla",
]

def build_model(model_name, input_dim, output_dim):
    if model_name == "DeepSeek Pretrained":
        return DeepSeekV3MoEPretrained(input_dim, output_dim)
    elif model_name == "DeepSeek Vanilla":
        return DeepSeekV3MoEVanilla(input_dim, output_dim)
    elif model_name == "GPT2 Pretrained":
        return GPT2Pretrained(input_dim, output_dim)
    elif model_name == "GPT2 Vanilla":
        # Example config for GPT-2 vanilla
        hidden_size = 512  # smaller than actual GPT2 for demonstration
        n_head = 8
        n_layer = 6
        n_positions = 512
        return GPT2Vanilla(input_dim, hidden_size, output_dim, n_head, n_layer, n_positions)
    elif model_name == "BERT Pretrained":
        return BERTPretrained(input_dim, output_dim)
    elif model_name == "BERT Vanilla":
        return BERTVanilla(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model name {model_name}")

seq_lengths = [5, 20]
num_runs = 10
num_epochs = 10
batch_size = 32
lr_default = 1e-4

# final_all_rmse[seq_len][model_name] -> list of RMSEs over multiple runs
final_all_rmse = {
    seq: {m: [] for m in model_names}
    for seq in seq_lengths
}

for run_idx in range(1, num_runs + 1):
    print(f"\n=== Run {run_idx}/{num_runs} ===")
    for seq_length in seq_lengths:
        print(f"--- Sequence Length = {seq_length} ---")
        # Prepare sequences
        train_in_seq, train_out_seq = create_sequences(X_train_t, Y_train_t, seq_length)
        val_in_seq,   val_out_seq   = create_sequences(X_val_t,   Y_val_t,   seq_length)
        test_in_seq,  test_out_seq  = create_sequences(X_test_t,  Y_test_t,  seq_length)

        train_loader = DataLoader(TensorDataset(train_in_seq, train_out_seq), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(val_in_seq,   val_out_seq),   batch_size=batch_size)
        test_loader  = DataLoader(TensorDataset(test_in_seq,  test_out_seq),  batch_size=batch_size)

        # Train/eval each model for this run
        for m_name in model_names:
            print(f"\nTraining {m_name} ...")
            # Build model
            model = build_model(m_name, input_dim, output_dim).to(device)
            # For huge pretrained models, freeze the backbone as needed.
            if m_name == "DeepSeek Pretrained":
                for param in model.model.parameters():
                    param.requires_grad = False
                # Ensure the additional layers remain trainable.
                for param in model.input_proj.parameters():
                    param.requires_grad = True
                for param in model.router.parameters():
                    param.requires_grad = True
                for param in model.output_proj.parameters():
                    param.requires_grad = True

            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            # Evaluate on test set
            preds = get_predictions(model, test_loader, device)
            final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
            rmse_val = compute_rmse(final_preds, Y_test)
            final_all_rmse[seq_length][m_name].append(rmse_val)
            print(f"{m_name} RMSE = {rmse_val:.4f}")
            # Save predictions in the base directory so you can re-plot
            save_name = f"fish{fish_num}_{m_name.replace(' ', '_')}_run{run_idx}_seq{seq_length}.npy"
            np.save(os.path.join(RESULTS_DIR, save_name), final_preds)
            print(f"Saved predictions: {save_name}")

            # --- Free up GPU memory after finishing this model ---
            del model
            torch.cuda.empty_cache()

# =============================================================================
# 4) Bar Plots for each seq_length
# =============================================================================
for seq_length in seq_lengths:
    means = []
    sems = []
    for m_name in model_names:
        vals = np.array(final_all_rmse[seq_length][m_name])
        means.append(vals.mean())
        sems.append(vals.std() / np.sqrt(len(vals)))

    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, yerr=sems, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel("RMSE")
    ax.set_title(f"Fish {fish_num} - Seq Length {seq_length}\nMean ± SEM over {num_runs} runs")
    plt.tight_layout()
    fig_name = f"fish{fish_num}_rmse_barplot_seq{seq_length}.png"
    plt.savefig(os.path.join(RESULTS_DIR, fig_name))
    plt.close()
    print(f"Saved bar plot: {fig_name}")

# =============================================================================
# 5) Significance Testing (Wilcoxon Signed-Rank)
#    Only compare each Pretrained vs. its Untrained counterpart
# =============================================================================
pairs = [
    ("DeepSeek Pretrained", "DeepSeek Vanilla"),
    ("GPT2 Pretrained",     "GPT2 Vanilla"),
    ("BERT Pretrained",     "BERT Vanilla"),
]

sig_results = []
sig_results.append(f"Significance Testing (Wilcoxon) for Fish {fish_num}:\n")
for seq_length in seq_lengths:
    sig_results.append(f"Sequence Length {seq_length}:\n")
    for (pre_m, van_m) in pairs:
        pre_vals = np.array(final_all_rmse[seq_length][pre_m])
        van_vals = np.array(final_all_rmse[seq_length][van_m])
        stat, p_val = wilcoxon(pre_vals, van_vals, alternative='two-sided')
        sig_results.append(
            f"  {pre_m} vs {van_m} => p-value = {p_val:.4e}"
        )
    sig_results.append("")

sig_file = os.path.join(RESULTS_DIR, "significance_results_wilcoxon.txt")
with open(sig_file, "w") as f:
    f.write("\n".join(sig_results))

print(f"Significance results saved: {sig_file}")
print("Done!")
