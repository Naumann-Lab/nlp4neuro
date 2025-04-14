#!/usr/bin/env python3
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    GPT2Model, GPT2Config, AdamW, AutoModel, AutoConfig, BertModel, BertConfig,
    BitsAndBytesConfig
)

from scipy.stats import wilcoxon
from tqdm import tqdm

###############################################################################
# 0) Setup & Data Loading
###############################################################################
RESULTS_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment3b_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths (adjust if necessary)
fish_num = 9
data_dir = f"/hpc/group/naumannlab/jjm132/data_prepped_for_models"
neural_data_orig = np.load(
    os.path.join(data_dir, f"fish{fish_num}_neural_data_matched.npy"), 
    allow_pickle=True
)[:, :-2]  # shape: (num_neurons, num_timepoints) in your original storage
tail_data_orig = np.load(
    os.path.join(data_dir, f"fish{fish_num}_tail_data_matched.npy"), 
    allow_pickle=True
)

# Transpose so each row is a frame
neural_data_orig = neural_data_orig.T  # shape: (num_timepoints, num_neurons)
print("Neural data shape:", neural_data_orig.shape)
print("Tail data shape:", tail_data_orig.shape)
assert neural_data_orig.shape[0] == tail_data_orig.shape[0], "Mismatch in data length"

total_frames = neural_data_orig.shape[0]
train_end = int(0.7 * total_frames)
val_end   = int(0.8 * total_frames)

X_train_full = neural_data_orig[:train_end]
X_val_full   = neural_data_orig[train_end:val_end]
X_test_full  = neural_data_orig[val_end:]
Y_train_full = tail_data_orig[:train_end]
Y_val_full   = tail_data_orig[train_end:val_end]
Y_test_full  = tail_data_orig[val_end:]

# Save ground truths (without removal) for final reference
np.save(os.path.join(RESULTS_DIR, "groundtruth_val.npy"), Y_val_full)
np.save(os.path.join(RESULTS_DIR, "groundtruth_test.npy"), Y_test_full)

print("Ground truth for val/test saved in", RESULTS_DIR)


###############################################################################
# 1) Utility Functions
###############################################################################
def create_sequences(inputs, targets, seq_length):
    """
    Make sliding-window sequences of length seq_length.
    outputs: (N - seq_length+1, seq_length, input_dim)
    """
    seq_in, seq_out = [], []
    for i in range(len(inputs) - seq_length + 1):
        seq_in.append(inputs[i : i + seq_length])
        seq_out.append(targets[i : i + seq_length])
    return torch.stack(seq_in), torch.stack(seq_out)

def average_sliding_window_predictions(predictions, seq_length, total_length):
    """
    Convert overlapping predictions back to frame-aligned predictions
    by averaging the overlapping windows.
    predictions: shape (num_windows, seq_length, output_dim).
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    num_windows, _, out_dim = predictions.shape
    averaged = np.zeros((total_length, out_dim))
    counts = np.zeros(total_length)
    for i in range(num_windows):
        averaged[i : i + seq_length, :] += predictions[i]
        counts[i : i + seq_length] += 1
    averaged /= counts[:, None]
    return averaged

def compute_rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))

def train_model(model, optimizer, train_loader, val_loader, device, num_epochs, use_position_ids=False):
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if use_position_ids:
                bsz, seq_len, _ = inputs.shape
                position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len).to(device)
                outputs = model(inputs, position_ids=position_ids)
            else:
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
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                if use_position_ids:
                    bsz, seq_len, _ = inputs.shape
                    position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len).to(device)
                    outputs = model(inputs, position_ids=position_ids)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

def get_predictions(model, data_loader, device, use_position_ids=False):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            if use_position_ids:
                bsz, seq_len, _ = inputs.shape
                position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len).to(device)
                outputs = model(inputs, position_ids=position_ids)
            else:
                outputs = model(inputs)
            preds_list.append(outputs.cpu())
    return torch.cat(preds_list, dim=0)


###############################################################################
# 2) Embedding Modules
###############################################################################
class VanillaLinearEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
    def forward(self, x):
        return self.linear(x)

class PositionalLinearEmbedding(nn.Module):
    """
    Adds absolute positional encodings.
    """
    def __init__(self, input_dim, hidden_size, max_len=1024):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.register_buffer('pos_encoding', self._get_positional_encoding(max_len, hidden_size), persistent=False)
    def _get_positional_encoding(self, max_len, hidden_size):
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape (1, max_len, hidden_size)
    def forward(self, x):
        # x shape: (B, L, input_dim)
        x = self.linear(x)
        L = x.size(1)
        x = x + self.pos_encoding[:, :L, :]
        return x

class RelativePositionEmbedding(nn.Module):
    """
    A small trainable relative position approach for demonstration.
    """
    def __init__(self, input_dim, hidden_size, max_dist=32):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.max_dist = max_dist
        num_positions = 2*max_dist + 1
        self.rel_embed = nn.Embedding(num_positions, hidden_size)
    def forward(self, x):
        x_lin = self.linear(x)
        B, L, _ = x_lin.shape
        positions = torch.arange(L, device=x_lin.device)
        offset = positions - positions[0]
        offset = torch.clamp(offset, -self.max_dist, self.max_dist)
        offset_index = offset + self.max_dist
        pos_emb = self.rel_embed(offset_index)  # shape (L, hidden_size)
        pos_emb = pos_emb.unsqueeze(0).expand(B, L, -1)
        return x_lin + pos_emb

class SparseAutoencoderEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, sparsity_weight=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_dim)
        self.activation = nn.ReLU()
        self.sparsity_weight = sparsity_weight
    def forward(self, x):
        B, L, input_dim = x.shape
        x_flat = x.reshape(B * L, input_dim)
        encoded = self.activation(self.encoder(x_flat))
        encoded = encoded.reshape(B, L, -1)
        return encoded

class SpectralEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        nn.init.orthogonal_(self.linear.weight)  # orthonormal init
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


###############################################################################
# 3) Model Classes with Embedding
###############################################################################
# Use 4-bit quantization for DeepSeek
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0
)

# (A) DeepSeek – hidden size 4096
hidden_size_deepseek = 4096
class DeepSeekWithEmbedding(nn.Module):
    def __init__(self, embedding, output_dim):
        super().__init__()
        self.embedding = embedding
        self.model = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)

    def forward(self, x):
        x_emb = self.embedding(x)
        outputs = self.model(inputs_embeds=x_emb, output_hidden_states=False)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# (B) GPT-2 – hidden size 768
gpt2_hidden_size = 768
class GPT2WithEmbedding(nn.Module):
    def __init__(self, embedding, output_dim):
        super().__init__()
        self.embedding = embedding
        self.transformer = GPT2Model.from_pretrained("gpt2")
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.output_proj = nn.Linear(gpt2_hidden_size, output_dim)
    def forward(self, x, position_ids=None):
        x_emb = self.embedding(x)
        outputs = self.transformer(inputs_embeds=x_emb, position_ids=position_ids)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# (C) BERT – hidden size 768
bert_hidden_size = 768
class BERTWithEmbedding(nn.Module):
    def __init__(self, embedding, output_dim):
        super().__init__()
        self.embedding = embedding
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.output_proj = nn.Linear(bert_hidden_size, output_dim)
    def forward(self, x, position_ids=None):
        x_emb = self.embedding(x)
        outputs = self.bert(inputs_embeds=x_emb)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits


###############################################################################
# 4) Setup for 2 Scenarios: top50_removed vs. random50_removed
###############################################################################
# Identify top 50 “most active” neurons by sum of abs across time.
abs_sums = np.sum(np.abs(neural_data_orig), axis=0)  # shape (num_neurons,)
top50_indices = np.argsort(abs_sums)[-50:]  # these are the 50 biggest

def remove_neurons_from_data(X_full, remove_indices):
    """
    X_full: shape (time, num_neurons)
    remove_indices: list or array of neuron indices to remove
    returns: X_new with shape (time, num_neurons - len(remove_indices))
    """
    keep_indices = [i for i in range(X_full.shape[1]) if i not in remove_indices]
    return X_full[:, keep_indices]

###############################################################################
# 5) Experiment Parameters
###############################################################################
NUM_RUNS    = 10
seq_length  = 20
num_epochs  = 3
batch_size  = 2
lr_default  = 1e-4

# Embeddings
embedding_models = {
    "Vanilla":      VanillaLinearEmbedding,
    "Positional":   PositionalLinearEmbedding,
    "RelativePos":  RelativePositionEmbedding,
    "Sparse":       SparseAutoencoderEmbedding,
    "Spectral":     SpectralEmbedding
}

# Model families
families = {
    "DeepSeek": {
        "ModelClass": DeepSeekWithEmbedding,
        "hidden_size": hidden_size_deepseek
    },
    "GPT2": {
        "ModelClass": GPT2WithEmbedding,
        "hidden_size": gpt2_hidden_size
    },
    "BERT": {
        "ModelClass": BERTWithEmbedding,
        "hidden_size": bert_hidden_size
    }
}

# For storing results: results_scenario[family][embedding] = list of 10 RMSEs
results_top50    = {f: {emb: [] for emb in embedding_models} for f in families}
results_random50 = {f: {emb: [] for emb in embedding_models} for f in families}

###############################################################################
# 6) Main Loop: Each scenario => 10 runs => each family => each embedding
###############################################################################
SCENARIOS = ["top50_removed", "random50_removed"]
for scenario in SCENARIOS:
    scenario_folder = os.path.join(RESULTS_DIR, scenario)
    os.makedirs(scenario_folder, exist_ok=True)
    
    for run_idx in range(1, NUM_RUNS+1):
        print(f"\n****** {scenario} | RUN {run_idx}/{NUM_RUNS} ******")
        run_folder = os.path.join(scenario_folder, f"run_{run_idx}")
        os.makedirs(run_folder, exist_ok=True)

        # For random scenario, pick 50 at random each run
        if scenario == "random50_removed":
            np.random.seed(999 + run_idx)
            all_neurons = np.arange(neural_data_orig.shape[1])
            remove_indices = np.random.choice(all_neurons, size=50, replace=False)
        else:
            # top50_removed
            remove_indices = top50_indices

        # Prepare data with these 50 removed
        X_train = remove_neurons_from_data(X_train_full, remove_indices)
        X_val   = remove_neurons_from_data(X_val_full,   remove_indices)
        X_test  = remove_neurons_from_data(X_test_full,  remove_indices)

        # Convert to torch
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
        X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train_full, dtype=torch.float32)
        Y_val_t   = torch.tensor(Y_val_full,   dtype=torch.float32)
        Y_test_t  = torch.tensor(Y_test_full,  dtype=torch.float32)

        # Make seq windows
        train_seq_x, train_seq_y = create_sequences(X_train_t, Y_train_t, seq_length)
        val_seq_x,   val_seq_y   = create_sequences(X_val_t,   Y_val_t,   seq_length)
        test_seq_x,  test_seq_y  = create_sequences(X_test_t,  Y_test_t,  seq_length)

        train_dataset = TensorDataset(train_seq_x, train_seq_y)
        val_dataset   = TensorDataset(val_seq_x,   val_seq_y)
        test_dataset  = TensorDataset(test_seq_x,  test_seq_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

        input_dim_scenario = X_train.shape[1]  # updated input dimension

        for family_name, fam_params in families.items():
            family_folder = os.path.join(run_folder, family_name.lower() + "_embedding_comparisons")
            os.makedirs(family_folder, exist_ok=True)

            for emb_name, emb_class in embedding_models.items():
                print(f"\n=== [{scenario.upper()}] {family_name} | Run {run_idx}: Embedding = {emb_name} ===")
                
                hidden_size = fam_params["hidden_size"]
                embedding_instance = emb_class(input_dim_scenario, hidden_size)
                ModelClass = fam_params["ModelClass"]
                model = ModelClass(embedding_instance, output_dim).to(device)

                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr_default
                )

                # Train
                train_model(model, optimizer, train_loader, val_loader, device, num_epochs, use_position_ids=False)

                # Predict
                preds_tensor = get_predictions(model, test_loader, device, use_position_ids=False)
                final_preds = average_sliding_window_predictions(preds_tensor, seq_length, len(X_test_t))

                # Compute RMSE
                rmse_val = compute_rmse(final_preds, Y_test_full)
                if scenario == "top50_removed":
                    results_top50[family_name][emb_name].append(rmse_val)
                else:
                    results_random50[family_name][emb_name].append(rmse_val)

                print(f">>>> {scenario.upper()} => {family_name} + {emb_name} RMSE = {rmse_val:.5f}")

                # Save predictions
                emb_subfolder = os.path.join(family_folder, emb_name.lower())
                os.makedirs(emb_subfolder, exist_ok=True)

                save_preds_path = os.path.join(
                    emb_subfolder, 
                    f"{family_name.lower()}_{emb_name.lower()}_preds_run{run_idx}.npy"
                )
                np.save(save_preds_path, final_preds)

                # Save ground truth (same each run, but for completeness):
                save_truth_path = os.path.join(
                    emb_subfolder,
                    f"{family_name.lower()}_{emb_name.lower()}_groundtruth.npy"
                )
                if not os.path.exists(save_truth_path):
                    np.save(save_truth_path, Y_test_full)


###############################################################################
# 7) After all runs: bar plots and significance tests
###############################################################################
plots_and_stats_folder = os.path.join(RESULTS_DIR, "final_plots_and_stats")
os.makedirs(plots_and_stats_folder, exist_ok=True)

# We'll create two bar plots per family: (1) top50_removed, (2) random50_removed
# Then we'll do a significance test comparing top50_removed vs random50_removed
# for each family+embedding.

for family_name in families:
    emb_names = list(embedding_models.keys())

    # For top50_removed
    rmse_lists_top = [results_top50[family_name][emb] for emb in emb_names]   # each is length 10
    means_top = [np.mean(r) for r in rmse_lists_top]
    stderrs_top = [np.std(r)/np.sqrt(len(r)) for r in rmse_lists_top]

    # For random50_removed
    rmse_lists_rand = [results_random50[family_name][emb] for emb in emb_names]
    means_rand = [np.mean(r) for r in rmse_lists_rand]
    stderrs_rand = [np.std(r)/np.sqrt(len(r)) for r in rmse_lists_rand]

    # Plot side-by-side bars
    x = np.arange(len(emb_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    bar1 = ax.bar(x - width/2, means_top, width, yerr=stderrs_top, capsize=3, label="Top50 Removed")
    bar2 = ax.bar(x + width/2, means_rand, width, yerr=stderrs_rand, capsize=3, label="Random50 Removed")

    ax.set_xticks(x)
    ax.set_xticklabels(emb_names, rotation=20)
    ax.set_ylabel("RMSE")
    ax.set_title(f"{family_name}: Top50 vs Random50 Removal (Mean ± SEM, n=10)")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(plots_and_stats_folder, f"{family_name.lower()}_top50_vs_random50_barplot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved bar plot for {family_name} to {plot_path}")

    # Significance test: top50_removed vs random50_removed, per embedding
    sig_file = os.path.join(plots_and_stats_folder, f"{family_name.lower()}_top50_vs_random50_significance.txt")
    with open(sig_file, "w") as f:
        f.write(f"Significance tests for {family_name}: Top50 vs Random50\n\n")
        for i, emb in enumerate(emb_names):
            data_top = results_top50[family_name][emb]
            data_rand = results_random50[family_name][emb]
            stat, pval = wilcoxon(data_top, data_rand)
            f.write(f"{emb}: p-value = {pval:.4e}\n")
    print(f"Significance results saved for {family_name} => {sig_file}")

print("Experiment 3b completed. Results, predictions, bar plots, significance tests saved.")
