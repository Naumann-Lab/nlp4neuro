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
RESULTS_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment3_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths (adjust if necessary)
fish_num = 9
data_dir = f"/hpc/group/naumannlab/jjm132/data_prepped_for_models"
neural_data = np.load(
    os.path.join(data_dir, f"fish{fish_num}_neural_data_matched.npy"), 
    allow_pickle=True
)[:, :-2]
tail_data = np.load(
    os.path.join(data_dir, f"fish{fish_num}_tail_data_matched.npy"), 
    allow_pickle=True
)

# Transpose neural data so that each row is a frame
neural_data = neural_data.T
print("Neural data shape:", neural_data.shape)
print("Tail data shape:", tail_data.shape)
assert neural_data.shape[0] == tail_data.shape[0], "Mismatch in data length"

total_frames = neural_data.shape[0]
train_end = int(0.7 * total_frames)
val_end = int(0.8 * total_frames)  # 70% train, 10% val, 20% test

X_train = neural_data[:train_end]
X_val   = neural_data[train_end:val_end]
X_test  = neural_data[val_end:]
Y_train = tail_data[:train_end]
Y_val   = tail_data[train_end:val_end]
Y_test  = tail_data[val_end:]

# Save ground truth for final reference
np.save(os.path.join(RESULTS_DIR, "groundtruth_val.npy"), Y_val)
np.save(os.path.join(RESULTS_DIR, "groundtruth_test.npy"), Y_test)
print("Ground truth for val/test saved in", RESULTS_DIR)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)

# Dimensions
input_dim  = X_train_t.size(-1)   # number of neurons
output_dim = Y_train_t.size(-1)   # number of tail features
print(f"Input_dim={input_dim}, Output_dim={output_dim}")


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
    Adds a trainable relative position bias or encoding to each token’s embedding.
    For simplicity, we’ll do a small trainable parameter table based on distance.
    """
    def __init__(self, input_dim, hidden_size, max_dist=32):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        # Relative positions up to +/- max_dist
        # table size: 2*max_dist+1
        self.max_dist = max_dist
        num_positions = 2*max_dist + 1
        self.rel_embed = nn.Embedding(num_positions, hidden_size)
    def forward(self, x):
        # x shape: (B, L, input_dim)
        x_lin = self.linear(x)
        B, L, _ = x_lin.shape
        # Add relative position encoding
        # index for each pair (i, j) the distance = j - i, clipped within [-max_dist, +max_dist]
        # We'll do it tokenwise, but simpler is just to add a row-wise shift for each token
        # i.e. for token i, we add the embedding for distance = 0
        # for token i+1, we add the embedding for distance = +1, etc. => a rough approach
        # A more thorough approach might require a full NxN attention matrix, but here we keep it simpler:
        positions = torch.arange(L, device=x_lin.device)
        # We'll expand to shape (B, L, hidden_size)
        # to approximate a "diagonal" relative shift
        # shift in range [-max_dist, max_dist]
        offset = positions - positions[0]  # from 0, 1, 2, ...
        offset = torch.clamp(offset, -self.max_dist, self.max_dist)
        offset_index = offset + self.max_dist  # shift to [0..2*max_dist]
        # shape (L,) => we broadcast to (B, L, hidden_size)
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
        # We do not decode unless we want the autoencoder loss. 
        # For the final model, we just return the encoded embeddings.
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
# Use 4-bit quantization for DeepSeek with BitsAndBytes
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0
)

# (A) DeepSeek – pre-trained model has hidden size 4096.
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
        # Freeze the main model:
        for param in self.model.parameters():
            param.requires_grad = False

        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)

    def forward(self, x):
        # x shape: (B, L, input_dim)
        x_emb = self.embedding(x)  # (B, L, 4096)
        outputs = self.model(inputs_embeds=x_emb, output_hidden_states=False)
        # last_hidden_state is outputs.last_hidden_state
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits


# (B) GPT‑2 – pre-trained GPT‑2 has hidden size 768.
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
        x_emb = self.embedding(x)  # shape: (B, L, 768)
        outputs = self.transformer(inputs_embeds=x_emb, position_ids=position_ids)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits


# (C) BERT – pre-trained BERT (bert-base-uncased) has hidden size 768.
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
        x_emb = self.embedding(x)  # (B, L, 768)
        outputs = self.bert(inputs_embeds=x_emb)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits


###############################################################################
# 4) Training Setup for Embedding Comparisons
###############################################################################
# We do 10 runs, each with a new seed or randomness context
NUM_RUNS = 10
seq_length = 20
num_epochs = 3     # can reduce if memory or time is an issue
batch_size = 2     # reduce to avoid OOM
lr_default = 1e-4

# Create sliding-window sequences (same for all experiments)
train_seq_x, train_seq_y = create_sequences(X_train_t, Y_train_t, seq_length)
val_seq_x,   val_seq_y   = create_sequences(X_val_t,   Y_val_t,   seq_length)
test_seq_x,  test_seq_y  = create_sequences(X_test_t,  Y_test_t,  seq_length)

train_dataset = TensorDataset(train_seq_x, train_seq_y)
val_dataset   = TensorDataset(val_seq_x,   val_seq_y)
test_dataset  = TensorDataset(test_seq_x,  test_seq_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

# Add embeddings in a dictionary
embedding_models = {
    "Vanilla":   VanillaLinearEmbedding,
    "Positional": PositionalLinearEmbedding,
    "RelativePos": RelativePositionEmbedding,
    "Sparse":    SparseAutoencoderEmbedding,
    "Spectral":  SpectralEmbedding
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

# We will store all RMSE runs in a dict: results[family][embedding] = list of RMSEs
results = {
    f: {emb: [] for emb in embedding_models.keys()}
    for f in families.keys()
}


###############################################################################
# 5) Main loop: 10 runs x 3 families x 5 embeddings
###############################################################################
for run_idx in range(1, NUM_RUNS+1):
    print(f"\n********** RUN {run_idx} / {NUM_RUNS} **********")
    run_folder = os.path.join(RESULTS_DIR, f"run_{run_idx}")
    os.makedirs(run_folder, exist_ok=True)

    # (Optional) set a new random seed for each run
    # This could be truly random or fixed for reproducibility
    seed = 1234 + run_idx
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for family_name, fam_params in families.items():
        family_folder = os.path.join(run_folder, family_name.lower() + "_embedding_comparisons")
        os.makedirs(family_folder, exist_ok=True)

        # For each embedding
        for emb_name, emb_class in embedding_models.items():
            print(f"\n=== [{family_name}] Run {run_idx}: Embedding = {emb_name} ===")

            # Instantiate embedding
            hidden_size = fam_params["hidden_size"]
            embedding_instance = emb_class(input_dim, hidden_size)

            # Instantiate model
            ModelClass = fam_params["ModelClass"]
            model = ModelClass(embedding_instance, output_dim).to(device)

            # Only embedding + final layer are trainable (the base model is frozen in constructor).
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_default
            )

            # Train
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs, use_position_ids=False)

            # Predict on test set
            preds_tensor = get_predictions(model, test_loader, device, use_position_ids=False)
            final_preds = average_sliding_window_predictions(preds_tensor, seq_length, len(X_test_t))

            # Compute RMSE
            rmse_val = compute_rmse(final_preds, Y_test)
            results[family_name][emb_name].append(rmse_val)
            print(f">>>> {family_name} + {emb_name} RMSE = {rmse_val:.5f}")

            # Save predictions
            emb_subfolder = os.path.join(family_folder, emb_name.lower())
            os.makedirs(emb_subfolder, exist_ok=True)

            save_preds_path = os.path.join(
                emb_subfolder, 
                f"{family_name.lower()}_{emb_name.lower()}_preds_run{run_idx}.npy"
            )
            np.save(save_preds_path, final_preds)

            # Also save ground truth for reference (though it's the same each run)
            # but we can store for completeness in the same folder
            save_truth_path = os.path.join(
                emb_subfolder,
                f"{family_name.lower()}_{emb_name.lower()}_groundtruth.npy"
            )
            if not os.path.exists(save_truth_path):
                np.save(save_truth_path, Y_test)


###############################################################################
# 6) After all runs: bar plots and significance tests
###############################################################################
# We create one bar plot per family, comparing embeddings. We'll show mean ± SEM.

final_plots_folder = os.path.join(RESULTS_DIR, "final_plots_and_stats")
os.makedirs(final_plots_folder, exist_ok=True)

for family_name in families.keys():
    emb_names = list(results[family_name].keys())
    # each is a list of 10 RMSE values
    rmse_lists = [results[family_name][emb] for emb in emb_names]
    means = [np.mean(r) for r in rmse_lists]
    stderrs = [np.std(r)/np.sqrt(len(r)) for r in rmse_lists]

    # Bar plot
    x = np.arange(len(emb_names))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x, means, yerr=stderrs, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(emb_names, rotation=15)
    ax.set_ylabel("RMSE")
    ax.set_title(f"{family_name} - Embedding Comparison (Mean ± SEM, n={NUM_RUNS})")

    plt.tight_layout()
    plot_path = os.path.join(final_plots_folder, f"{family_name.lower()}_embeddings_barplot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"{family_name} bar plot saved to: {plot_path}")

    # Wilcoxon significance test across embeddings
    # We'll do a pairwise test: e.g. compare each embedding vs. each other. 
    # Or you can pick a baseline. For brevity, let's pairwise test each embedding vs. each other.
    sig_file = os.path.join(final_plots_folder, f"{family_name.lower()}_wilcoxon_significance.txt")
    with open(sig_file, "w") as f:
        f.write(f"Significance tests for {family_name}\n")
        f.write("Pairwise Wilcoxon tests across embeddings:\n\n")
        for i in range(len(emb_names)):
            for j in range(i+1, len(emb_names)):
                emb1, emb2 = emb_names[i], emb_names[j]
                data1, data2 = results[family_name][emb1], results[family_name][emb2]
                stat, pval = wilcoxon(data1, data2)
                f.write(f"{emb1} vs. {emb2}: p-value = {pval:.4e}\n")
    print(f"{family_name} significance results saved to: {sig_file}")

print("Experiment 3 completed: 10 runs with 3 families x 5 embeddings each, predictions/plots/stats saved.")
