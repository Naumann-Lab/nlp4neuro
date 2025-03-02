#!/usr/bin/env python3
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model, GPT2Config, AdamW, AutoModel, AutoConfig, BertModel, BertConfig
from tqdm import tqdm

# =============================================================================
# 0) Setup & Data Loading
# =============================================================================
RESULTS_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment3_results"  # separate folder for experiment 3
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths (adjust if necessary)
fish_num = 9
data_dir = f"/hpc/group/naumannlab/jjm132/data_prepped_for_models"
neural_data = np.load(os.path.join(data_dir, f"fish{fish_num}_neural_data_matched.npy"), allow_pickle=True)[:,:-2]
tail_data   = np.load(os.path.join(data_dir, f"fish{fish_num}_tail_data_matched.npy"), allow_pickle=True)

# Transpose neural data so that each row is a frame
neural_data = neural_data.T  
print("Neural data shape:", neural_data.shape)
print("Tail data shape:", tail_data.shape)
assert neural_data.shape[0] == tail_data.shape[0], "Mismatch in data length"

total_frames = neural_data.shape[0]
train_end = int(0.7 * total_frames)
val_end   = int(0.8 * total_frames)  # 70% train, 10% val, 20% test

X_train = neural_data[:train_end]
X_val   = neural_data[train_end:val_end]
X_test  = neural_data[val_end:]
Y_train = tail_data[:train_end]
Y_val   = tail_data[train_end:val_end]
Y_test  = tail_data[val_end:]

# Save ground truth for evaluation
np.save(os.path.join(RESULTS_DIR, "final_predictions_groundtruth_val.npy"), Y_val)
np.save(os.path.join(RESULTS_DIR, "final_predictions_groundtruth_test.npy"), Y_test)
print("Ground truth for val/test saved.")

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
Y_val_t   = torch.tensor(Y_val, dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test, dtype=torch.float32)

# Get dimensions
input_dim = X_train_t.size(-1)    # number of neurons
output_dim = Y_train_t.size(-1)    # number of tail features
print(f"Input_dim={input_dim}, Output_dim={output_dim}")

# =============================================================================
# 1) Utility Functions
# =============================================================================
def create_sequences(inputs, targets, seq_length):
    sequences_inputs = []
    sequences_targets = []
    for i in range(len(inputs) - seq_length + 1):
        sequences_inputs.append(inputs[i: i + seq_length])
        sequences_targets.append(targets[i: i + seq_length])
    return torch.stack(sequences_inputs), torch.stack(sequences_targets)

def average_sliding_window_predictions(predictions, seq_length, total_length):
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    num_windows, _, out_dim = predictions.shape
    averaged = np.zeros((total_length, out_dim))
    counts = np.zeros(total_length)
    for i in range(num_windows):
        averaged[i: i + seq_length, :] += predictions[i]
        counts[i: i + seq_length] += 1
    averaged /= counts[:, None]
    return averaged

def compute_rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))

def train_model(model, optimizer, train_loader, val_loader, device, num_epochs, use_position_ids=False):
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
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
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False):
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
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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

# =============================================================================
# 2) Embedding Modules
# =============================================================================
class VanillaLinearEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
    def forward(self, x):
        return self.linear(x)

class PositionalLinearEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, max_len=1024):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.register_buffer('pos_encoding', self._get_positional_encoding(max_len, hidden_size), persistent=False)
    def _get_positional_encoding(self, max_len, hidden_size):
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    def forward(self, x):
        x = self.linear(x)
        L = x.size(1)
        x = x + self.pos_encoding[:, :L, :]
        return x

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
        decoded = self.decoder(encoded)
        encoded = encoded.reshape(B, L, -1)
        decoded = decoded.reshape(B, L, input_dim)
        return encoded
    def get_embedding(self, x):
        return self.forward(x)
    def loss(self, x, decoded, encoded):
        recon_loss = torch.mean((x - decoded) ** 2)
        sparsity_loss = torch.mean(torch.abs(encoded))
        return recon_loss + self.sparsity_weight * sparsity_loss

class SpectralEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        nn.init.orthogonal_(self.linear.weight)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Dictionary of embedding variants to test.
embedding_models = {
    "Vanilla": VanillaLinearEmbedding,
    "Positional": PositionalLinearEmbedding,
    "Sparse": SparseAutoencoderEmbedding,
    "Spectral": SpectralEmbedding
}

# =============================================================================
# 3) Model Classes with Embedding
# =============================================================================
# For each model family, the model takes an embedding module instance as input.
# Note: The output dimension of the embedding must match the transformer’s hidden size.

# (A) DeepSeek – pre-trained model has hidden size 4096.
hidden_size_deepseek = 4096
class DeepSeekWithEmbedding(nn.Module):
    def __init__(self, embedding, output_dim):
        super().__init__()
        # The embedding module should output (B, L, hidden_size_deepseek)
        self.embedding = embedding
        self.model = AutoModel.from_pretrained("deepseek-ai/deepseek-coder-7b", trust_remote_code=True)
        self.router = nn.Linear(hidden_size_deepseek, 8)  # using 8 experts
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)
    def forward(self, x):
        x_emb = self.embedding(x)
        outputs = self.model(inputs_embeds=x_emb, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, 2, dim=-1)[1]
        aggregated_output = torch.zeros_like(hidden_states)
        for i in range(2):
            expert_idx = top_k_experts[:, :, i].unsqueeze(-1).expand_as(hidden_states)
            expert_output = torch.gather(hidden_states, dim=-1, index=expert_idx)
            aggregated_output += expert_output / 2
        logits = self.output_proj(aggregated_output)
        return logits

# (B) GPT‑2 – pre-trained GPT‑2 has hidden size 768.
gpt2_hidden_size = 768
class GPT2WithEmbedding(nn.Module):
    def __init__(self, embedding, output_dim):
        super().__init__()
        self.embedding = embedding
        self.transformer = GPT2Model.from_pretrained("gpt2")
        self.output_proj = nn.Linear(gpt2_hidden_size, output_dim)
    def forward(self, x, position_ids=None):
        # Here, x is first passed through the embedding.
        x_emb = self.embedding(x)  # Expected shape: (B, L, gpt2_hidden_size)
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
        self.output_proj = nn.Linear(bert_hidden_size, output_dim)
    def forward(self, x, position_ids=None):
        x_emb = self.embedding(x)  # Expected shape: (B, L, bert_hidden_size)
        outputs = self.bert(inputs_embeds=x_emb)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# =============================================================================
# 4) Training Setup for Embedding Comparisons
# =============================================================================
# Use a fixed sequence length of 20 and training parameters as below.
seq_length = 20
num_epochs = 5
batch_size = 32
lr_default = 1e-4

# Create sliding-window sequences (same for all experiments)
train_seq_x, train_seq_y = create_sequences(X_train_t, Y_train_t, seq_length)
val_seq_x, val_seq_y     = create_sequences(X_val_t, Y_val_t, seq_length)
test_seq_x, test_seq_y   = create_sequences(X_test_t, Y_test_t, seq_length)

train_dataset = TensorDataset(train_seq_x, train_seq_y)
val_dataset   = TensorDataset(val_seq_x, val_seq_y)
test_dataset  = TensorDataset(test_seq_x, test_seq_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# For each model family, we will loop over embedding variants.
# We also save each family’s results in a dedicated subfolder.
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

# Dictionary to store RMSE results for each family and each embedding.
results = {family: {} for family in families}

# Loop over model families and embedding variants.
for family, params in families.items():
    family_folder = os.path.join(RESULTS_DIR, family.lower() + "_embedding_comparisons")
    os.makedirs(family_folder, exist_ok=True)
    ModelClass = params["ModelClass"]
    hidden_size_model = params["hidden_size"]
    for emb_name, emb_class in embedding_models.items():
        print("\n========================================")
        print(f"Training {family} with {emb_name} Embedding...")
        print("========================================")
        # Instantiate the embedding module.
        # Ensure its output dimension equals the model’s hidden size.
        embedding_instance = emb_class(input_dim, hidden_size_model)
        # Instantiate the model.
        model = ModelClass(embedding_instance, output_dim).to(device)
        # Freeze the backbone transformer parameters.
        if family == "DeepSeek":
            for param in model.model.parameters():
                param.requires_grad = False
        elif family == "GPT2":
            for param in model.transformer.parameters():
                param.requires_grad = False
        elif family == "BERT":
            for param in model.bert.parameters():
                param.requires_grad = False

        # Set up optimizer (trainable parameters include the embedding, and any head layers)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
        # Train the model
        train_model(model, optimizer, train_loader, val_loader, device, num_epochs, use_position_ids=False)
        # Get predictions on test set
        preds, _ = get_predictions(model, test_loader, device, use_position_ids=False)
        final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
        # Save predictions in a subfolder for this embedding variant.
        emb_folder = os.path.join(family_folder, emb_name.lower())
        os.makedirs(emb_folder, exist_ok=True)
        np.save(os.path.join(emb_folder, f"final_predictions_{family.lower()}_{emb_name.lower()}_test.npy"), final_preds)
        rmse = compute_rmse(final_preds, Y_test)
        results[family][emb_name] = rmse
        print(f"{family} with {emb_name} Embedding RMSE: {rmse:.4f}")

# =============================================================================
# 5) Final Grouped Bar Plot (Overall RMSE Across Embedding Variants)
# =============================================================================
# For each family, plot a grouped bar plot comparing embedding types.
for family, emb_results in results.items():
    emb_names = list(emb_results.keys())
    rmse_vals = [emb_results[name] for name in emb_names]
    x = np.arange(len(emb_names))
    width = 0.6
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, rmse_vals, width=width, color='skyblue')
    ax.set_xlabel("Embedding Type")
    ax.set_ylabel("Overall RMSE")
    ax.set_title(f"{family} Embedding Comparison (Sequence Length = {seq_length})")
    ax.set_xticks(x)
    ax.set_xticklabels(emb_names)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"grouped_rmse_{family.lower()}_embedding_comparison.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Final grouped bar plot saved to {plot_path}")

print("Experiment 3 completed: All models trained with different embeddings, predictions saved, and plots generated.")
