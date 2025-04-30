import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2Model, AdamW, AutoModel, BertModel, BertConfig, BitsAndBytesConfig
from scipy.stats import mannwhitneyu

# Quantization config for DeepSeek model
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,  # or load_in_4bit=True, if desired
    llm_int8_threshold=6.0  # optional, sets threshold for zero out
)

# =============================================================================
# 0) Setup
# =============================================================================

# Base results directory for experiment 1 (results saved here will not conflict with others)
BASE_SAVE_DIR = os.path.join(os.getcwd(), f"/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment_6")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Run experiment for fish 9, 10, 11, 12, and 13.
fish_list = [9, 10, 11, 12, 13]

# Model parameters
seq_lengths = [5, 10, 15, 20]
num_epochs = 10
batch_size = 32
lr_default = 1e-4
num_runs = 10  # number of repeats of the pipeline

# =============================================================================
# Main loop: iterate over each fish
# =============================================================================
for fish_num in fish_list:
    print("\n====================================")
    print(f"Processing Fish {fish_num}")
    print("====================================")

    # Define a folder for this fish's experiment results inside BASE_SAVE_DIR
    fish_save_dir = os.path.join(BASE_SAVE_DIR, f"fish{fish_num}")
    os.makedirs(fish_save_dir, exist_ok=True)

    # =============================================================================
    # 1) Load Data & Train/Val/Test Split for current fish
    # =============================================================================
    neural_data = np.load(f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish_num}_neural_data_matched.npy", allow_pickle=True)[:,:-2]
    tail_data = np.load(f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish_num}_tail_data_matched.npy", allow_pickle=True)

    # Transpose neural data so that shape becomes (num_frames, num_neurons)
    neural_data = neural_data.T
    print("Neural data shape:", neural_data.shape)
    print("Tail data shape:", tail_data.shape)

    assert neural_data.shape[0] == tail_data.shape[0], "Mismatch in data length"

    total_frames = neural_data.shape[0]
    train_end = int(0.7 * total_frames)
    val_end = int(0.8 * total_frames)  # 70% train, 10% val, 20% test

    X_train = neural_data[:train_end]
    X_val = neural_data[train_end:val_end]
    X_test = neural_data[val_end:]
    Y_train = tail_data[:train_end]
    Y_val = tail_data[train_end:val_end]
    Y_test = tail_data[val_end:]

    # Save ground truth for later evaluation for this fish
    np.save(os.path.join(fish_save_dir, f"fish{fish_num}_final_predictions_groundtruth_val.npy"), Y_val)
    np.save(os.path.join(fish_save_dir, f"fish{fish_num}_final_predictions_groundtruth_test.npy"), Y_test)
    print("Ground truth for val/test saved.")

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    # Get dimensions
    input_dim = X_train_t.size(-1)
    output_dim = Y_train_t.size(-1)
    print(f"Input_dim={input_dim}, Output_dim={output_dim}")

    # =============================================================================
    # 2) Helper Functions
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
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0
            for inputs, targets in train_loader:
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
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
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
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        return train_losses, val_losses

    def get_predictions(model, data_loader, device, use_position_ids=False):
        model.eval()
        preds_list = []
        targets_list = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if use_position_ids:
                    bsz, seq_len, _ = inputs.shape
                    position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len).to(device)
                    outputs = model(inputs, position_ids=position_ids)
                else:
                    outputs = model(inputs)
                preds_list.append(outputs.cpu())
                targets_list.append(targets.cpu())
        all_preds = torch.cat(preds_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        return all_preds, all_targets

    # ──────────────────────────────────────────────────────────────────────────────
# 0) KEEP everything above “# 3) Model Definitions” exactly as-is
# ──────────────────────────────────────────────────────────────────────────────

# 3) Model Definitions  ──► **delete everything except DeepSeek-V3 MoE**.
hidden_size_deepseek, num_experts, top_k = 4096, 8, 2
class DeepSeekV3MoE(nn.Module):
    ...

# ──────────────────────────────────────────────────────────────────────────────
# 4) Main loop – strip to DeepSeek only
# ──────────────────────────────────────────────────────────────────────────────
seq_lengths = [10]            # ← pick one or keep the full list if needed
num_runs    = 5               # ← reduce if GPU RAM is tight

for run in range(1, num_runs + 1):
    for seq_length in seq_lengths:
        ...                                   # create sequences & loaders

        # ——— TRAIN / FINE-TUNE ————————————————————————————————
        model = DeepSeekV3MoE(input_dim, output_dim).to(device)

        # freeze DeepSeek backbone
        for p in model.model.parameters():      p.requires_grad = False
        # train input-proj, router, output-proj
        optim_ds = AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=1e-4
        )
        train_model(model, optim_ds, train_loader, val_loader,
                    device, num_epochs, use_position_ids=False)

        # ——— SAVE PREDICTIONS & RMSE ————————————————————————————
        preds, _ = get_predictions(model, test_loader, device)
        final_preds = average_sliding_window_predictions(
            preds, seq_length, len(X_test_t)
        )
        np.save(os.path.join(seq_folder,
               f"fish{fish_num}_deepseek_preds_run{run}.npy"), final_preds)
        rmse = compute_rmse(final_preds, Y_test)
        print(f"DeepSeek RMSE (fish {fish_num}, run {run}): {rmse:.4f}")

        # ─────────────────────────────────────────────────────────────
        # 5) Neuron-importance attribution  (gradient ⨉ input saliency)
        # ─────────────────────────────────────────────────────────────
        def feature_importance(model, data_loader):
            model.eval()
            all_imp = torch.zeros(input_dim, device=device)
            tot_tokens = 0
            for X, _ in data_loader:
                X = X.to(device).requires_grad_(True)
                out = model(X).sum()          # sum over batch, seq, dim
                out.backward()
                # |grad ⨉ input| averaged over seq-steps & batch
                sal = (X.grad * X).abs().mean(dim=(0,1))
                all_imp += sal.detach()
                tot_tokens += 1
            return (all_imp / tot_tokens).cpu().numpy()

        imp_scores = feature_importance(model, val_loader)
        np.save(os.path.join(seq_folder,
               f"fish{fish_num}_importance_run{run}.npy"), imp_scores)

        # quick text report of top-10 neurons
        top_idx = imp_scores.argsort()[-10:][::-1]
        print("Top-10 contributing neurons:", top_idx.tolist(),
              "with scores:", imp_scores[top_idx])

# ──────────────────────────────────────────────────────────────────────────────
# 6) OPTIONAL – visual check: bar plot of top-k
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
k = 20
plt.figure(figsize=(8,4))
plt.bar(range(k), imp_scores[top_idx[:k]])
plt.xticks(range(k), top_idx[:k], rotation=45)
plt.ylabel("saliency (|grad × input|)")
plt.title(f"Fish {fish_num} – Top-{k} neuron importances")
plt.tight_layout()
plt.savefig(os.path.join(seq_folder, f"fish{fish_num}_top{k}_saliency.png"))
