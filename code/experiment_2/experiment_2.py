import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # For headless environments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Key changes below!
# -----------------------------
# 1) Switch from 8-bit to 4-bit quantization for DeepSeek
# 2) Freeze all pretrained model parameters except for the input/output projections
# 3) Lower batch_size from 32 to 2
# 4) Fewer epochs from 10 to 3
# -----------------------------

from transformers import (
    GPT2Config, GPT2Model, AdamW, AutoModel, BertModel, BertConfig,
    BitsAndBytesConfig
)
from scipy.stats import wilcoxon

################################################################################
# Quantization config for DeepSeek model
################################################################################
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,   # switched to 4-bit for tighter memory
    llm_int8_threshold=6.0
)

################################################################################
# 0) Setup
################################################################################

BASE_SAVE_DIR = os.path.join(os.getcwd(), f"/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment_2")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

fish_list = [9, 10, 11, 12, 13]

seq_lengths = [5, 20]
num_epochs = 3        # reduced from 10 to lower GPU usage
batch_size = 2        # reduced from 32 to lower GPU usage
lr_default = 1e-4
num_runs = 10         # number of repeats of the pipeline

################################################################################
# 1) Helper Functions
################################################################################
def create_sequences(inputs, targets, seq_length):
    """ Create sliding-window sequences of length seq_length. """
    sequences_inputs = []
    sequences_targets = []
    for i in range(len(inputs) - seq_length + 1):
        sequences_inputs.append(inputs[i: i + seq_length])
        sequences_targets.append(targets[i: i + seq_length])
    return torch.stack(sequences_inputs), torch.stack(sequences_targets)

def average_sliding_window_predictions(predictions, seq_length, total_length):
    """ 
    Convert overlapping predictions back to frame-aligned predictions by 
    averaging the overlapping windows. 
    """
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
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

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

################################################################################
# 2) Model Definitions
################################################################################

# ----------------------
# For the *pretrained* models below, we will freeze the main transformer
# parameters to drastically reduce memory usage. Only the input_proj/output_proj
# will remain trainable. The untrained versions remain fully trainable by default.
# ----------------------

# ------------------------------------------------------------------------------ 
# GPT-2 (Pretrained)
# ------------------------------------------------------------------------------
class GPT2PretrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained("gpt2")
        hidden_size = self.transformer.config.n_embd
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

        # Freeze all GPT-2 parameters:
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, x, position_ids=None):
        # position_ids is ignored for GPT-2 in this custom version
        x = self.input_proj(x)
        outputs = self.transformer(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# ------------------------------------------------------------------------------
# GPT-2 (Untrained / Vanilla)
# ------------------------------------------------------------------------------
class GPT2UntrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Instead of from_pretrained, create an untrained GPT2Model from config
        config = GPT2Config()
        self.transformer = GPT2Model(config)  # uninitialized, random weights
        hidden_size = config.n_embd
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, position_ids=None):
        x = self.input_proj(x)
        outputs = self.transformer(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# ------------------------------------------------------------------------------
# BERT (Pretrained)
# ------------------------------------------------------------------------------
class BERTPretrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # We'll use a standard base BERT from huggingface
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

        # Freeze BERT parameters:
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x, position_ids=None):
        x = self.input_proj(x)
        outputs = self.bert(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# ------------------------------------------------------------------------------
# BERT (Untrained / Vanilla)
# ------------------------------------------------------------------------------
class BERTUntrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Build a fresh BERT config for random init
        config = BertConfig()
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, position_ids=None):
        x = self.input_proj(x)
        outputs = self.bert(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# ------------------------------------------------------------------------------
# DeepSeek (Pretrained)
# ------------------------------------------------------------------------------
class DeepSeekPretrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size_deepseek=4096):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size_deepseek)
        # Load pretrained DeepSeek model
        self.model = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config
        )
        # Output projection
        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)

        # Freeze all core model parameters:
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, position_ids=None):
        x = self.input_proj(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# ------------------------------------------------------------------------------
# DeepSeek (Untrained / Vanilla)
# ------------------------------------------------------------------------------
class DeepSeekUntrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size_deepseek=4096):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size_deepseek)
        # Create a fresh config from the pretrained config for architecture shape:
        dummy_config = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b", trust_remote_code=True
        ).config

        # Now create a new random model instance with that config
        self.model = AutoModel.from_config(dummy_config, trust_remote_code=True)

        # Output projection
        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)

    def forward(self, x, position_ids=None):
        x = self.input_proj(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

################################################################################
# 3) Main Experiment 2 Loop
################################################################################

model_variants = [
    ("GPT2 Pretrained", GPT2PretrainedModel),
    ("GPT2 Untrained", GPT2UntrainedModel),
    ("BERT Pretrained", BERTPretrainedModel),
    ("BERT Untrained", BERTUntrainedModel),
    ("DeepSeek Pretrained", DeepSeekPretrainedModel),
    ("DeepSeek Untrained", DeepSeekUntrainedModel),
]

for fish_num in fish_list:
    print("\n########################################")
    print(f"Starting Experiment 2 for Fish {fish_num}")
    print("########################################")

    fish_save_dir = os.path.join(BASE_SAVE_DIR, f"fish{fish_num}")
    os.makedirs(fish_save_dir, exist_ok=True)

    ############################################################################
    # 3.1) Load data & train/val/test split
    ############################################################################
    neural_data = np.load(
        f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish_num}_neural_data_matched.npy",
        allow_pickle=True
    )[:, :-2]
    tail_data = np.load(
        f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish_num}_tail_data_matched.npy",
        allow_pickle=True
    )
    # Transpose neural data: (num_frames, num_neurons)
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

    # Save ground truth for test in the fish folder
    np.save(os.path.join(fish_save_dir, f"fish{fish_num}_groundtruth_test.npy"), Y_test)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    input_dim = X_train_t.size(-1)
    output_dim = Y_train_t.size(-1)
    print(f"Input_dim={input_dim}, Output_dim={output_dim}")

    ############################################################################
    # 3.2) Storage structure for final RMSE results for each model and seq_length
    ############################################################################
    final_rmse = {seq_len: {m[0]: [] for m in model_variants} for seq_len in seq_lengths}

    for run in range(1, num_runs + 1):
        print(f"\n=== Fish {fish_num} - Run {run}/{num_runs} ===")
        run_folder = os.path.join(fish_save_dir, f"run_{run}")
        os.makedirs(run_folder, exist_ok=True)

        for seq_length in seq_lengths:
            print(f"\n--- Sequence Length = {seq_length} ---")
            seq_folder = os.path.join(run_folder, f"seq_{seq_length}")
            os.makedirs(seq_folder, exist_ok=True)

            # Create sliding-window sequences
            train_neural_seq, train_tail_seq = create_sequences(X_train_t, Y_train_t, seq_length)
            val_neural_seq, val_tail_seq = create_sequences(X_val_t, Y_val_t, seq_length)
            test_neural_seq, test_tail_seq = create_sequences(X_test_t, Y_test_t, seq_length)

            # Create DataLoaders
            train_dataset = TensorDataset(train_neural_seq, train_tail_seq)
            val_dataset = TensorDataset(val_neural_seq, val_tail_seq)
            test_dataset = TensorDataset(test_neural_seq, test_tail_seq)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            # Train/evaluate each of the 6 model variants
            for model_name, model_class in model_variants:
                print(f"\nTraining {model_name} ...")
                # Instantiate model
                model = model_class(input_dim, output_dim).to(device)

                # For untrained models, everything is trainable by default.
                # For pretrained models, the above classes already freeze the base.

                optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)

                train_model(
                    model, optimizer, train_loader, val_loader,
                    device, num_epochs, use_position_ids=False
                )

                # Get test predictions
                preds, _ = get_predictions(model, test_loader, device, use_position_ids=False)
                final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))

                # Save predictions
                save_pred_path = os.path.join(seq_folder, f"{model_name.replace(' ','_')}_test_preds.npy")
                np.save(save_pred_path, final_preds)

                # Compute RMSE
                rmse_val = compute_rmse(final_preds, Y_test)
                final_rmse[seq_length][model_name].append(rmse_val)
                print(f"{model_name} - RMSE: {rmse_val:.4f}")

    # After all runs for this fish, make bar plots + do Wilcoxon tests

    ############################################################################
    # 3.3) Create bar plots for each seq_length
    ############################################################################
    for seq_length in seq_lengths:
        model_names = [m[0] for m in model_variants]
        rmse_data = [final_rmse[seq_length][mn] for mn in model_names]
        rmse_means = [np.mean(r) for r in rmse_data]
        rmse_stderrs = [np.std(r) / np.sqrt(len(r)) for r in rmse_data]

        x = np.arange(len(model_names))
        width = 0.6
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, rmse_means, yerr=rmse_stderrs, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20)
        ax.set_ylabel("RMSE")
        ax.set_title(f"Fish {fish_num} - Sequence Length {seq_length}\n(Mean ± SEM over {num_runs} runs)")
        plt.tight_layout()

        plot_path = os.path.join(fish_save_dir, f"fish{fish_num}_seq{seq_length}_bar_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved bar plot to {plot_path}")

    ############################################################################
    # 3.4) Wilcoxon test: pretrained vs. untrained for GPT2, BERT, DeepSeek
    ############################################################################
    significance_results = []
    for seq_length in seq_lengths:
        significance_results.append(f"--- Sequence Length = {seq_length} ---")

        # GPT-2
        gp = np.array(final_rmse[seq_length]["GPT2 Pretrained"])
        gu = np.array(final_rmse[seq_length]["GPT2 Untrained"])
        stat, pval = wilcoxon(gp, gu)
        significance_results.append(f"GPT2 (Pretrained vs. Untrained): p-value = {pval:.4e}")

        # BERT
        bp = np.array(final_rmse[seq_length]["BERT Pretrained"])
        bu = np.array(final_rmse[seq_length]["BERT Untrained"])
        stat, pval = wilcoxon(bp, bu)
        significance_results.append(f"BERT (Pretrained vs. Untrained): p-value = {pval:.4e}")

        # DeepSeek
        dp = np.array(final_rmse[seq_length]["DeepSeek Pretrained"])
        du = np.array(final_rmse[seq_length]["DeepSeek Untrained"])
        stat, pval = wilcoxon(dp, du)
        significance_results.append(f"DeepSeek (Pretrained vs. Untrained): p-value = {pval:.4e}")

        significance_results.append("")  # blank line

    sig_file_path = os.path.join(fish_save_dir, f"fish{fish_num}_wilcoxon_results.txt")
    with open(sig_file_path, "w") as f:
        for line in significance_results:
            f.write(line + "\n")
    print(f"Wilcoxon test results saved to {sig_file_path}")

    print(f"==== Finished Experiment 2 for Fish {fish_num} ====\n")

print("Experiment 2 completed for all specified fish.")
