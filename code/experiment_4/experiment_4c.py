#!/usr/bin/env python
"""
experiment_4_comparison.py

This script compares pretrained and nonpretrained variants of neural models on a new
behavior prediction task: bout duration prediction. For each bout detected from the tail data,
a fixed window of neural activity preceding the bout onset is used as input and the bout duration (in seconds)
is the target.

Models compared:
  - GPT-2 (Pretrained and Nonpretrained variants)
  - BERT (Pretrained and Nonpretrained variants)
  - DeepSeek-V3 MoE (Pretrained and Nonpretrained variants)
  - LSTM (trained from scratch)
  - Reservoir Computer (trained from scratch)

Results (predictions and RMSE) are saved for each fish, and grouped bar plots and significance tests are generated.

Author: Your Name
Date: 2024-XX-XX
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model, BertModel, BertConfig, AutoModel, BitsAndBytesConfig, AdamW
from scipy.stats import mannwhitneyu

# =============================================================================
# 0) Setup
# =============================================================================

BASE_SAVE_DIR = os.path.join(os.getcwd(), "results", "experiment_4_comparison")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# List of fish to process
fish_list = [9, 10, 11, 12, 13]

# Experiment parameters
seq_lengths = [5, 10, 15, 20]  # neural window lengths (frames)
num_epochs = 10
batch_size = 32
lr_default = 1e-4
num_runs = 5  # number of repeated runs

# Imaging frequency for tail recordings (Hz)
tail_hz = 30  # adjust as needed

# Quantization config used for DeepSeek model
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# =============================================================================
# 1) Behavioral Analysis Function (Bout Extraction)
# =============================================================================

def analyze_tail(frametimes_df, stimulus_df, tail_df, tail_hz, stimulus_s=5, strength_boundary=0.25, 
                 min_on_s=0.1, cont_cutoff_s=0.05):
    """
    Detects tail bouts from the tail data.
    
    Returns a DataFrame with columns:
      - 'bout_start_frame'
      - 'bout_end_frame'
      - 'bout_duration' (in seconds)
    """
    hz = tail_hz
    # Create a figure (can be saved later) to show raw tail trace and detected bouts
    fig, ax = plt.subplots(3, 1, dpi=150, figsize=(12, 8))
    
    tail_df = tail_df.fillna(method='ffill')
    
    # Plot raw tail signal
    ax[0].plot(tail_df['frame'], tail_df['tail_sum'], linewidth=0.5, color='black')
    ax[0].set_xlim([0, tail_df['frame'].max()])
    ax[0].set_xlabel("Frame")
    ax[0].set_ylabel("Raw Tail Angle (rad)")
    ax[0].set_title("Raw Tail Trace")
    
    # Calibrate by subtracting baseline
    baseline = tail_df['tail_sum'].mean()
    tail_df['tail_sum'] = tail_df['tail_sum'] - baseline
    tail_df['tail_sum'] = tail_df['tail_sum'].fillna(method='ffill')
    
    # Plot positive and negative components
    pos = np.where(tail_df['tail_sum'] > 0, tail_df['tail_sum'], 0)
    neg = np.where(tail_df['tail_sum'] < 0, tail_df['tail_sum'], 0)
    ax[1].plot(tail_df['frame'], pos, linewidth=0.5, color='maroon', label='Positive')
    ax[1].plot(tail_df['frame'], neg, linewidth=0.5, color='midnightblue', label='Negative')
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Tail Angle (rad)")
    ax[1].set_title("Positive/Negative Tail Movements")
    ax[1].legend()
    
    # Compute local standard deviation as a measure of tail "strength"
    smooth_tailframe = int(0.05 * hz)
    std_vals = [np.std(tail_df['tail_sum'].iloc[i - smooth_tailframe:i + smooth_tailframe])
                for i in range(smooth_tailframe, len(tail_df) - smooth_tailframe)]
    tail_df['std'] = [0] * smooth_tailframe + std_vals + [0] * smooth_tailframe
    ax[2].plot(tail_df['frame'], tail_df['std'], linewidth=0.5, color='black')
    
    # Detect bouts: where local std exceeds threshold
    bout_on = (tail_df['std'] > strength_boundary).astype(int)
    on_index = np.where(np.diff(bout_on) == 1)[0] + smooth_tailframe
    off_index = np.where(np.diff(bout_on) == -1)[0] + smooth_tailframe

    if len(on_index) != 0 and len(off_index) != 0:
        if on_index[0] > off_index[0]:
            on_index = np.concatenate(([0], on_index))
        if on_index[-1] > off_index[-1]:
            off_index = np.concatenate((off_index, [len(tail_df) - 1]))
        bout_tuples = [(on, off) for on, off in zip(on_index, off_index) if off > on]
    else:
        bout_tuples = []
    
    # Optionally group bouts that are close in time
    cont_on_index = []
    cont_off_index = []
    if bout_tuples:
        cont_on_index = [bout_tuples[0][0]]
        if len(bout_tuples) > 1:
            intervals = np.array([bout_tuples[i][0] - bout_tuples[i - 1][1] for i in range(1, len(bout_tuples))])
            big_interval = intervals > (cont_cutoff_s * hz)
            for i, flag in enumerate(big_interval):
                if flag:
                    cont_on_index.append(bout_tuples[i+1][0])
                    cont_off_index.append(bout_tuples[i][1])
            cont_off_index.append(bout_tuples[-1][1])
    cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index) 
                   if (off - on) > (min_on_s * hz)]
    
    # Shade detected bouts
    for on, off in cont_tuples:
        ax[2].axvspan(tail_df['frame'].iloc[on], tail_df['frame'].iloc[off], color='pink', alpha=0.5)
    
    ax[2].axhline(strength_boundary, linestyle=':', color='purple', linewidth=1)
    ax[2].set_xlabel("Frame")
    ax[2].set_ylabel("Tail Strength (std)")
    ax[2].set_title("Tail Strength and Detected Bouts")
    
    plt.tight_layout()
    
    # Build bout dataframe (bout duration in seconds)
    bout_df = pd.DataFrame({
        'bout_start_frame': [tail_df['frame'].iloc[on] for on, off in cont_tuples],
        'bout_end_frame': [tail_df['frame'].iloc[off] for on, off in cont_tuples]
    })
    if not bout_df.empty:
        bout_df['bout_duration'] = (bout_df['bout_end_frame'] - bout_df['bout_start_frame']) / hz
    else:
        bout_df['bout_duration'] = []
    
    return bout_df

# =============================================================================
# 2) Dataset Creation for Bout Duration Prediction
# =============================================================================

def create_bout_dataset(neural_data, bout_df, seq_length):
    """
    For each bout (with onset after at least seq_length frames), use the neural data window 
    preceding the bout onset as input and the bout duration as target.
    
    Returns:
      X: shape (num_examples, seq_length, num_neurons)
      Y: shape (num_examples, 1)
    """
    inputs = []
    targets = []
    for idx, row in bout_df.iterrows():
        bout_onset = int(row['bout_start_frame'])
        if bout_onset >= seq_length:
            x_seq = neural_data[bout_onset - seq_length : bout_onset]
            inputs.append(x_seq)
            targets.append(row['bout_duration'])
    if inputs:
        return np.stack(inputs), np.array(targets).reshape(-1, 1)
    else:
        return None, None

# =============================================================================
# 3) Basic Training, Prediction, and RMSE Functions
# =============================================================================

def train_model(model, optimizer, train_loader, val_loader, device, num_epochs):
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
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    return model

def get_predictions(model, data_loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for inputs, targs in data_loader:
            inputs, targs = inputs.to(device), targs.to(device)
            outputs = model(inputs)
            preds.append(outputs.cpu())
            targets.append(targs.cpu())
    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    return preds, targets

def compute_rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))

# =============================================================================
# 4) Model Architectures with Pretrained vs Nonpretrained Options
# =============================================================================

# --- GPT-2 based model ---
class CustomGPT2Model(nn.Module):
    def __init__(self, input_dim, output_dim=1, pretrained=True):
        super(CustomGPT2Model, self).__init__()
        if pretrained:
            # Load pretrained GPT-2 transformer
            self.transformer = GPT2Model.from_pretrained("gpt2")
        else:
            # Initialize a GPT2 model with random weights
            from transformers import GPT2Config
            config = GPT2Config()
            self.transformer = GPT2Model(config)
        hidden_size = self.transformer.config.n_embd
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_length, input_dim)
        x = self.input_proj(x)
        outputs = self.transformer(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        last_hidden = hidden_states[:, -1, :]
        logits = self.output_proj(last_hidden)
        return logits

# --- BERT based model ---
class CustomBERTModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim=1, pretrained=True):
        super(CustomBERTModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        if pretrained:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            config = BertConfig(hidden_size=hidden_size, num_hidden_layers=6, num_attention_heads=12,
                                intermediate_size=hidden_size*4)
            self.bert = BertModel(config)
        self.output_proj = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.bert(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        last_hidden = hidden_states[:, -1, :]
        logits = self.output_proj(last_hidden)
        return logits

# --- DeepSeek-V3 MoE based model ---
class DeepSeekV3MoE(nn.Module):
    def __init__(self, input_dim, output_dim=1, pretrained=True):
        super(DeepSeekV3MoE, self).__init__()
        self.input_proj = nn.Linear(input_dim, 4096)
        if pretrained:
            self.model = AutoModel.from_pretrained("deepseek-ai/deepseek-coder-7b",
                                                     trust_remote_code=True,
                                                     device_map="auto",
                                                     quantization_config=quant_config)
        else:
            # For nonpretrained, use from_config with default settings
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("deepseek-ai/deepseek-coder-7b")
            self.model = AutoModel.from_config(config)
        self.output_proj = nn.Linear(4096, output_dim)
        self.router = nn.Linear(4096, 8)  # num_experts = 8
        self.softmax = nn.Softmax(dim=-1)
        self.top_k = 2
    
    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # shape: (batch, seq_length, 4096)
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, self.top_k, dim=-1)[1]
        aggregated_output = torch.zeros_like(hidden_states)
        for i in range(self.top_k):
            expert_idx = top_k_experts[:, :, i].unsqueeze(-1).expand_as(hidden_states)
            expert_output = torch.gather(hidden_states, dim=-1, index=expert_idx)
            aggregated_output += expert_output / self.top_k
        last_hidden = aggregated_output[:, -1, :]
        logits = self.output_proj(last_hidden)
        return logits

# --- LSTM Model (always trained from scratch) ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

# --- Reservoir Computer (always from scratch) ---
class ReservoirComputer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, alpha=0.9):
        super(ReservoirComputer, self).__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        W_in = torch.rand(input_dim, hidden_dim) - 0.5
        W = torch.rand(hidden_dim, hidden_dim) - 0.5
        self.W_in = nn.Parameter(W_in, requires_grad=False)
        self.W = nn.Parameter(W, requires_grad=False)
        self.readout = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        for t in range(seq_len):
            x_t = x[:, t, :]
            hidden = (1 - self.alpha) * hidden + self.alpha * torch.tanh(x_t @ self.W_in + hidden @ self.W)
        return self.readout(hidden)

# =============================================================================
# 5) Main Experiment Loop: Fine-tuning on Bout Duration Prediction with Both Pretrained and Nonpretrained Variants
# =============================================================================

# For model variants, we will label them as:
# "GPT2 Pretrained", "GPT2 Nonpretrained", "BERT Pretrained", "BERT Nonpretrained", "DeepSeek Pretrained", "DeepSeek Nonpretrained",
# plus "LSTM" and "Reservoir" (only one variant each).

model_names_list = ["GPT2 Pretrained", "GPT2 Nonpretrained", "BERT Pretrained", "BERT Nonpretrained",
                    "DeepSeek Pretrained", "DeepSeek Nonpretrained", "LSTM", "Reservoir"]

# Dictionary to store RMSE values across runs for each sequence length
final_all_rmse = {seq: {model: [] for model in model_names_list} for seq in seq_lengths}

for fish_num in fish_list:
    print("\n====================================")
    print(f"Processing Fish {fish_num} for bout duration prediction")
    print("====================================")
    
    # Create folder for this fish
    fish_save_dir = os.path.join(BASE_SAVE_DIR, f"fish{fish_num}")
    os.makedirs(fish_save_dir, exist_ok=True)
    
    # -------------------------------
    # Load Neural and Tail Data
    # -------------------------------
    neural_data_path = f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish_num}_neural_data_matched.npy"
    tail_data_path = f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish_num}_tail_data_matched.npy"
    
    neural_data = np.load(neural_data_path, allow_pickle=True)[:, :-2]
    tail_data = np.load(tail_data_path, allow_pickle=True)
    
    # Transpose neural data: shape (num_frames, num_neurons)
    neural_data = neural_data.T
    print("Neural data shape:", neural_data.shape)
    print("Tail data shape:", tail_data.shape)
    
    assert neural_data.shape[0] == tail_data.shape[0], "Mismatch in neural and tail data lengths"
    
    # -------------------------------
    # Create Tail DataFrame and Extract Bout Events
    # -------------------------------
    if tail_data.ndim > 1:
        tail_signal = tail_data[:, 0]
    else:
        tail_signal = tail_data
    tail_df = pd.DataFrame({
        'frame': np.arange(len(tail_signal)),
        'tail_sum': tail_signal
    })
    frametimes_df = pd.DataFrame({
        'frame': np.arange(len(tail_signal)),
        'time': np.arange(len(tail_signal)) / tail_hz
    })
    stimulus_df = pd.DataFrame(columns=['frame', 'stim_name'])
    
    bout_df = analyze_tail(frametimes_df, stimulus_df, tail_df, tail_hz, stimulus_s=5)
    if bout_df.empty:
        print(f"No bouts detected for Fish {fish_num}; skipping.")
        continue
    bout_df.to_csv(os.path.join(fish_save_dir, f"fish{fish_num}_bout_summary.csv"), index=False)
    plt.savefig(os.path.join(fish_save_dir, f"fish{fish_num}_tail_bout_analysis.png"))
    plt.close()
    
    # -------------------------------
    # Loop over sequence lengths and runs
    # -------------------------------
    for seq_length in seq_lengths:
        print(f"\nFish {fish_num}: Using neural window length = {seq_length} frames")
        X, Y = create_bout_dataset(neural_data, bout_df, seq_length)
        if X is None:
            print(f"Not enough bouts with sufficient neural history for seq_length = {seq_length}; skipping.")
            continue
        
        num_examples = X.shape[0]
        train_end = int(0.7 * num_examples)
        val_end = int(0.8 * num_examples)
        X_train, Y_train = X[:train_end], Y[:train_end]
        X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
        X_test, Y_test = X[val_end:], Y[val_end:]
        
        # Save ground truth for later evaluation
        np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_groundtruth_val_seq{seq_length}.npy"), Y_val)
        np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_groundtruth_test_seq{seq_length}.npy"), Y_test)
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        Y_test_t = torch.tensor(Y_test, dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=batch_size)
        
        for run in range(1, num_runs + 1):
            print(f"  Run {run} of {num_runs}")
            rmse_results = {}
            
            # --- GPT-2 variants ---
            print("    Training GPT-2 Pretrained variant...")
            model = CustomGPT2Model(input_dim=neural_data.shape[1], output_dim=1, pretrained=True).to(device)
            optimizer = AdamW(model.parameters(), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            preds, _ = get_predictions(model, test_loader, device)
            rmse = compute_rmse(preds, Y_test)
            rmse_results["GPT2 Pretrained"] = rmse
            np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_preds_gpt2_pretrained_seq{seq_length}_run{run}.npy"), preds)
            print(f"      RMSE: {rmse:.4f}")
            
            print("    Training GPT-2 Nonpretrained variant...")
            model = CustomGPT2Model(input_dim=neural_data.shape[1], output_dim=1, pretrained=False).to(device)
            optimizer = AdamW(model.parameters(), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            preds, _ = get_predictions(model, test_loader, device)
            rmse = compute_rmse(preds, Y_test)
            rmse_results["GPT2 Nonpretrained"] = rmse
            np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_preds_gpt2_nonpretrained_seq{seq_length}_run{run}.npy"), preds)
            print(f"      RMSE: {rmse:.4f}")
            
            # --- BERT variants ---
            print("    Training BERT Pretrained variant...")
            bert_hidden_size = 768
            model = CustomBERTModel(input_dim=neural_data.shape[1], hidden_size=bert_hidden_size, output_dim=1, pretrained=True).to(device)
            optimizer = AdamW(model.parameters(), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            preds, _ = get_predictions(model, test_loader, device)
            rmse = compute_rmse(preds, Y_test)
            rmse_results["BERT Pretrained"] = rmse
            np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_preds_bert_pretrained_seq{seq_length}_run{run}.npy"), preds)
            print(f"      RMSE: {rmse:.4f}")
            
            print("    Training BERT Nonpretrained variant...")
            model = CustomBERTModel(input_dim=neural_data.shape[1], hidden_size=bert_hidden_size, output_dim=1, pretrained=False).to(device)
            optimizer = AdamW(model.parameters(), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            preds, _ = get_predictions(model, test_loader, device)
            rmse = compute_rmse(preds, Y_test)
            rmse_results["BERT Nonpretrained"] = rmse
            np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_preds_bert_nonpretrained_seq{seq_length}_run{run}.npy"), preds)
            print(f"      RMSE: {rmse:.4f}")
            
            # --- DeepSeek variants ---
            print("    Training DeepSeek Pretrained variant...")
            model = DeepSeekV3MoE(input_dim=neural_data.shape[1], output_dim=1, pretrained=True).to(device)
            # Freeze backbone parameters as before
            for param in model.model.parameters():
                param.requires_grad = False
            for param in model.router.parameters():
                param.requires_grad = True
            for param in model.output_proj.parameters():
                param.requires_grad = True
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            preds, _ = get_predictions(model, test_loader, device)
            rmse = compute_rmse(preds, Y_test)
            rmse_results["DeepSeek Pretrained"] = rmse
            np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_preds_deepseek_pretrained_seq{seq_length}_run{run}.npy"), preds)
            print(f"      RMSE: {rmse:.4f}")
            
            print("    Training DeepSeek Nonpretrained variant...")
            model = DeepSeekV3MoE(input_dim=neural_data.shape[1], output_dim=1, pretrained=False).to(device)
            optimizer = AdamW(model.parameters(), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            preds, _ = get_predictions(model, test_loader, device)
            rmse = compute_rmse(preds, Y_test)
            rmse_results["DeepSeek Nonpretrained"] = rmse
            np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_preds_deepseek_nonpretrained_seq{seq_length}_run{run}.npy"), preds)
            print(f"      RMSE: {rmse:.4f}")
            
            # --- LSTM (only one variant) ---
            print("    Training LSTM...")
            lstm_hidden_dim = 256
            model = LSTMModel(input_dim=neural_data.shape[1], hidden_dim=lstm_hidden_dim, output_dim=1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            preds, _ = get_predictions(model, test_loader, device)
            rmse = compute_rmse(preds, Y_test)
            rmse_results["LSTM"] = rmse
            np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_preds_lstm_seq{seq_length}_run{run}.npy"), preds)
            print(f"      RMSE: {rmse:.4f}")
            
            # --- Reservoir (only one variant) ---
            print("    Training Reservoir Computer...")
            res_hidden_dim = 256
            model = ReservoirComputer(input_dim=neural_data.shape[1], hidden_dim=res_hidden_dim, output_dim=1).to(device)
            optimizer = optim.Adam(model.readout.parameters(), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            preds, _ = get_predictions(model, test_loader, device)
            rmse = compute_rmse(preds, Y_test)
            rmse_results["Reservoir"] = rmse
            np.save(os.path.join(fish_save_dir, f"fish{fish_num}_bout_preds_reservoir_seq{seq_length}_run{run}.npy"), preds)
            print(f"      RMSE: {rmse:.4f}")
            
            # Record RMSE for each model variant for this run
            for model_name in model_names_list:
                final_all_rmse[seq_length][model_name].append(rmse_results[model_name])
    
    # -------------------------------
    # Grouped Bar Plot for Each Fish: RMSE Across Sequence Lengths
    # -------------------------------
    mean_rmse_by_model = {model: [] for model in model_names_list}
    stderr_rmse_by_model = {model: [] for model in model_names_list}
    for seq in seq_lengths:
        for model in model_names_list:
            rmse_vals = np.array(final_all_rmse[seq][model])
            mean_rmse_by_model[model].append(np.mean(rmse_vals))
            stderr_rmse_by_model[model].append(np.std(rmse_vals) / np.sqrt(len(rmse_vals)))
    
    x = np.arange(len(seq_lengths))
    num_models = len(model_names_list)
    width = 0.8 / num_models
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(model_names_list):
        offset = (i - num_models/2) * width + width/2
        ax.bar(x + offset, mean_rmse_by_model[model], width=width, yerr=stderr_rmse_by_model[model],
               capsize=5, label=model)
    ax.set_xlabel("Sequence Length (frames)")
    ax.set_ylabel("Bout Duration Prediction RMSE (s)")
    ax.set_title(f"Bout Duration Prediction RMSE Across Sequence Lengths for Fish {fish_num}\n(Mean ± SEM over runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(fish_save_dir, f"fish{fish_num}_grouped_rmse_bout_duration.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Grouped RMSE plot saved to {plot_path}")
    
    # -------------------------------
    # Significance Testing (Mann–Whitney U Test)
    # -------------------------------
    significance_results = ""
    for seq in seq_lengths:
        significance_results += f"Sequence Length {seq} frames:\n"
        for i in range(len(model_names_list)):
            for j in range(i+1, len(model_names_list)):
                model_a = model_names_list[i]
                model_b = model_names_list[j]
                vals_a = np.array(final_all_rmse[seq][model_a])
                vals_b = np.array(final_all_rmse[seq][model_b])
                stat, p_val = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
                significance_results += f"  {model_a} vs {model_b}: p-value = {p_val:.4e}\n"
        significance_results += "\n"
    sig_file_path = os.path.join(fish_save_dir, f"fish{fish_num}_significance_results_bout_duration.txt")
    with open(sig_file_path, "w") as f:
        f.write(significance_results)
    print(f"Significance test results saved to {sig_file_path}")
    
    print(f"Fish {fish_num}: Bout duration prediction completed.\n")

print("Experiment 4 Comparison (Pretrained vs Nonpretrained on Bout Duration) completed for all fish.")
