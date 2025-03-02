#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2Model, AdamW, AutoModel, BertModel, BertConfig, AutoConfig
from scipy.stats import wilcoxon

# =============================================================================
# 0) Setup & Data Loading
# =============================================================================
RESULTS_DIR = f"/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment2_results"  # separate folder for experiment 2
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use the same data as in experiment 1 (adjust paths as needed)
fish_num = 9
data_dir = f"/hpc/group/naumannlab/jjm132/data_prepped_for_models"
neural_data = np.load(os.path.join(data_dir, f"fish{fish_num}_neural_data_matched.npy"), allow_pickle=True)[:,:-2]
tail_data   = np.load(os.path.join(data_dir, f"fish{fish_num}_tail_data_matched.npy"), allow_pickle=True)

# Transpose neural data so shape is (num_frames, num_neurons)
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

# Save ground truth for later evaluation
np.save(os.path.join(RESULTS_DIR, "final_predictions_groundtruth_val.npy"), Y_val)
np.save(os.path.join(RESULTS_DIR, "final_predictions_groundtruth_test.npy"), Y_test)
print("Ground truth for val/test saved.")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
Y_val_t   = torch.tensor(Y_val, dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test, dtype=torch.float32)

# Get dimensions
input_dim = X_train_t.size(-1)   # number of neurons
output_dim = Y_train_t.size(-1)   # number of tail features
print(f"Input_dim={input_dim}, Output_dim={output_dim}")

# =============================================================================
# 1) Helper Functions
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
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

def get_predictions(model, data_loader, device, use_position_ids=False):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for inputs, _ in data_loader:
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
# 2) Model Definitions
# =============================================================================

# ---- DeepSeek Models (MoE) ----
hidden_size_deepseek = 4096
num_experts = 8
top_k = 2

class DeepSeekV3MoE(nn.Module):
    # Pretrained DeepSeek (using pretrained weights)
    def __init__(self, input_dim, output_dim):
        super(DeepSeekV3MoE, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size_deepseek)
        self.model = AutoModel.from_pretrained("deepseek-ai/deepseek-coder-7b", trust_remote_code=True)
        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)
        self.router = nn.Linear(hidden_size_deepseek, num_experts)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, top_k, dim=-1)[1]
        aggregated_output = torch.zeros_like(hidden_states)
        for i in range(top_k):
            expert_idx = top_k_experts[:, :, i].unsqueeze(-1).expand_as(hidden_states)
            expert_output = torch.gather(hidden_states, dim=-1, index=expert_idx)
            aggregated_output += expert_output / top_k
        logits = self.output_proj(aggregated_output)
        return logits

class DeepSeekV3MoEVanilla(nn.Module):
    # Vanilla DeepSeek: instantiated from config (untrained)
    def __init__(self, input_dim, output_dim):
        super(DeepSeekV3MoEVanilla, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size_deepseek)
        config = AutoConfig.from_pretrained("deepseek-ai/deepseek-coder-7b")
        self.model = AutoModel.from_config(config)
        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)
        self.router = nn.Linear(hidden_size_deepseek, num_experts)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, top_k, dim=-1)[1]
        aggregated_output = torch.zeros_like(hidden_states)
        for i in range(top_k):
            expert_idx = top_k_experts[:, :, i].unsqueeze(-1).expand_as(hidden_states)
            expert_output = torch.gather(hidden_states, dim=-1, index=expert_idx)
            aggregated_output += expert_output / top_k
        logits = self.output_proj(aggregated_output)
        return logits

# ---- GPT-2 Models ----
class CustomGPT2ModelPretrained(nn.Module):
    # GPT-2 with pretrained weights
    def __init__(self, input_dim, output_dim):
        super(CustomGPT2ModelPretrained, self).__init__()
        self.transformer = GPT2Model.from_pretrained("gpt2")
        hidden_size = self.transformer.config.n_embd
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)
    def forward(self, x, position_ids=None):
        x = self.input_proj(x)
        outputs = self.transformer(inputs_embeds=x, position_ids=position_ids)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

class CustomGPT2ModelScratch(nn.Module):
    # GPT-2 from scratch (vanilla)
    def __init__(self, input_dim, hidden_size, output_dim, n_head, n_layer, n_positions):
        super(CustomGPT2ModelScratch, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        config = GPT2Config(
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=n_positions,
            vocab_size=1,
        )
        self.transformer = GPT2Model(config)
        self.output_proj = nn.Linear(hidden_size, output_dim)
    def forward(self, x, position_ids=None):
        x = self.input_proj(x)
        outputs = self.transformer(inputs_embeds=x, position_ids=position_ids)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# ---- BERT Models ----
class CustomBERTModelPretrained(nn.Module):
    # BERT with pretrained weights
    def __init__(self, input_dim, output_dim):
        super(CustomBERTModelPretrained, self).__init__()
        hidden_size = 768
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.output_proj = nn.Linear(hidden_size, output_dim)
    def forward(self, x, position_ids=None):
        x = self.input_proj(x)
        outputs = self.bert(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

class CustomBERTModelVanilla(nn.Module):
    # BERT from scratch (vanilla)
    def __init__(self, input_dim, output_dim):
        super(CustomBERTModelVanilla, self).__init__()
        hidden_size = 768
        self.input_proj = nn.Linear(input_dim, hidden_size)
        config = BertConfig(hidden_size=hidden_size, num_hidden_layers=6, num_attention_heads=12,
                            intermediate_size=hidden_size * 4)
        self.bert = BertModel(config)
        self.output_proj = nn.Linear(hidden_size, output_dim)
    def forward(self, x, position_ids=None):
        x = self.input_proj(x)
        outputs = self.bert(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

# =============================================================================
# 3) Main Loop: 10 Runs, Single Sequence Length of 20
# =============================================================================
model_names = ["DeepSeek Pretrained", "DeepSeek Vanilla", 
               "GPT2 Pretrained", "GPT2 Vanilla",
               "BERT Pretrained", "BERT Vanilla"]

seq_lengths = [20]  # Only sequence length 20 is used
num_epochs = 5
batch_size = 32
lr_default = 1e-4
num_runs = 10

# Structure to store RMSE for each model across runs
final_all_rmse = {seq: {model: [] for model in model_names} for seq in seq_lengths}

# For GPT2 vanilla hyperparameters (example values)
gpt2_hidden_size = 768
gpt2_n_head = 12
gpt2_n_layer = 6
gpt2_n_positions = 1024

for run in range(1, num_runs + 1):
    print("\n==============================")
    print(f"Run {run} of {num_runs}")
    print("==============================")
    
    run_folder = os.path.join(RESULTS_DIR, f"run_{run}")
    os.makedirs(run_folder, exist_ok=True)
    
    for seq_length in seq_lengths:
        print("\n------------------------------")
        print(f"Run {run}, Sequence Length = {seq_length}")
        print("------------------------------")
        
        seq_folder = os.path.join(run_folder, f"seq_{seq_length}")
        os.makedirs(seq_folder, exist_ok=True)
        
        train_seq_x, train_seq_y = create_sequences(X_train_t, Y_train_t, seq_length)
        val_seq_x, val_seq_y     = create_sequences(X_val_t, Y_val_t, seq_length)
        test_seq_x, test_seq_y   = create_sequences(X_test_t, Y_test_t, seq_length)
        
        train_loader = DataLoader(TensorDataset(train_seq_x, train_seq_y), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(val_seq_x, val_seq_y), batch_size=batch_size)
        test_loader  = DataLoader(TensorDataset(test_seq_x, test_seq_y), batch_size=batch_size)
        
        rmse_results = {}
        
        # --- DeepSeek Pretrained ---
        print("\nTraining DeepSeek Pretrained...")
        model = DeepSeekV3MoE(input_dim, output_dim).to(device)
        for param in model.model.parameters():
            param.requires_grad = False
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
        train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
        torch.save(model.state_dict(), os.path.join(seq_folder, f"deepseek_moe_pretrained_weights_run{run}.pth"))
        preds = get_predictions(model, test_loader, device)
        final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
        np.save(os.path.join(seq_folder, f"final_predictions_deepseek_moe_pretrained_test_run{run}.npy"), final_preds)
        rmse = compute_rmse(final_preds, Y_test)
        rmse_results["DeepSeek Pretrained"] = rmse
        print(f"DeepSeek Pretrained RMSE: {rmse:.4f}")
        
        # --- DeepSeek Vanilla ---
        print("\nTraining DeepSeek Vanilla...")
        model = DeepSeekV3MoEVanilla(input_dim, output_dim).to(device)
        for param in model.model.parameters():
            param.requires_grad = False
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
        train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
        torch.save(model.state_dict(), os.path.join(seq_folder, f"deepseek_moe_vanilla_weights_run{run}.pth"))
        preds = get_predictions(model, test_loader, device)
        final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
        np.save(os.path.join(seq_folder, f"final_predictions_deepseek_moe_vanilla_test_run{run}.npy"), final_preds)
        rmse = compute_rmse(final_preds, Y_test)
        rmse_results["DeepSeek Vanilla"] = rmse
        print(f"DeepSeek Vanilla RMSE: {rmse:.4f}")
        
        # --- GPT2 Pretrained ---
        print("\nTraining GPT2 Pretrained...")
        model = CustomGPT2ModelPretrained(input_dim, output_dim).to(device)
        optimizer = AdamW(model.parameters(), lr=lr_default)
        train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
        torch.save(model.state_dict(), os.path.join(seq_folder, f"gpt2_pretrained_weights_run{run}.pth"))
        preds = get_predictions(model, test_loader, device)
        final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
        np.save(os.path.join(seq_folder, f"final_predictions_gpt2_pretrained_test_run{run}.npy"), final_preds)
        rmse = compute_rmse(final_preds, Y_test)
        rmse_results["GPT2 Pretrained"] = rmse
        print(f"GPT2 Pretrained RMSE: {rmse:.4f}")
        
        # --- GPT2 Vanilla ---
        print("\nTraining GPT2 Vanilla...")
        model = CustomGPT2ModelScratch(input_dim, gpt2_hidden_size, output_dim, gpt2_n_head, gpt2_n_layer, gpt2_n_positions).to(device)
        optimizer = AdamW(model.parameters(), lr=lr_default)
        train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
        torch.save(model.state_dict(), os.path.join(seq_folder, f"gpt2_vanilla_weights_run{run}.pth"))
        preds = get_predictions(model, test_loader, device)
        final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
        np.save(os.path.join(seq_folder, f"final_predictions_gpt2_vanilla_test_run{run}.npy"), final_preds)
        rmse = compute_rmse(final_preds, Y_test)
        rmse_results["GPT2 Vanilla"] = rmse
        print(f"GPT2 Vanilla RMSE: {rmse:.4f}")
        
        # --- BERT Pretrained ---
        print("\nTraining BERT Pretrained...")
        model = CustomBERTModelPretrained(input_dim, output_dim).to(device)
        optimizer = AdamW(model.parameters(), lr=lr_default)
        train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
        torch.save(model.state_dict(), os.path.join(seq_folder, f"bert_pretrained_weights_run{run}.pth"))
        preds = get_predictions(model, test_loader, device)
        final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
        np.save(os.path.join(seq_folder, f"final_predictions_bert_pretrained_test_run{run}.npy"), final_preds)
        rmse = compute_rmse(final_preds, Y_test)
        rmse_results["BERT Pretrained"] = rmse
        print(f"BERT Pretrained RMSE: {rmse:.4f}")
        
        # --- BERT Vanilla ---
        print("\nTraining BERT Vanilla...")
        model = CustomBERTModelVanilla(input_dim, output_dim).to(device)
        optimizer = AdamW(model.parameters(), lr=lr_default)
        train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
        torch.save(model.state_dict(), os.path.join(seq_folder, f"bert_vanilla_weights_run{run}.pth"))
        preds = get_predictions(model, test_loader, device)
        final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
        np.save(os.path.join(seq_folder, f"final_predictions_bert_vanilla_test_run{run}.npy"), final_preds)
        rmse = compute_rmse(final_preds, Y_test)
        rmse_results["BERT Vanilla"] = rmse
        print(f"BERT Vanilla RMSE: {rmse:.4f}")
        
        for model_name in model_names:
            final_all_rmse[seq_length][model_name].append(rmse_results[model_name])

# =============================================================================
# 4) Final Grouped Bar Plot (Overall RMSE)
# =============================================================================
mean_rmse_by_model = {model: [] for model in model_names}
stderr_rmse_by_model = {model: [] for model in model_names}
for seq in seq_lengths:
    for model in model_names:
        vals = np.array(final_all_rmse[seq][model])
        mean_rmse_by_model[model].append(np.mean(vals))
        stderr_rmse_by_model[model].append(np.std(vals) / np.sqrt(len(vals)))

x = np.arange(len(seq_lengths))
num_models = len(model_names)
width = 0.8 / num_models

fig, ax = plt.subplots(figsize=(12, 6))
for i, model in enumerate(model_names):
    offset = (i - num_models/2)*width + width/2
    ax.bar(x + offset, mean_rmse_by_model[model], width=width, yerr=stderr_rmse_by_model[model],
           capsize=5, label=model)
ax.set_xlabel("Sequence Length")
ax.set_ylabel("Overall RMSE")
ax.set_title("Model Comparison: Pretrained vs Vanilla (Mean ± SEM over runs)")
ax.set_xticks(x)
ax.set_xticklabels(seq_lengths)
ax.legend()
plt.tight_layout()
final_plot_path = os.path.join(RESULTS_DIR, "grouped_rmse_comparison_experiment2.png")
plt.savefig(final_plot_path)
print(f"Final grouped bar plot saved to {final_plot_path}")

# =============================================================================
# 5) Significance Testing using Wilcoxon Signed-Rank Test
# =============================================================================
# For each model family compare the pretrained vs vanilla version (paired test).
significance_results = ""
for seq in seq_lengths:
    significance_results += f"Sequence Length {seq}:\n"
    for family in ["DeepSeek", "GPT2", "BERT"]:
        key_pre = f"{family} Pretrained"
        key_van = f"{family} Vanilla"
        vals_pre = np.array(final_all_rmse[seq][key_pre])
        vals_van = np.array(final_all_rmse[seq][key_van])
        stat, p_val = wilcoxon(vals_pre, vals_van)
        significance_results += f"  {family} Pretrained vs {family} Vanilla: p-value = {p_val:.4e}\n"
    significance_results += "\n"

sig_file_path = os.path.join(RESULTS_DIR, "significance_results_wilcoxon.txt")
with open(sig_file_path, "w") as f:
    f.write(significance_results)
print(f"Significance test results saved to {sig_file_path}")

print("Experiment 2 completed: All models trained, predictions saved, plots and significance tests generated.")
