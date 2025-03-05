import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# Import transformers and quantization config (for DeepSeek)
from transformers import GPT2Model, AdamW, AutoModel, BertModel, BertConfig, BitsAndBytesConfig

################################################################################
# 1) Set Directories and Experiment Parameters
################################################################################
BASE_SAVE_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment_5"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# Data directory pattern
DATA_DIR_PATTERN = "/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{}_neural_data_matched.npy"

# Experiment parameters
fish_list = [9, 10, 11, 12, 13]
seq_lengths = [5, 10, 15, 20]
num_runs = 10  # number of repeats of the pipeline

# CPC training parameters (adjust as needed)
num_epochs = 5
batch_size = 32
lr_cpc = 1e-4
prediction_step = 1  # predicting 1 step ahead in CPC
latent_dim = 128  # common latent dimension for CPC
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################################################################################
# 2) Define CPC Loss (InfoNCE)
################################################################################
def info_nce_loss(z, c, prediction_step=1, temperature=0.07):
    """
    Computes InfoNCE loss between context c_t and future latent z_{t+k}.

    Args:
        z: Tensor of shape (B, T, latent_dim) -- latent embeddings.
        c: Tensor of shape (B, T, latent_dim) -- context embeddings.
        prediction_step: int, number of steps ahead to predict.
        temperature: float, softmax temperature.
    
    Returns:
        Scalar tensor with averaged InfoNCE loss.
    """
    B, T, latent_dim = z.shape
    device = z.device
    total_loss = 0.0
    valid_steps = 0

    for t in range(T - prediction_step):
        c_t = c[:, t, :]                    # (B, latent_dim)
        z_tk = z[:, t + prediction_step, :]   # (B, latent_dim)

        # Use all time steps in the batch as negatives.
        z_all = z.view(B * T, latent_dim)
        c_t_expanded = c_t.unsqueeze(1).expand(-1, B * T, -1).contiguous().view(B * T, latent_dim)
        logits = torch.sum(c_t_expanded * z_all, dim=1)  # (B*T,)
        logits = logits.view(B, T)
        logits = logits / temperature

        targets = torch.tensor([t + prediction_step] * B, dtype=torch.long, device=device)
        loss_fn = nn.CrossEntropyLoss()
        batch_loss = loss_fn(logits, targets)
        total_loss += batch_loss
        valid_steps += 1

    return total_loss / valid_steps

################################################################################
# 3) Define Generic CPC Trainer and Evaluator
################################################################################
def train_cpc(model, train_loader, val_loader, device, num_epochs=10, prediction_step=1, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for x_batch in train_loader:
            x = x_batch[0].to(device)
            z, c = model(x)
            loss = info_nce_loss(z, c, prediction_step=prediction_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_batch in val_loader:
                x = x_batch[0].to(device)
                z, c = model(x)
                loss = info_nce_loss(z, c, prediction_step=prediction_step)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    return best_val_loss

def evaluate_cpc(model, data_loader, device, prediction_step=1):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch in data_loader:
            x = x_batch[0].to(device)
            z, c = model(x)
            loss = info_nce_loss(z, c, prediction_step=prediction_step)
            total_loss += loss.item()
    return total_loss / len(data_loader)

################################################################################
# 4) Define CPC Models (Adapted from your original models)
################################################################################
# 4.1 GPT-2 CPC Model
class GPT2CPC(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.gpt2_base = GPT2Model.from_pretrained("gpt2")
        hidden_size = self.gpt2_base.config.n_embd
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        x_emb = self.input_proj(x_flat)
        x_emb = x_emb.view(B, T, -1)
        outputs = self.gpt2_base(inputs_embeds=x_emb)
        hidden_states = outputs.last_hidden_state  # (B, T, hidden_size)
        z = self.out_proj(hidden_states)
        c = z
        return z, c

# 4.2 LSTM CPC Model
class LSTMCPC(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.proj_z = nn.Linear(hidden_dim, latent_dim)
        self.proj_c = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        z = self.proj_z(out)
        c = self.proj_c(out)
        return z, c

# 4.3 Reservoir CPC Model
class ReservoirCPC(nn.Module):
    def __init__(self, input_dim, res_size, latent_dim, alpha=0.9):
        super().__init__()
        self.res_size = res_size
        self.alpha = alpha
        W_in = torch.rand(input_dim, res_size) - 0.5
        W = torch.rand(res_size, res_size) - 0.5
        self.W_in = nn.Parameter(W_in, requires_grad=False)
        self.W = nn.Parameter(W, requires_grad=False)
        self.readout_z = nn.Linear(res_size, latent_dim)
        self.readout_c = nn.Linear(res_size, latent_dim)

    def forward(self, x):
        B, T, D = x.shape
        hidden_states = []
        h = torch.zeros(B, self.res_size, device=x.device)
        for t in range(T):
            x_t = x[:, t, :]
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(x_t @ self.W_in + h @ self.W)
            hidden_states.append(h.unsqueeze(1))
        hidden_states = torch.cat(hidden_states, dim=1)
        z = self.readout_z(hidden_states)
        c = self.readout_c(hidden_states)
        return z, c

# 4.4 DeepSeek CPC Model
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
class DeepSeekCPC(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        hidden = self.input_proj(x_flat)
        hidden = hidden.view(B, T, -1)
        outputs = self.model(inputs_embeds=hidden, output_hidden_states=True)
        last_h = outputs.last_hidden_state
        z = self.out_proj(last_h)
        c = z
        return z, c

# 4.5 BERT CPC Model
class BERTCPC(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim):
        super().__init__()
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=hidden_size * 4
        )
        self.bert = BertModel(config)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        x_emb = self.input_proj(x_flat)
        x_emb = x_emb.view(B, T, -1)
        outputs = self.bert(inputs_embeds=x_emb)
        h = outputs.last_hidden_state
        z = self.out_proj(h)
        c = z
        return z, c

################################################################################
# 5) Data Preparation Helper
################################################################################
def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i+seq_len])
    return torch.stack(sequences)

################################################################################
# 6) Main Experiment Loop: Self-Supervised CPC Training Across Models
################################################################################
def run_self_supervised_experiment():
    # Dictionary to store CPC losses for significance tests and plotting.
    model_names = ["GPT2", "LSTM", "Reservoir", "DeepSeek", "BERT"]
    final_all_cpc_losses = {seq: {model: [] for model in model_names} for seq in seq_lengths}
    
    for fish_num in fish_list:
        print(f"\n========== Processing Fish {fish_num} ==========")
        
        # Create a folder for this fish's results.
        fish_save_dir = os.path.join(BASE_SAVE_DIR, f"fish{fish_num}")
        os.makedirs(fish_save_dir, exist_ok=True)
        
        # Load neural data. We follow your original convention of removing the last two columns.
        neural_data = np.load(DATA_DIR_PATTERN.format(fish_num), allow_pickle=True)[:,:-2]
        # Transpose to (num_frames, num_neurons) if necessary.
        neural_data = neural_data.T
        print(f"Fish {fish_num}: Neural data shape: {neural_data.shape}")
        
        total_frames = neural_data.shape[0]
        train_end = int(0.7 * total_frames)
        val_end = int(0.8 * total_frames)
        X_train = neural_data[:train_end]
        X_val = neural_data[train_end:val_end]
        X_test = neural_data[val_end:]
        
        # Convert data to torch tensors.
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t   = torch.tensor(X_val, dtype=torch.float32)
        X_test_t  = torch.tensor(X_test, dtype=torch.float32)
        input_dim = X_train_t.shape[1]
        
        for run_id in range(1, num_runs + 1):
            print(f"\n--- Fish {fish_num}: Run {run_id} of {num_runs} ---")
            for seq_len in seq_lengths:
                print(f"\nSequence Length = {seq_len}")
                # Create sliding-window sequences.
                train_seq = create_sequences(X_train_t, seq_len)
                val_seq   = create_sequences(X_val_t, seq_len)
                test_seq  = create_sequences(X_test_t, seq_len)
                
                train_dataset = TensorDataset(train_seq)
                val_dataset   = TensorDataset(val_seq)
                test_dataset  = TensorDataset(test_seq)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # ------------------- GPT2 -------------------
                print("[Model: GPT2]")
                model_gpt2 = GPT2CPC(input_dim, latent_dim).to(device)
                train_cpc(model_gpt2, train_loader, val_loader, device, num_epochs=num_epochs,
                          prediction_step=prediction_step, lr=lr_cpc)
                test_loss_gpt2 = evaluate_cpc(model_gpt2, test_loader, device, prediction_step=prediction_step)
                print(f"GPT2 Test CPC Loss: {test_loss_gpt2:.4f}")
                final_all_cpc_losses[seq_len]["GPT2"].append(test_loss_gpt2)
                
                # ------------------- LSTM -------------------
                print("[Model: LSTM]")
                model_lstm = LSTMCPC(input_dim, hidden_dim=256, latent_dim=latent_dim).to(device)
                train_cpc(model_lstm, train_loader, val_loader, device, num_epochs=num_epochs,
                          prediction_step=prediction_step, lr=lr_cpc)
                test_loss_lstm = evaluate_cpc(model_lstm, test_loader, device, prediction_step=prediction_step)
                print(f"LSTM Test CPC Loss: {test_loss_lstm:.4f}")
                final_all_cpc_losses[seq_len]["LSTM"].append(test_loss_lstm)
                
                # ------------------- Reservoir -------------------
                print("[Model: Reservoir]")
                model_res = ReservoirCPC(input_dim, res_size=256, latent_dim=latent_dim, alpha=0.9).to(device)
                train_cpc(model_res, train_loader, val_loader, device, num_epochs=num_epochs,
                          prediction_step=prediction_step, lr=lr_cpc)
                test_loss_res = evaluate_cpc(model_res, test_loader, device, prediction_step=prediction_step)
                print(f"Reservoir Test CPC Loss: {test_loss_res:.4f}")
                final_all_cpc_losses[seq_len]["Reservoir"].append(test_loss_res)
                
                # ------------------- DeepSeek -------------------
                try:
                    print("[Model: DeepSeek]")
                    model_deepseek = DeepSeekCPC(input_dim, hidden_dim=512, latent_dim=latent_dim).to(device)
                    train_cpc(model_deepseek, train_loader, val_loader, device, num_epochs=num_epochs,
                              prediction_step=prediction_step, lr=lr_cpc)
                    test_loss_deepseek = evaluate_cpc(model_deepseek, test_loader, device, prediction_step=prediction_step)
                    print(f"DeepSeek Test CPC Loss: {test_loss_deepseek:.4f}")
                    final_all_cpc_losses[seq_len]["DeepSeek"].append(test_loss_deepseek)
                except Exception as e:
                    print("DeepSeek model failed to load/train. Recording NaN for CPC loss.")
                    final_all_cpc_losses[seq_len]["DeepSeek"].append(np.nan)
                
                # ------------------- BERT -------------------
                print("[Model: BERT]")
                model_bert = BERTCPC(input_dim, hidden_size=256, latent_dim=latent_dim).to(device)
                train_cpc(model_bert, train_loader, val_loader, device, num_epochs=num_epochs,
                          prediction_step=prediction_step, lr=lr_cpc)
                test_loss_bert = evaluate_cpc(model_bert, test_loader, device, prediction_step=prediction_step)
                print(f"BERT Test CPC Loss: {test_loss_bert:.4f}")
                final_all_cpc_losses[seq_len]["BERT"].append(test_loss_bert)
    
    ############################################################################
    # 7) Summarize Results: Significance Testing and Bar Plots
    ############################################################################
    for seq_len in seq_lengths:
        print(f"\n=== CPC Loss Results for seq_len={seq_len} ===")
        for model_name in model_names:
            arr = np.array(final_all_cpc_losses[seq_len][model_name])
            print(f"{model_name}: mean={arr.mean():.4f}, std={arr.std():.4f}")
            
        print("\nPairwise Significance Tests (Mann–Whitney U):")
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                mA, mB = model_names[i], model_names[j]
                arrA = np.array(final_all_cpc_losses[seq_len][mA])
                arrB = np.array(final_all_cpc_losses[seq_len][mB])
                try:
                    stat, p_val = mannwhitneyu(arrA, arrB, alternative='two-sided')
                    print(f"  {mA} vs {mB}: p-value = {p_val:.4e}")
                except Exception as ex:
                    print(f"  {mA} vs {mB}: comparison failed ({ex}).")
    
    for seq_len in seq_lengths:
        means = []
        sems = []
        for m in model_names:
            vals = np.array(final_all_cpc_losses[seq_len][m])
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)))
        x = np.arange(len(model_names))
        plt.figure()
        plt.bar(x, means, yerr=sems, capsize=5)
        plt.xticks(x, model_names)
        plt.ylabel("InfoNCE Loss (lower is better)")
        plt.title(f"CPC Loss Comparison (seq_len={seq_len})")
        plt.tight_layout()
        # Save the figure in the fish save directory
        plot_path = os.path.join(BASE_SAVE_DIR, f"cpc_loss_comparison_seq{seq_len}.png")
        plt.savefig(plot_path)
        print(f"Bar plot saved to {plot_path}")
        plt.show()
    
    print("Self-supervised CPC experiment complete!")

if __name__ == "__main__":
    run_self_supervised_experiment()
