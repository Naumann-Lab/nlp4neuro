import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2Model, AdamW, AutoModel, BertModel, BertConfig, BitsAndBytesConfig
from scipy.stats import wilcoxon

################################################################################
# Optional: Print out some GPU info at startup to see if there's enough memory.
################################################################################
if torch.cuda.is_available():
    print("[INFO] Using GPU:", torch.cuda.get_device_name(0))
    print("[INFO] Initial memory allocated:", torch.cuda.memory_allocated(0) / 1e9, "GB")
    print("[INFO] Initial memory reserved:", torch.cuda.memory_reserved(0) / 1e9, "GB")

################################################################################
# Adjust if needed (reduce batch size if OOM).
################################################################################
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR_DEFAULT = 1e-4
NUM_RUNS = 10
SEQ_LENGTHS = [5, 20]  # as requested
FISH_LIST = [9, 10, 11, 12, 13]

BASE_SAVE_DIR = os.path.join(os.getcwd(), "results", "experiment_2_corrected")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] experiment_2_corrected running on device: {device}")

################################################################################
# Quantization config for DeepSeek model (8-bit, set threshold, etc.)
################################################################################
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

################################################################################
# Helper Functions
################################################################################
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
    num_windows, seq_len, out_dim = predictions.shape
    averaged = np.zeros((total_length, out_dim))
    counts = np.zeros(total_length)
    for i in range(num_windows):
        averaged[i : i + seq_length, :] += predictions[i]
        counts[i : i + seq_length] += 1
    averaged /= counts[:, None]
    return averaged

def compute_rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))

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
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    return

def get_predictions(model, data_loader, device):
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds_list.append(outputs.cpu())
            targets_list.append(targets.cpu())
    all_preds = torch.cat(preds_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    return all_preds, all_targets

################################################################################
# GPT2 (pretrained vs untrained)
################################################################################
class GPT2PretrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained("gpt2")
        hidden_size = self.transformer.config.n_embd
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.transformer(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

class GPT2UntrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        config = GPT2Config()
        self.transformer = GPT2Model(config)  # random weights
        hidden_size = config.n_embd
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.transformer(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

################################################################################
# BERT (pretrained vs untrained)
################################################################################
class BERTPretrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.bert(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

class BERTUntrainedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        config = BertConfig()  # random init
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.bert(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        return logits

################################################################################
# DeepSeek (Pretrained MoE) - same approach from experiment_4
# Using partial freeze, router, top_k, etc.
################################################################################
HIDDEN_SIZE_DEEPSEEK = 4096
NUM_EXPERTS = 8
TOP_K = 2

class DeepSeekPretrainedMoE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, HIDDEN_SIZE_DEEPSEEK)
        # Load pretrained DeepSeek
        self.model = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config
        )
        self.output_proj = nn.Linear(HIDDEN_SIZE_DEEPSEEK, output_dim)
        # MoE Router
        self.router = nn.Linear(HIDDEN_SIZE_DEEPSEEK, NUM_EXPERTS)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # last hidden states
        # MoE routing
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, TOP_K, dim=-1)[1]
        aggregated_output = torch.zeros_like(hidden_states)

        for i in range(TOP_K):
            expert_idx = top_k_experts[:, :, i].unsqueeze(-1).expand_as(hidden_states)
            expert_output = torch.gather(hidden_states, dim=-1, index=expert_idx)
            aggregated_output += expert_output / TOP_K
        # final linear
        logits = self.output_proj(aggregated_output)
        return logits

################################################################################
# DeepSeek (Untrained MoE) - random init, but same architecture
################################################################################
class DeepSeekUntrainedMoE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, HIDDEN_SIZE_DEEPSEEK)
        # Instead of from_pretrained, load from config for random init
        base_config = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True
        ).config
        self.model = AutoModel.from_config(base_config, trust_remote_code=True)
        self.output_proj = nn.Linear(HIDDEN_SIZE_DEEPSEEK, output_dim)
        # MoE Router
        self.router = nn.Linear(HIDDEN_SIZE_DEEPSEEK, NUM_EXPERTS)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, TOP_K, dim=-1)[1]
        aggregated_output = torch.zeros_like(hidden_states)

        for i in range(TOP_K):
            expert_idx = top_k_experts[:, :, i].unsqueeze(-1).expand_as(hidden_states)
            expert_output = torch.gather(hidden_states, dim=-1, index=expert_idx)
            aggregated_output += expert_output / TOP_K
        logits = self.output_proj(aggregated_output)
        return logits

################################################################################
# 3) Main Experiment 2
# Compare: GPT-2 (P vs U), BERT (P vs U), DeepSeek (P vs U)
################################################################################

model_variants = [
    ("GPT2 Pretrained", GPT2PretrainedModel),
    ("GPT2 Untrained", GPT2UntrainedModel),
    ("BERT Pretrained", BERTPretrainedModel),
    ("BERT Untrained", BERTUntrainedModel),
    ("DeepSeek Pretrained", DeepSeekPretrainedMoE),
    ("DeepSeek Untrained", DeepSeekUntrainedMoE),
]

for fish_num in FISH_LIST:
    print("\n===========================================")
    print(f"Starting Experiment 2 for Fish {fish_num}")
    print("===========================================")

    fish_save_dir = os.path.join(BASE_SAVE_DIR, f"fish{fish_num}")
    os.makedirs(fish_save_dir, exist_ok=True)

    # Load data
    neural_data = np.load(
        f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish_num}_neural_data_matched.npy",
        allow_pickle=True
    )[:, :-2]
    tail_data = np.load(
        f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish_num}_tail_data_matched.npy",
        allow_pickle=True
    )
    neural_data = neural_data.T  # shape: (num_frames, num_neurons)
    print("Neural data shape:", neural_data.shape)
    print("Tail data shape:", tail_data.shape)

    assert neural_data.shape[0] == tail_data.shape[0], "Mismatch in data length"
    total_frames = neural_data.shape[0]

    # Train/val/test split
    train_end = int(0.7 * total_frames)
    val_end = int(0.8 * total_frames)

    X_train = neural_data[:train_end]
    Y_train = tail_data[:train_end]
    X_val   = neural_data[train_end:val_end]
    Y_val   = tail_data[train_end:val_end]
    X_test  = neural_data[val_end:]
    Y_test  = tail_data[val_end:]

    np.save(os.path.join(fish_save_dir, f"fish{fish_num}_groundtruth_test.npy"), Y_test)

    # Convert to torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)

    input_dim = X_train_t.size(-1)
    output_dim = Y_train_t.size(-1)
    print(f"[Fish {fish_num}] input_dim={input_dim}, output_dim={output_dim}")

    # final_rmse[seq_len][model_name] = list of RMSE across runs
    final_rmse = {s: {m[0]: [] for m in model_variants} for s in SEQ_LENGTHS}

    ############################################################################
    # Multiple runs
    ############################################################################
    for run in range(1, NUM_RUNS + 1):
        print(f"\n=== Fish {fish_num} - Run {run}/{NUM_RUNS} ===")
        run_folder = os.path.join(fish_save_dir, f"run_{run}")
        os.makedirs(run_folder, exist_ok=True)

        for seq_length in SEQ_LENGTHS:
            print(f"\n--- Sequence Length = {seq_length} ---")
            seq_folder = os.path.join(run_folder, f"seq_{seq_length}")
            os.makedirs(seq_folder, exist_ok=True)

            # Create sliding-window sequences
            train_in_seq, train_out_seq = create_sequences(X_train_t, Y_train_t, seq_length)
            val_in_seq,   val_out_seq   = create_sequences(X_val_t,   Y_val_t,   seq_length)
            test_in_seq,  test_out_seq  = create_sequences(X_test_t,  Y_test_t,  seq_length)

            train_loader = DataLoader(TensorDataset(train_in_seq, train_out_seq), 
                                      batch_size=BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(TensorDataset(val_in_seq,   val_out_seq), 
                                      batch_size=BATCH_SIZE)
            test_loader  = DataLoader(TensorDataset(test_in_seq,  test_out_seq), 
                                      batch_size=BATCH_SIZE)

            for model_name, model_class in model_variants:
                print(f"\n[Run {run}] Training {model_name} ...")
                model = model_class(input_dim, output_dim).to(device)

                # If DeepSeek Pretrained, freeze all but router & output_proj
                if model_name == "DeepSeek Pretrained":
                    for param in model.model.parameters():
                        param.requires_grad = False
                    for param in model.router.parameters():
                        param.requires_grad = True
                    for param in model.output_proj.parameters():
                        param.requires_grad = True
                    for param in model.input_proj.parameters():
                        param.requires_grad = True

                # If still OOM, you can also freeze the input_proj or reduce the dimension.

                optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_DEFAULT)

                # Print GPU usage before training
                if torch.cuda.is_available():
                    print("[GPU before training] allocated:", 
                          torch.cuda.memory_allocated(0) / 1e9, "GB",
                          "| reserved:", torch.cuda.memory_reserved(0) / 1e9, "GB")

                train_model(model, optimizer, train_loader, val_loader, device, NUM_EPOCHS)

                # Print GPU usage after training
                if torch.cuda.is_available():
                    print("[GPU after training] allocated:", 
                          torch.cuda.memory_allocated(0) / 1e9, "GB",
                          "| reserved:", torch.cuda.memory_reserved(0) / 1e9, "GB")

                preds, _ = get_predictions(model, test_loader, device)
                final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))

                # Save predictions
                pred_file = os.path.join(seq_folder, f"{model_name.replace(' ','_')}_test_preds.npy")
                np.save(pred_file, final_preds)

                # Compute RMSE
                rmse_val = compute_rmse(final_preds, Y_test)
                final_rmse[seq_length][model_name].append(rmse_val)
                print(f"{model_name} - RMSE: {rmse_val:.4f}")

    ############################################################################
    # 4) Make bar plots comparing all six variants for each seq_length
    ############################################################################
    for seq_length in SEQ_LENGTHS:
        mnames = [m[0] for m in model_variants]
        rmse_data = [final_rmse[seq_length][mn] for mn in mnames]
        rmse_means = [np.mean(r) for r in rmse_data]
        rmse_stderrs = [np.std(r)/np.sqrt(len(r)) for r in rmse_data]

        x = np.arange(len(mnames))
        width = 0.6
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, rmse_means, yerr=rmse_stderrs, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(mnames, rotation=20)
        ax.set_ylabel("RMSE")
        ax.set_title(f"Fish {fish_num} - Seq Length {seq_length} (Mean ± SEM over {NUM_RUNS} runs)")
        plt.tight_layout()

        plot_path = os.path.join(fish_save_dir, f"fish{fish_num}_seq{seq_length}_barplot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved bar plot for seq {seq_length} to {plot_path}")

    ############################################################################
    # 5) Wilcoxon tests: pretrained vs untrained for each model class
    ############################################################################
    significance_results = []
    for seq_length in SEQ_LENGTHS:
        significance_results.append(f"--- Sequence Length = {seq_length} ---")

        # GPT-2
        gp = np.array(final_rmse[seq_length]["GPT2 Pretrained"])
        gu = np.array(final_rmse[seq_length]["GPT2 Untrained"])
        stat, pval = wilcoxon(gp, gu)
        significance_results.append(f"GPT2 (P vs U): p-value = {pval:.4e}")

        # BERT
        bp = np.array(final_rmse[seq_length]["BERT Pretrained"])
        bu = np.array(final_rmse[seq_length]["BERT Untrained"])
        stat, pval = wilcoxon(bp, bu)
        significance_results.append(f"BERT (P vs U): p-value = {pval:.4e}")

        # DeepSeek
        dp = np.array(final_rmse[seq_length]["DeepSeek Pretrained"])
        du = np.array(final_rmse[seq_length]["DeepSeek Untrained"])
        stat, pval = wilcoxon(dp, du)
        significance_results.append(f"DeepSeek (P vs U): p-value = {pval:.4e}")

        significance_results.append("")

    sig_file_path = os.path.join(fish_save_dir, f"fish{fish_num}_wilcoxon_results.txt")
    with open(sig_file_path, "w") as f:
        for line in significance_results:
            f.write(line + "\n")
    print(f"[INFO] Wilcoxon results saved to {sig_file_path}")
    print(f"=== Done with Fish {fish_num} ===\n")

print("[INFO] Experiment 2 completed for all fish.")
