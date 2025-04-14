import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    GPT2Config, GPT2Model, AdamW, AutoModel, BertModel, BertConfig, BitsAndBytesConfig
)
from scipy.stats import wilcoxon

################################################################################
# 1) Global Parameters
################################################################################
# Adjust as needed
BASE_SAVE_DIR = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment_2"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# To avoid OOM with a large float 7B model, use a small batch_size
BATCH_SIZE   = 2   
NUM_EPOCHS   = 3   
LR_DEFAULT   = 1e-4
NUM_RUNS     = 10        # now 10 runs/trials
SEQ_LENGTHS  = [5, 20]
FISH_LIST    = [9, 10, 11, 12, 13]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("[INFO] GPU:", torch.cuda.get_device_name(0))
print(f"[INFO] Running on device: {device}")

################################################################################
# 2) 8-bit Quantization Config for DeepSeek PRETRAINED
################################################################################
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

################################################################################
# 3) Helper Functions
################################################################################
def create_sequences(inputs, targets, seq_length):
    """
    Make sliding-window sequences of length seq_length.
    outputs: (N - seq_length+1, seq_length, input_dim)
    """
    seq_x, seq_y = [], []
    for i in range(len(inputs) - seq_length + 1):
        seq_x.append(inputs[i : i + seq_length])
        seq_y.append(targets[i : i + seq_length])
    return torch.stack(seq_x), torch.stack(seq_y)

def average_sliding_window_predictions(predictions, seq_length, total_length):
    """
    Convert overlapping predictions back to frame-aligned by averaging the overlaps.
    predictions: (num_windows, seq_length, output_dim)
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
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

def get_predictions(model, data_loader, device):
    model.eval()
    preds_list, targets_list = [], []
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
# 4) Model Classes
################################################################################
# GPT-2 Pretrained vs Untrained
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

# BERT Pretrained vs Untrained
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

# DeepSeek Pretrained (8-bit) vs Untrained (float, random)
HIDDEN_SIZE_DEEPSEEK = 4096
NUM_EXPERTS = 8
TOP_K = 2

class DeepSeekPretrainedMoE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, HIDDEN_SIZE_DEEPSEEK)
        # 8-bit pretrained
        self.model = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config
        )
        self.router = nn.Linear(HIDDEN_SIZE_DEEPSEEK, NUM_EXPERTS)
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(HIDDEN_SIZE_DEEPSEEK, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        # MoE routing
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, TOP_K, dim=-1)[1]
        aggregated = torch.zeros_like(hidden_states)
        for i in range(TOP_K):
            idx = top_k_experts[..., i].unsqueeze(-1).expand_as(hidden_states)
            expert_output = torch.gather(hidden_states, dim=-1, index=idx)
            aggregated += expert_output / TOP_K
        return self.output_proj(aggregated)

class DeepSeekUntrainedMoE(nn.Module):
    """
    Builds a random float32 7B model from config. Partial freeze to only train
    input_proj, router, output_proj.  No 8-bit available if bitsandbytes is old.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, HIDDEN_SIZE_DEEPSEEK)
        # Step 1: get config from deepseek-coder-7b
        base_config = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-coder-7b",
            trust_remote_code=True
        ).config
        # Step 2: random model
        self.model = AutoModel.from_config(base_config, trust_remote_code=True)
        with torch.no_grad():
            for param in self.model.parameters():
                param.normal_(mean=0.0, std=0.02)
        # Router + output
        self.router = nn.Linear(HIDDEN_SIZE_DEEPSEEK, NUM_EXPERTS)
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(HIDDEN_SIZE_DEEPSEEK, output_dim)
        # random init for router & output_proj
        with torch.no_grad():
            for param in self.router.parameters():
                param.normal_(mean=0.0, std=0.02)
            for param in self.output_proj.parameters():
                param.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        x = self.input_proj(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        # MoE routing
        router_logits = self.router(hidden_states)
        routing_weights = self.softmax(router_logits)
        top_k_experts = torch.topk(routing_weights, TOP_K, dim=-1)[1]
        aggregated = torch.zeros_like(hidden_states)
        for i in range(TOP_K):
            idx = top_k_experts[..., i].unsqueeze(-1).expand_as(hidden_states)
            expert_output = torch.gather(hidden_states, dim=-1, index=idx)
            aggregated += expert_output / TOP_K
        return self.output_proj(aggregated)

################################################################################
# 5) Model Variants
################################################################################
model_variants = [
    ("GPT2 Pretrained", GPT2PretrainedModel),
    ("GPT2 Untrained", GPT2UntrainedModel),
    ("BERT Pretrained", BERTPretrainedModel),
    ("BERT Untrained", BERTUntrainedModel),
    ("DeepSeek Pretrained", DeepSeekPretrainedMoE),
    ("DeepSeek Untrained", DeepSeekUntrainedMoE),
]

################################################################################
# 6) Main Experiment
################################################################################
for fish_num in FISH_LIST:
    print("\n===========================================")
    print(f"Starting Experiment 2 for Fish {fish_num}")
    print("===========================================")

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
    assert neural_data.shape[0] == tail_data.shape[0], "Mismatch in data length"
    print("Neural data shape:", neural_data.shape)
    print("Tail data shape:", tail_data.shape)

    total_frames = neural_data.shape[0]
    train_end = int(0.7 * total_frames)
    val_end   = int(0.8 * total_frames)

    X_train = neural_data[:train_end]
    X_val   = neural_data[train_end:val_end]
    X_test  = neural_data[val_end:]
    Y_train = tail_data[:train_end]
    Y_val   = tail_data[train_end:val_end]
    Y_test  = tail_data[val_end:]

    # Save the ground truth test data (once per fish) in the base directory
    gt_path = os.path.join(BASE_SAVE_DIR, f"fish{fish_num}_test_groundtruth.npy")
    np.save(gt_path, Y_test)

    # Convert to torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)

    input_dim  = X_train_t.size(-1)
    output_dim = Y_train_t.size(-1)
    print(f"[Fish {fish_num}] input_dim={input_dim}, output_dim={output_dim}")

    # final_rmse[seq_len][model_name] = list of RMSE across runs
    final_rmse = {
        seq_len: {mv[0]: [] for mv in model_variants}
        for seq_len in SEQ_LENGTHS
    }

    ############################################################################
    # Multiple runs: 10 trials
    ############################################################################
    for run_idx in range(1, NUM_RUNS+1):
        print(f"\n=== Fish {fish_num} - Run {run_idx}/{NUM_RUNS} ===")

        for seq_length in SEQ_LENGTHS:
            print(f"\n--- Sequence Length = {seq_length} ---")

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

            # Train/Eval all model variants
            for model_name, model_class in model_variants:
                print(f"\n[Run {run_idx}] Training {model_name} ...")

                # If DeepSeek Untrained => big float model
                if model_name == "DeepSeek Untrained":
                    model = model_class(input_dim, output_dim).to(device)
                else:
                    model = model_class(input_dim, output_dim).to(device)

                # Partial freeze for DeepSeek
                if "DeepSeek" in model_name:
                    # freeze base model
                    for param in model.model.parameters():
                        param.requires_grad = False
                    # unfreeze input_proj, router, output_proj
                    for param in model.input_proj.parameters():
                        param.requires_grad = True
                    for param in model.router.parameters():
                        param.requires_grad = True
                    for param in model.output_proj.parameters():
                        param.requires_grad = True

                optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR_DEFAULT)

                # Train
                train_model(model, optimizer, train_loader, val_loader, device, NUM_EPOCHS)

                # Predict on test
                preds_tensor, _ = get_predictions(model, test_loader, device)
                final_preds = average_sliding_window_predictions(
                    preds_tensor, seq_length, len(X_test_t)
                )

                # Save predictions in base directory:
                # e.g. fish9_model_DeepSeek_Untrained_run5_seq20_test_preds.npy
                pred_filename = (
                    f"fish{fish_num}_model_{model_name.replace(' ','_')}"
                    f"_run{run_idx}_seq{seq_length}_test_preds.npy"
                )
                pred_path = os.path.join(BASE_SAVE_DIR, pred_filename)
                np.save(pred_path, final_preds)

                # Compute RMSE
                rmse_val = compute_rmse(final_preds, Y_test)
                final_rmse[seq_length][model_name].append(rmse_val)
                print(f"{model_name} => RMSE: {rmse_val:.4f}")

    ############################################################################
    # After all runs => bar plots + Wilcoxon
    ############################################################################
    # One bar plot per seq_length
    for seq_length in SEQ_LENGTHS:
        mnames = [mv[0] for mv in model_variants]
        rmse_data = [final_rmse[seq_length][mn] for mn in mnames]
        means = [np.mean(r) for r in rmse_data]
        stderrs = [np.std(r)/np.sqrt(len(r)) for r in rmse_data]

        x = np.arange(len(mnames))
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(x, means, yerr=stderrs, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(mnames, rotation=20)
        ax.set_ylabel("RMSE")
        ax.set_title(
            f"Fish {fish_num} - SeqLength {seq_length}\n"
            f"Mean ± SEM over {NUM_RUNS} runs"
        )
        plt.tight_layout()

        # Save bar plot in fish subfolder
        barplot_path = os.path.join(
            BASE_SAVE_DIR,
            f"fish{fish_num}_seq{seq_length}_barplot.png"
        )
        plt.savefig(barplot_path)
        plt.close()
        print(f"Saved bar plot => {barplot_path}")

    # Wilcoxon test: pretrained vs. untrained for each family
    sig_results = []
    for seq_length in SEQ_LENGTHS:
        sig_results.append(f"=== Sequence Length = {seq_length} ===")

        # GPT-2
        gp = np.array(final_rmse[seq_length]["GPT2 Pretrained"])
        gu = np.array(final_rmse[seq_length]["GPT2 Untrained"])
        stat, pval = wilcoxon(gp, gu)
        sig_results.append(f"GPT2 (Pre vs Un): p-value = {pval:.4e}")

        # BERT
        bp = np.array(final_rmse[seq_length]["BERT Pretrained"])
        bu = np.array(final_rmse[seq_length]["BERT Untrained"])
        stat, pval = wilcoxon(bp, bu)
        sig_results.append(f"BERT (Pre vs Un): p-value = {pval:.4e}")

        # DeepSeek
        dp = np.array(final_rmse[seq_length]["DeepSeek Pretrained"])
        du = np.array(final_rmse[seq_length]["DeepSeek Untrained"])
        stat, pval = wilcoxon(dp, du)
        sig_results.append(f"DeepSeek (Pre vs Un): p-value = {pval:.4e}")

        sig_results.append("")

    # Save significance results in base directory
    sig_filename = f"fish{fish_num}_wilcoxon_results.txt"
    sig_path = os.path.join(BASE_SAVE_DIR, sig_filename)
    with open(sig_path, "w") as f:
        for line in sig_results:
            f.write(line + "\n")
    print(f"[INFO] Wilcoxon results => {sig_path}")
    print(f"=== Done with Fish {fish_num} ===\n")

print("[INFO] All done. Experiment 2 completed for all fish!")
