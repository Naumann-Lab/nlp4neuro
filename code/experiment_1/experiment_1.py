# Jacob Morra, June 2025
# RNNs + transformers are trained + tested on sequence-to-sequence data, i.e. larval zebrafish neuron population decoding.

# imports
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2Model, AdamW, AutoModel, BertModel, BertConfig, BitsAndBytesConfig
from scipy.stats import mannwhitneyu

# CONFIGURATION --------

# use .yaml file to set directories...
from config import DATA_DIR, RESULTS_DIR

# load quantized version of DeepSeek-c7b
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# save data in experiment 1 sub-directory
BASE_SAVE_DIR = RESULTS_DIR / "experiment_1"
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# load data from experiment sub-directory
DATA_DIR = DATA_DIR / "exp1-4_data" / "data_prepped_for_models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DATA ASSIGNMENT -------
fish_list = [9, 10, 11, 12, 13]

# PARAMETERS --------
seq_lengths = [5, 10, 15, 20]
num_epochs = 10
batch_size = 32
lr_default = 1e-4
num_runs = 10

# run the main loop
for fish_num in fish_list:
    print(f"Processing Fish {fish_num}...")

    fish_save_dir = BASE_SAVE_DIR / f"fish{fish_num}"
    fish_save_dir.mkdir(parents=True, exist_ok=True)

    neural_data = np.load(str(DATA_DIR / f"fish{fish_num}_neural_data_matched.npy"), allow_pickle=True)[:, :-2]
    tail_data = np.load(str(DATA_DIR / f"fish{fish_num}_tail_data_matched.npy"), allow_pickle=True)

    neural_data = neural_data.T
    print("Neural data shape:", neural_data.shape)
    print("Tail data shape:", tail_data.shape)

    assert neural_data.shape[0] == tail_data.shape[0], "Mismatch in data length"

    total_frames = neural_data.shape[0]
    train_end = int(0.7 * total_frames)
    val_end = int(0.8 * total_frames)

    X_train = neural_data[:train_end]
    X_val = neural_data[train_end:val_end]
    X_test = neural_data[val_end:]
    Y_train = tail_data[:train_end]
    Y_val = tail_data[train_end:val_end]
    Y_test = tail_data[val_end:]

    np.save(str(fish_save_dir / f"fish{fish_num}_final_predictions_groundtruth_val.npy"), Y_val)
    np.save(str(fish_save_dir / f"fish{fish_num}_final_predictions_groundtruth_test.npy"), Y_test)
    print("Ground truth for val/test saved.")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    input_dim = X_train_t.size(-1)
    output_dim = Y_train_t.size(-1)
    print(f"Input_dim={input_dim}, Output_dim={output_dim}")

    def create_sequences(inputs, targets, seq_length):
        sequences_inputs = []
        sequences_targets = []
        for i in range(len(inputs) - seq_length + 1):
            sequences_inputs.append(inputs[i : i + seq_length])
            sequences_targets.append(targets[i : i + seq_length])
        return torch.stack(sequences_inputs), torch.stack(sequences_targets)

    # generate output predictions, averaging overlapping sequence frames
    def average_sliding_window_predictions(predictions, seq_length, total_length):
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
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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

    # GPT-2
    class CustomGPT2ModelPretrained(nn.Module):
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

    # LSTM
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out)
            return out

    # RC
    class ReservoirComputer(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.9):
            super().__init__()
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
            outputs = []
            for t in range(seq_len):
                x_t = x[:, t, :]
                hidden = (1 - self.alpha) * hidden + self.alpha * torch.tanh(x_t @ self.W_in + hidden @ self.W)
                y_t = self.readout(hidden)
                outputs.append(y_t.unsqueeze(1))
            return torch.cat(outputs, dim=1)

    # DeepSeek-c7b
    hidden_size_deepseek = 4096
    num_experts = 8
    top_k = 2

    class DeepSeekV3MoE(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_size_deepseek)
            self.model = AutoModel.from_pretrained(
                "deepseek-ai/deepseek-coder-7b",
                trust_remote_code=True,
                device_map="auto",
                quantization_config=quant_config
            )
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

    # BERT-bu
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
            return self.output_proj(hidden_states)

    model_names_list = ["GPT2 Pretrained", "LSTM", "Reservoir", "DeepSeek MoE", "BERT"]

    final_all_rmse = {seq: {model: [] for model in model_names_list} for seq in seq_lengths}

    for run in range(1, num_runs + 1):
        print("\n==============================")
        print(f"Fish {fish_num} - Run {run} of {num_runs}")
        print("==============================")

        run_folder = fish_save_dir / f"run_{run}"
        run_folder.mkdir(exist_ok=True)

        for seq_length in seq_lengths:
            print("\n------------------------------")
            print(f"Fish {fish_num} - Run {run}, Sequence Length = {seq_length}")
            print("------------------------------")

            seq_folder = run_folder / f"seq_{seq_length}"
            seq_folder.mkdir(exist_ok=True)

            train_neural_seq, train_tail_seq = create_sequences(X_train_t, Y_train_t, seq_length)
            val_neural_seq, val_tail_seq = create_sequences(X_val_t, Y_val_t, seq_length)
            test_neural_seq, test_tail_seq = create_sequences(X_test_t, Y_test_t, seq_length)

            train_dataset = TensorDataset(train_neural_seq, train_tail_seq)
            val_dataset = TensorDataset(val_neural_seq, val_tail_seq)
            test_dataset = TensorDataset(test_neural_seq, test_tail_seq)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            rmse_results = {}

            print("\nTraining GPT-2 (Pretrained)...")
            model_gpt2_pretrained = CustomGPT2ModelPretrained(input_dim, output_dim).to(device)
            optimizer_gpt2_pretrained = AdamW(model_gpt2_pretrained.parameters(), lr=lr_default)
            train_model(model_gpt2_pretrained, optimizer_gpt2_pretrained, train_loader, val_loader, device, num_epochs)

            preds, _ = get_predictions(model_gpt2_pretrained, test_loader, device)
            final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
            save_path = seq_folder / f"fish{fish_num}_final_predictions_gpt2_pretrained_test_run{run}.npy"
            np.save(str(save_path), final_preds)

            rmse = compute_rmse(final_preds, Y_test)
            rmse_results["GPT2 Pretrained"] = rmse
            print(f"GPT-2 Pretrained Overall RMSE: {rmse:.4f}")

            print("\nTraining LSTM...")
            lstm_hidden_dim = 256
            model_lstm = LSTMModel(input_dim, lstm_hidden_dim, output_dim).to(device)
            optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=lr_default)
            train_model(model_lstm, optimizer_lstm, train_loader, val_loader, device, num_epochs)

            preds, _ = get_predictions(model_lstm, test_loader, device)
            final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
            save_path = seq_folder / f"fish{fish_num}_final_predictions_lstm_test_run{run}.npy"
            np.save(str(save_path), final_preds)

            rmse = compute_rmse(final_preds, Y_test)
            rmse_results["LSTM"] = rmse
            print(f"LSTM Overall RMSE: {rmse:.4f}")

            print("\nTraining Reservoir Computer...")
            res_hidden_dim = 256
            alpha = 0.9
            model_res = ReservoirComputer(input_dim, res_hidden_dim, output_dim, alpha).to(device)
            optimizer_res = optim.Adam(model_res.readout.parameters(), lr=lr_default)
            train_model(model_res, optimizer_res, train_loader, val_loader, device, num_epochs)

            preds, _ = get_predictions(model_res, test_loader, device)
            final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
            save_path = seq_folder / f"fish{fish_num}_final_predictions_reservoir_test_run{run}.npy"
            np.save(str(save_path), final_preds)

            rmse = compute_rmse(final_preds, Y_test)
            rmse_results["Reservoir"] = rmse
            print(f"Reservoir Overall RMSE: {rmse:.4f}")

            print("\nTraining DeepSeek-V3 MoE...")
            model_deepseek = DeepSeekV3MoE(input_dim, output_dim).to(device)
            for param in model_deepseek.model.parameters():
                param.requires_grad = False
            for param in model_deepseek.router.parameters():
                param.requires_grad = True
            for param in model_deepseek.output_proj.parameters():
                param.requires_grad = True

            optimizer_deepseek = AdamW(filter(lambda p: p.requires_grad, model_deepseek.parameters()), lr=lr_default)
            train_model(model_deepseek, optimizer_deepseek, train_loader, val_loader, device, num_epochs)

            preds, _ = get_predictions(model_deepseek, test_loader, device)
            final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
            save_path = seq_folder / f"fish{fish_num}_final_predictions_deepseek_moe_test_run{run}.npy"
            np.save(str(save_path), final_preds)

            rmse = compute_rmse(final_preds, Y_test)
            rmse_results["DeepSeek MoE"] = rmse
            print(f"DeepSeek MoE Overall RMSE: {rmse:.4f}")

            print("\nTraining Small BERT Model...")
            bert_hidden_size = 768
            model_bert = BERTPretrainedModel(input_dim, bert_hidden_size, output_dim).to(device)
            optimizer_bert = AdamW(model_bert.parameters(), lr=lr_default)
            train_model(model_bert, optimizer_bert, train_loader, val_loader, device, num_epochs)

            preds, _ = get_predictions(model_bert, test_loader, device)
            final_preds = average_sliding_window_predictions(preds, seq_length, len(X_test_t))
            save_path = seq_folder / f"fish{fish_num}_final_predictions_bert_test_run{run}.npy"
            np.save(str(save_path), final_preds)

            rmse = compute_rmse(final_preds, Y_test)
            rmse_results["BERT"] = rmse
            print(f"BERT Overall RMSE: {rmse:.4f}")

            for model in model_names_list:
                final_all_rmse[seq_length][model].append(rmse_results[model])

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
        offset = (i - num_models / 2) * width + width / 2
        ax.bar(x + offset, mean_rmse_by_model[model], width=width, yerr=stderr_rmse_by_model[model], capsize=5, label=model)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Overall RMSE")
    ax.set_title(f"Overall RMSE Comparison Across Sequence Lengths for Fish {fish_num}\n(Mean ± SEM over runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend()
    plt.tight_layout()
    final_plot_path = fish_save_dir / f"fish{fish_num}_grouped_rmse_comparison.png"
    plt.savefig(str(final_plot_path))
    print(f"Final grouped bar plot saved to {final_plot_path}")

    significance_results = ""
    for seq in seq_lengths:
        significance_results += f"Sequence Length {seq}:\n"
        for i in range(len(model_names_list)):
            for j in range(i + 1, len(model_names_list)):
                model_a = model_names_list[i]
                model_b = model_names_list[j]
                vals_a = np.array(final_all_rmse[seq][model_a])
                vals_b = np.array(final_all_rmse[seq][model_b])
                stat, p_val = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                significance_results += f"  {model_a} vs {model_b}: p-value = {p_val:.4e}\n"
        significance_results += "\n"

    sig_file_path = fish_save_dir / f"fish{fish_num}_significance_results.txt"
    with open(sig_file_path, "w") as f:
        f.write(significance_results)
    print(f"Significance test results saved to {sig_file_path}")

    print(f"Fish {fish_num}: All models trained, predictions saved, and significance tests completed.\n")

print("Experiment 1 completed for fish 9, 10, 11, 12, and 13.")
