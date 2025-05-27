import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model, GPT2Config, AdamW, AutoModel, AutoConfig, BertModel, BertConfig, BitsAndBytesConfig
from scipy.stats import wilcoxon
from tqdm import tqdm

# if needed, change to where you would like model results to be saved
BASE_SAVE_DIR = os.path.join(os.getcwd(), os.pardir, os.pardir, "results", "experiment_3")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# rename...
RESULTS_DIR = BASE_SAVE_DIR

# this should point to where the exp1-4_data folder and subfolders are...
DATA_DIR = os.path.join(os.getcwd(), os.pardir, os.pardir, "exp1-4_data", "data_prepped_for_models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

fish_num = 9
data_dir = DATA_DIR # rename...
neural_data = np.load(os.path.join(data_dir, f"fish{fish_num}_neural_data_matched.npy"), allow_pickle=True)[:, :-2]
tail_data   = np.load(os.path.join(data_dir, f"fish{fish_num}_tail_data_matched.npy"), allow_pickle=True)

neural_data = neural_data.T
assert neural_data.shape[0] == tail_data.shape[0]

total_frames = neural_data.shape[0]
train_end = int(0.7 * total_frames)
val_end   = int(0.8 * total_frames)

X_train = neural_data[:train_end]
X_val   = neural_data[train_end:val_end]
X_test  = neural_data[val_end:]
Y_train = tail_data[:train_end]
Y_val   = tail_data[train_end:val_end]
Y_test  = tail_data[val_end:]

np.save(os.path.join(RESULTS_DIR, "groundtruth_val.npy"), Y_val)
np.save(os.path.join(RESULTS_DIR, "groundtruth_test.npy"), Y_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)

input_dim  = X_train_t.size(-1)
output_dim = Y_train_t.size(-1)

def create_sequences(inputs, targets, seq_length):
    seq_in, seq_out = [], []
    for i in range(len(inputs) - seq_length + 1):
        seq_in.append(inputs[i : i + seq_length])
        seq_out.append(targets[i : i + seq_length])
    return torch.stack(seq_in), torch.stack(seq_out)

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
        return x + self.pos_encoding[:, :L, :]

class RelativePositionEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, max_dist=32):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.max_dist = max_dist
        num_positions = 2 * max_dist + 1
        self.rel_embed = nn.Embedding(num_positions, hidden_size)
    def forward(self, x):
        x_lin = self.linear(x)
        B, L, _ = x_lin.shape
        positions = torch.arange(L, device=x_lin.device)
        offset = torch.clamp(positions - positions[0], -self.max_dist, self.max_dist)
        offset_index = offset + self.max_dist
        pos_emb = self.rel_embed(offset_index).unsqueeze(0).expand(B, L, -1)
        return x_lin + pos_emb

class SparseAutoencoderEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, sparsity_weight=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        B, L, input_dim = x.shape
        x_flat = x.reshape(B * L, input_dim)
        encoded = self.activation(self.encoder(x_flat))
        return encoded.reshape(B, L, -1)

class SpectralEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        nn.init.orthogonal_(self.linear.weight)
        self.activation = nn.Tanh()
    def forward(self, x):
        return self.activation(self.linear(x))

quant_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_threshold=6.0)

hidden_size_deepseek = 4096
class DeepSeekWithEmbedding(nn.Module):
    def __init__(self, embedding, output_dim):
        super().__init__()
        self.embedding = embedding
        self.model = AutoModel.from_pretrained("deepseek-ai/deepseek-coder-7b", trust_remote_code=True, device_map="auto", quantization_config=quant_config)
        for param in self.model.parameters():
            param.requires_grad = False
        self.output_proj = nn.Linear(hidden_size_deepseek, output_dim)
    def forward(self, x):
        x_emb = self.embedding(x)
        outputs = self.model(inputs_embeds=x_emb)
        return self.output_proj(outputs.last_hidden_state)

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
        x_emb = self.embedding(x)
        outputs = self.transformer(inputs_embeds=x_emb, position_ids=position_ids)
        return self.output_proj(outputs.last_hidden_state)

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
        x_emb = self.embedding(x)
        outputs = self.bert(inputs_embeds=x_emb)
        return self.output_proj(outputs.last_hidden_state)

NUM_RUNS   = 10
seq_length = 20
num_epochs = 3
batch_size = 2
lr_default = 1e-4

train_seq_x, train_seq_y = create_sequences(X_train_t, Y_train_t, seq_length)
val_seq_x,   val_seq_y   = create_sequences(X_val_t,   Y_val_t,   seq_length)
test_seq_x,  test_seq_y  = create_sequences(X_test_t,  Y_test_t,  seq_length)

train_loader = DataLoader(TensorDataset(train_seq_x, train_seq_y), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_seq_x,   val_seq_y),   batch_size=batch_size)
test_loader  = DataLoader(TensorDataset(test_seq_x,  test_seq_y),  batch_size=batch_size)

embedding_models = {
    "Vanilla":    VanillaLinearEmbedding,
    "Positional": PositionalLinearEmbedding,
    "RelativePos": RelativePositionEmbedding,
    "Sparse":     SparseAutoencoderEmbedding,
    "Spectral":   SpectralEmbedding
}

families = {
    "DeepSeek": {"ModelClass": DeepSeekWithEmbedding, "hidden_size": hidden_size_deepseek},
    "GPT2":     {"ModelClass": GPT2WithEmbedding,      "hidden_size": gpt2_hidden_size},
    "BERT":     {"ModelClass": BERTWithEmbedding,      "hidden_size": bert_hidden_size}
}

results = {f: {emb: [] for emb in embedding_models} for f in families}

for run_idx in range(1, NUM_RUNS+1):
    run_folder = os.path.join(RESULTS_DIR, f"run_{run_idx}")
    os.makedirs(run_folder, exist_ok=True)
    seed = 1234 + run_idx
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for family_name, fam_params in families.items():
        family_folder = os.path.join(run_folder, family_name.lower() + "_embedding_comparisons")
        os.makedirs(family_folder, exist_ok=True)

        for emb_name, emb_class in embedding_models.items():
            hidden_size = fam_params["hidden_size"]
            embedding_instance = emb_class(input_dim, hidden_size)
            ModelClass = fam_params["ModelClass"]
            model = ModelClass(embedding_instance, output_dim).to(device)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default)
            train_model(model, optimizer, train_loader, val_loader, device, num_epochs)
            preds_tensor = get_predictions(model, test_loader, device)
            final_preds = average_sliding_window_predictions(preds_tensor, seq_length, len(X_test_t))
            rmse_val = compute_rmse(final_preds, Y_test)
            results[family_name][emb_name].append(rmse_val)
            emb_subfolder = os.path.join(family_folder, emb_name.lower())
            os.makedirs(emb_subfolder, exist_ok=True)
            np.save(os.path.join(emb_subfolder, f"{family_name.lower()}_{emb_name.lower()}_preds_run{run_idx}.npy"), final_preds)
            truth_path = os.path.join(emb_subfolder, f"{family_name.lower()}_{emb_name.lower()}_groundtruth.npy")
            if not os.path.exists(truth_path):
                np.save(truth_path, Y_test)

final_plots_folder = os.path.join(RESULTS_DIR, "final_plots_and_stats")
os.makedirs(final_plots_folder, exist_ok=True)

for family_name in families:
    emb_names = list(results[family_name].keys())
    rmse_lists = [results[family_name][emb] for emb in emb_names]
    means = [np.mean(r) for r in rmse_lists]
    stderrs = [np.std(r)/np.sqrt(len(r)) for r in rmse_lists]
    x = np.arange(len(emb_names))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x, means, yerr=stderrs, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(emb_names, rotation=15)
    ax.set_ylabel("RMSE")
    ax.set_title(f"{family_name} - Embedding Comparison (Mean ± SEM, n={NUM_RUNS})")
    plt.tight_layout()
    plt.savefig(os.path.join(final_plots_folder, f"{family_name.lower()}_embeddings_barplot.png"))
    plt.close()
    sig_file = os.path.join(final_plots_folder, f"{family_name.lower()}_wilcoxon_significance.txt")
    with open(sig_file, "w") as f:
        f.write(f"Significance tests for {family_name}\nPairwise Wilcoxon tests:\n\n")
        for i in range(len(emb_names)):
            for j in range(i+1, len(emb_names)):
                emb1, emb2 = emb_names[i], emb_names[j]
                stat, pval = wilcoxon(results[family_name][emb1], results[family_name][emb2])
                f.write(f"{emb1} vs {emb2}: p-value = {pval:.4e}\n")

print("Experiment 3 completed.")
