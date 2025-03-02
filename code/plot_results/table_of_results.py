import os
import numpy as np

# -------------------------------------------------------------------------
# Part A: Plot 1–2 data (RMSE and Correlation vs. Sequence Length for multiple models)
# -------------------------------------------------------------------------
BASE_SAVE_DIR = os.path.join(os.getcwd(), "experiment_1", "compare_models_results")
seq_lengths = [5, 10, 15, 20]

model_names = ["GPT-2", "LSTM", "RC", "DeepSeek-7B", "BERT"]
model_file_keys = {
    "GPT-2": "gpt2_pretrained",
    "LSTM": "lstm",
    "RC": "reservoir",
    "DeepSeek-7B": "deepseek_moe",
    "BERT": "bert"
}

# Load ground truth (assumed same for all sequence lengths)
gt_path = os.path.join(BASE_SAVE_DIR, "final_predictions_groundtruth_test.npy")
ground_truth = np.load(gt_path)

# Compute RMSE and correlation coefficient across seq_lengths for each model
rmse_results = {model: [] for model in model_names}
corr_results = {model: [] for model in model_names}

for seq in seq_lengths:
    seq_folder = os.path.join(BASE_SAVE_DIR, f"seq_{seq}")
    for model in model_names:
        file_key = model_file_keys[model]
        pred_path = os.path.join(seq_folder, f"final_predictions_{file_key}_test.npy")
        predictions = np.load(pred_path)
        # Compute RMSE
        rmse_val = np.sqrt(np.mean((predictions - ground_truth)**2))
        rmse_results[model].append(rmse_val)
        # Compute Pearson correlation coefficient between predictions and ground truth
        corr_coef = np.corrcoef(predictions.ravel(), ground_truth.ravel())[0, 1]
        corr_results[model].append(corr_coef)

# -------------------------------------------------------------------------
# Part B: Plot 3 data (GPT-2 Pre-trained vs. GPT-2 Scratch)
# -------------------------------------------------------------------------
gpt2_pretrained_path = os.path.join(BASE_SAVE_DIR, "final_predictions_gpt2_pretrained_test.npy")
gpt2_scratch_path    = os.path.join(BASE_SAVE_DIR, "final_predictions_gpt2_scratch_test.npy")
gpt2_pretrained      = np.load(gpt2_pretrained_path)
gpt2_scratch         = np.load(gpt2_scratch_path)

rmse_pretrained = np.sqrt(np.mean((gpt2_pretrained - ground_truth) ** 2))
rmse_scratch    = np.sqrt(np.mean((gpt2_scratch - ground_truth) ** 2))

# -------------------------------------------------------------------------
# Part C: Plot 4 data (Embedding RMSE)
# -------------------------------------------------------------------------
EMBEDDING_RESULTS_DIR = os.path.join(os.getcwd(), "experiment_1", "compare_embeddings_results")
embedding_methods_plot = ["vanilla", "positional", "cnn", "spectral", "sparse"]
embedding_labels = [
    "Linear",
    "Positional",
    "1D Convolution",
    "Laplacian Eigenmap",
    "Sparse Autoencoder"
]

rmse_embed = []
for method in embedding_methods_plot:
    rmse_path = os.path.join(EMBEDDING_RESULTS_DIR, f"final_rmse_{method}.npy")
    rmse_val = float(np.load(rmse_path))  # Should be a single scalar
    rmse_embed.append(rmse_val)

# -------------------------------------------------------------------------
# Generate ONE LaTeX table with ALL results
# -------------------------------------------------------------------------
latex_lines = []
latex_lines.append(r"\begin{table*}[ht!]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Summary of RMSE and Correlation Results from Plots 1--2, 3, and 4 (All in One Table).}")
latex_lines.append(r"\label{tab:all_results}")
latex_lines.append(r"\begin{tabular}{lcccc}")
latex_lines.append(r"\toprule")

# ------------------ PART A: Plot 1--2 Heading ------------------
latex_lines.append(r"\multicolumn{5}{l}{\textbf{Plot 1--2: RMSE and Correlation vs. Sequence Length}} \\")
latex_lines.append(r"\midrule")
latex_lines.append(r"\textbf{Model} & $s=5$ & $s=10$ & $s=15$ & $s=20$ \\")
latex_lines.append(r"\midrule")
for model in model_names:
    # First row: RMSE values with the model name
    rmse_line = f"{model}"
    for i in range(len(seq_lengths)):
        rmse_line += f" & {rmse_results[model][i]:.3f}"
    rmse_line += r" \\"
    # Second row: correlation coefficients (empty first cell)
    corr_line = " "  # Empty cell for the model name column
    for i in range(len(seq_lengths)):
        corr_line += f" & r = {corr_results[model][i]:.2f}"
    corr_line += r" \\"
    latex_lines.append(rmse_line)
    latex_lines.append(corr_line)

# ------------------ PART B: Plot 3 (GPT-2 variants) ------------------
latex_lines.append(r"\midrule")
latex_lines.append(r"\multicolumn{5}{l}{\textbf{Plot 3: GPT-2 Pre-trained vs. Scratch}} \\")
latex_lines.append(r"\midrule")
latex_lines.append(r"\textbf{Model Variant} & \multicolumn{4}{c}{\textbf{RMSE}} \\")
latex_lines.append(r"\midrule")
latex_lines.append(f"GPT-2 (Pre-trained) & \\multicolumn{{4}}{{c}}{{{rmse_pretrained:.3f}}} \\\\")
latex_lines.append(f"GPT-2 (Scratch)    & \\multicolumn{{4}}{{c}}{{{rmse_scratch:.3f}}} \\\\")

# ------------------ PART C: Plot 4 (Embeddings) ------------------
latex_lines.append(r"\midrule")
latex_lines.append(r"\multicolumn{5}{l}{\textbf{Plot 4: RMSE by Embedding Method}} \\")
latex_lines.append(r"\midrule")
latex_lines.append(r"\textbf{Embedding} & \multicolumn{4}{c}{\textbf{RMSE}} \\")
latex_lines.append(r"\midrule")
for emb_label, emb_rmse in zip(embedding_labels, rmse_embed):
    latex_lines.append(f"{emb_label} & \\multicolumn{{4}}{{c}}{{{emb_rmse:.3f}}} \\\\")
    
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table*}")

# Print the table to stdout
table_latex = "\n".join(latex_lines)
print(table_latex)
