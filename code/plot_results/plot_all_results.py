# plot 1

import os
import numpy as np
import matplotlib.pyplot as plt

# Define directories and sequence lengths
BASE_SAVE_DIR = "compare_models_results"
seq_lengths = [5, 10, 15, 20]

# Models and file key mapping
model_names = ["GPT-2", "LSTM", "RC", "DeepSeek-7B", "BERT"]
model_file_keys = {
    "GPT-2": "gpt2_pretrained",
    "LSTM": "lstm",
    "RC": "reservoir",
    "DeepSeek-7B": "deepseek_moe",
    "BERT": "bert"
}

# Load ground truth test predictions
gt_path = os.path.join(BASE_SAVE_DIR, "final_predictions_groundtruth_test.npy")
ground_truth = np.load(gt_path)

# Initialize dictionary to collect RMSE values per model
rmse_results = {model: [] for model in model_names}

# Compute RMSE for each sequence length and model
for seq in seq_lengths:
    seq_folder = os.path.join(BASE_SAVE_DIR, f"seq_{seq}")
    for model in model_names:
        file_key = model_file_keys[model]
        pred_path = os.path.join(seq_folder, f"final_predictions_{file_key}_test.npy")
        predictions = np.load(pred_path)
        rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
        rmse_results[model].append(rmse)

# Create grouped bar plot
x = np.arange(len(seq_lengths))  # one group per sequence length
num_models = len(model_names)
width = 0.8 / num_models  # bar width within a group

plt.figure(figsize=(10, 6), dpi=300)
bars = []  # Store bar containers for annotation
for i, model in enumerate(model_names):
    offset = (i - num_models / 2) * width + width / 2
    bar = plt.bar(x + offset, rmse_results[model], width=width, label=model)
    bars.append(bar)

# Add RMSE values above each bar with extra space
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        offset = 0.005  # Increase or decrease this value to adjust space between the bar and text
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=10
        )

plt.xlabel(r"$s$", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.xticks(x, seq_lengths, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

# Save as PNG
png_plot_path = os.path.join(BASE_SAVE_DIR, "grouped_rmse_comparison_highdpi.png")
plt.savefig(png_plot_path, dpi=300)
print(f"Grouped bar plot saved to {png_plot_path}")

# Also save as PDF
pdf_plot_path = os.path.join(BASE_SAVE_DIR, "grouped_rmse_comparison_highdpi.pdf")
plt.savefig(pdf_plot_path, format='pdf', dpi=300)
print(f"Grouped bar plot also saved to {pdf_plot_path}")

plt.show()



# plot 1b

import os
import numpy as np
import matplotlib.pyplot as plt

# Define the base directory
BASE_SAVE_DIR = "compare_models_results"

# File paths for the ground truth and the two GPT-2 variants
gt_path = os.path.join(BASE_SAVE_DIR, "final_predictions_groundtruth_test.npy")
gpt2_pretrained_path = os.path.join(BASE_SAVE_DIR, "final_predictions_gpt2_pretrained_test.npy")
gpt2_scratch_path = os.path.join(BASE_SAVE_DIR, "final_predictions_gpt2_scratch_test.npy")

# Load the data
ground_truth = np.load(gt_path)
gpt2_pretrained = np.load(gpt2_pretrained_path)
gpt2_scratch = np.load(gpt2_scratch_path)

# Compute RMSE for GPT-2 Pretrained
rmse_pretrained = np.sqrt(np.mean((gpt2_pretrained - ground_truth) ** 2))

# Compute RMSE for GPT-2 Scratch
rmse_scratch = np.sqrt(np.mean((gpt2_scratch - ground_truth) ** 2))

# Prepare data for plotting
models = ["Pre-trained", "Un-trained"]
rmses = [rmse_pretrained, rmse_scratch]

# Create a bar plot
plt.figure(figsize=(3, 3), dpi=300)
bars = plt.bar(models, rmses, color=['skyblue', 'salmon'], width=0.5)

# Annotate bars with RMSE values (rotated 90 degrees for clarity, with 3 decimals)
offset = 0.005# Adjust this value to increase/decrease the space between the bar and the text
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + offset,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=12
    )

plt.ylabel("RMSE", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save and display the plot as PNG
png_plot_path = os.path.join(BASE_SAVE_DIR, "gpt2_vs_scratch_rmse_comparison_highdpi.png")
plt.savefig(png_plot_path, dpi=300)
print(f"GPT-2 vs. GPT-2 Scratch RMSE bar plot saved to {png_plot_path}")

# Also save as PDF
pdf_plot_path = os.path.join(BASE_SAVE_DIR, "gpt2_vs_scratch_rmse_comparison_highdpi.pdf")
plt.savefig(pdf_plot_path, format='pdf', dpi=300)
print(f"GPT-2 vs. GPT-2 Scratch RMSE bar plot also saved to {pdf_plot_path}")

plt.show()



# plot 2b

import os
import matplotlib.pyplot as plt

# Define the pixel coordinates for the y-axis mapping:
# RMSE = 0.08 at pixel 40, RMSE = 0.00 at pixel 543.
y_top = 40      # corresponds to RMSE = 0.08
y_bottom = 543  # corresponds to RMSE = 0.00
max_rmse = 0.08

def pixel_to_rmse(y_pixel):
    """
    Convert a pixel y-coordinate (from the top) into an RMSE value.
    """
    return max_rmse * (y_bottom - y_pixel) / (y_bottom - y_top)

# Given pixel positions for the top of each bar (from the top):
blue_bar_top = 137    # Blue bar top at pixel 137
orange_bar_top = 61   # Orange bar top at pixel 61

# Compute the RMSE values:
blue_rmse = pixel_to_rmse(blue_bar_top)
orange_rmse = pixel_to_rmse(orange_bar_top)

# Now, redraw the bar plot using these RMSE values:
labels = ["Pre-trained", "Un-trained"]
values = [blue_rmse, orange_rmse]
colors = ["dodgerblue", "coral"]

plt.figure(figsize=(3, 3), dpi=300)
bars = plt.bar(labels, values, color=colors, width=0.5)

# Annotate bars with RMSE values (rotated 90 degrees) with extra vertical space
offset = 0.002  # Adjust this value to control the space between the bar top and the text
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + offset,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=12
    )

plt.ylabel("RMSE", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save the plot as PNG and PDF
BASE_SAVE_DIR = "compare_models_results"
png_plot_path = os.path.join(BASE_SAVE_DIR, "pretrained_vs_vanilla_deepseek_rmse_comparison_highdpi.png")
plt.savefig(png_plot_path, dpi=300)
print(f"Pre-trained vs. Un-trained RMSE bar plot saved to {png_plot_path}")

pdf_plot_path = os.path.join(BASE_SAVE_DIR, "pretrained_vs_vanilla_deepseek_rmse_comparison_highdpi.pdf")
plt.savefig(pdf_plot_path, format='pdf', dpi=300)
print(f"Pre-trained vs. Un-trained RMSE bar plot saved to {pdf_plot_path}")

plt.show()


# plot 3

import os
import numpy as np
import matplotlib.pyplot as plt

# Define parameters and directories
BASE_SAVE_DIR = "compare_models_results"
seq_length = 20
seq_folder = os.path.join(BASE_SAVE_DIR, f"seq_{seq_length}")

# List of models (using the same keys as before)
model_names = ["GPT-2", "LSTM", "RC", "DeepSeek-7B", "BERT"]
model_file_keys = {
    "GPT-2": "gpt2_pretrained",
    "LSTM": "lstm",
    "RC": "reservoir",
    "DeepSeek-7B": "deepseek_moe",
    "BERT": "bert"
}

# Load ground truth test data
gt_path = os.path.join(BASE_SAVE_DIR, "final_predictions_groundtruth_test.npy")
ground_truth = np.load(gt_path)

# Load predictions for each model
predictions_dict = {}
for model in model_names:
    file_key = model_file_keys[model]
    pred_path = os.path.join(seq_folder, f"final_predictions_{file_key}_test.npy")
    predictions_dict[model] = np.load(pred_path)

# For clarity, we plot a segment (for example, the first 890 time steps).
start_idx = 0
end_idx = 890
time_axis = np.arange(start_idx, end_idx)

# Define the tail columns (assumes tail components are columns 1 through 6)
tail_columns = slice(1, 7)  # Columns 1 to 6 (6 tail components)

plt.figure(figsize=(12, 5), dpi=300)
# Compute tail sum for ground truth and plot it in bold black (without alpha)
tail_sum_gt = np.sum(ground_truth[start_idx:end_idx, tail_columns], axis=1)
plt.plot(time_axis, tail_sum_gt, 'k-', linewidth=2.5, label="Ground truth")

# Compute and plot tail sum for each model's predictions with an alpha value for transparency
for model in model_names:
    tail_sum_pred = np.sum(predictions_dict[model][start_idx:end_idx, tail_columns], axis=1)
    plt.plot(time_axis, tail_sum_pred, linewidth=1.8, label=model, alpha=0.7)

plt.xlabel(r"$t$", fontsize=14)
plt.ylabel(r"$\theta_{\mathrm{sum}}$", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot as PNG and PDF
png_plot_path = os.path.join(BASE_SAVE_DIR, "test_predictions_seq20_tail_sum_highdpi.png")
plt.savefig(png_plot_path, dpi=300)
print(f"Test set tail sum predictions plot saved to {png_plot_path}")

pdf_plot_path = os.path.join(BASE_SAVE_DIR, "test_predictions_seq20_tail_sum_highdpi.pdf")
plt.savefig(pdf_plot_path, format='pdf', dpi=300)
print(f"Test set tail sum predictions plot saved to {pdf_plot_path}")

plt.show()
