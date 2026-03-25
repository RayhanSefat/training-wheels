import torch
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

# Configuration
checkpoint_dir = 'therapml/train_model/models/'
output_dir = 'therapml/train_model/gradient_plots/'
os.makedirs(output_dir, exist_ok=True)

# Gather all checkpoint files and sort by iteration
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_iter_*.pt"))
checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Dictionary to store gradients: {param_suffix: [[val_iter1], [val_iter2], ...]}
param_grads = defaultdict(list)
iterations = []

for file_path in checkpoint_files:
    try:
        iteration = int(file_path.split('_')[-1].split('.')[0])
        checkpoint = torch.load(file_path, map_location='cpu')
        model_weights = checkpoint.get('model_state_dict', checkpoint)
        iterations.append(iteration)
        # Temporary dict to collect values by suffix for this iteration
        suffix_vals = defaultdict(list)
        for name, param in model_weights.items():
            # Get the suffix (e.g., 'linear.weight')
            suffix = '.'.join(name.split('.')[-2:]) if '.' in name else name
            grad_val = param.abs().mean().item()
            suffix_vals[suffix].append(grad_val)
        # For each suffix, average across blocks for this iteration
        for suffix, vals in suffix_vals.items():
            avg_val = sum(vals) / len(vals)
            param_grads[suffix].append(avg_val)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

# Plot for each parameter suffix
for suffix, grads in param_grads.items():
    plt.figure()
    plt.plot(iterations, grads, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Average Weight Magnitude")
    plt.title(f"Weight Flow for {suffix} (merged across blocks)")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"weight_traj_{suffix.replace('.', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot for {suffix} at {save_path}")
