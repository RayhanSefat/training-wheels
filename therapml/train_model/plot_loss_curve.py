import torch
import os
import re
import matplotlib.pyplot as plt
from .common import CHECKPOINT_FOLDER

def extract_loss_from_checkpoints(checkpoint_dir):
    steps = []
    losses = []

    pattern = re.compile(r"checkpoint_iter_(\d+)\.pt")
    
    files = [f for f in os.listdir(checkpoint_dir) if pattern.match(f)]
    
    files.sort(key=lambda x: int(pattern.match(x).group(1)))

    for filename in files:
        step = int(pattern.match(filename).group(1))
        file_path = os.path.join(checkpoint_dir, filename)
        
        checkpoint = torch.load(file_path, map_location='cpu')
        
        loss_val = checkpoint["loss"]
        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()
        
        steps.append(step)
        losses.append(loss_val)        

    return steps, losses

def plot_checkpoints(checkpoint_dir):
    steps, losses = extract_loss_from_checkpoints(checkpoint_dir)
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', linestyle='-', color='#2ca02c', markersize=4)
    
    plt.title("Validation Loss from Model Checkpoints", fontsize=14)
    plt.xlabel("Iteration (Step)", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("therapml/train_model/checkpoint_loss_curve_model_6.png")
    plt.show()

if __name__ == "__main__":
    plot_checkpoints(CHECKPOINT_FOLDER)