import json
import matplotlib.pyplot as plt
import os

output_dir = "therapml/dpo/dpo_model_adapter/checkpoint-500"
state_file = os.path.join(output_dir, "trainer_state.json")

if not os.path.exists(state_file):
    print(f"Error: Could not find {state_file}.")
    print("If your training crashed before finishing, check inside the 'checkpoint-XXX' folders instead.")
    exit()

with open(state_file, "r") as f:
    state_data = json.load(f)

log_history = state_data.get("log_history", [])

train_steps = []
train_losses = []

for log in log_history:
    if "loss" in log and "step" in log:
        train_steps.append(log["step"])
        train_losses.append(log["loss"])

plt.figure(figsize=(10, 6))

if train_steps:
    plt.plot(train_steps, train_losses, label='Training Loss', color='blue', linewidth=2)

plt.title('DPO Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

output_image = "therapml/dpo/loss_curves.png"
plt.savefig(output_image, dpi=300, bbox_inches="tight")
print(f"Plot saved successfully as {output_image}")

plt.show()