from huggingface_hub import create_repo, upload_folder

# Create repo on HF
repo_id = "RayhanSefatTBD/my-lora-adapter-1"
create_repo(repo_id, exist_ok=True)

# Upload the folder
upload_folder(
    folder_path="therapml/lora/gpt2_eli5_results/final_adapter",
    repo_id=repo_id,
    repo_type="model"
)
