import torch

if torch.cuda.is_available():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch was installed without CUDA support.")
