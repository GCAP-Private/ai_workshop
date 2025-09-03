import torch

def main():
    print("Hello, World!")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")

if __name__ == "__main__":
    main()
