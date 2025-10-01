import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Disable tokenizers parallelism to avoid forking issues with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_environment():
    """Load environment-specific .env file based on DEPLOYMENT_ENV variable."""

    deployment_env = os.getenv("DEPLOYMENT_ENV", "local")

    # Map deployment environments to .env files
    env_files = {
        "local": ".env",
        "docker": "docker.env",
        "apptainer": "apptainer.env",
    }

    env_file = env_files.get(deployment_env, ".env")
    env_path = Path(__file__).parent.parent / env_file

    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"Warning: {env_file} not found, using defaults")

# Load environment variables on module import
load_environment()


@dataclass
class Config:
    """Unified configuration class for the text classifier."""

    # Model parameters
    model_name: str = os.getenv("MODEL_NAME", "bert-base-uncased")
    num_classes: int = int(os.getenv("NUM_CLASSES", "8"))
    max_length: int = int(os.getenv("MAX_LENGTH", "512"))
    dropout: float = float(os.getenv("DROPOUT", "0.1"))
    hidden_size: int = int(os.getenv("HIDDEN_SIZE", "768"))

    # Training parameters
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    num_epochs: int = int(os.getenv("NUM_EPOCHS", "3"))
    warmup_steps: int = int(os.getenv("WARMUP_STEPS", "500"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.01"))
    max_grad_norm: float = float(os.getenv("MAX_GRAD_NORM", "1.0"))
    device: str = "auto"  # Will be automatically detected during training
    seed: int = int(os.getenv("SEED", "42"))
    save_dir: str = os.getenv("SAVE_DIR", "./checkpoints")
    log_dir: str = os.getenv("LOG_DIR", "./logs")
    log_interval: int = int(os.getenv("LOG_INTERVAL", "100"))
    resume_from_checkpoint: str = os.getenv("RESUME_FROM_CHECKPOINT", "")  # Path to checkpoint file

    # Data parameters
    train_file: str = os.getenv("TRAIN_FILE", "./data/train_data.parquet")
    val_file: str = os.getenv("VAL_FILE", "./data/val_data.parquet")
    test_file: str = os.getenv("TEST_FILE", "./data/test_data.parquet")
    text_column: str = os.getenv("TEXT_COLUMN", "text")
    label_column: str = os.getenv("LABEL_COLUMN", "label")
    val_split: float = float(os.getenv("VAL_SPLIT", "0.2"))

    # Distributed training parameters (single-node multi-GPU)
    distributed: bool = False  # Will be set automatically based on GPU count
    world_size: int = 1  # Will be set automatically
    rank: int = 0  # Will be set automatically
    local_rank: int = 0  # Will be set automatically

    # GPU settings
    force_single_gpu: bool = os.getenv("FORCE_SINGLE_GPU", "false").lower() == "true"
