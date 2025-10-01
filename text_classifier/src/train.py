"""
Training script for BERT-based text classifier.

This module handles the complete training pipeline including data loading,
model training, validation, checkpointing, and distributed training support.

Supported Training Modes:
- Single GPU (CUDA): Automatically detected
- Multi-GPU (CUDA): Use torchrun for distributed training (NVIDIA GPUs only)
- Apple Silicon (MPS): Single device only, no multi-GPU support
- CPU: Fallback when no GPU is available

Note: Distributed training is only available with NVIDIA CUDA GPUs.
Apple Silicon (MPS) does not support PyTorch distributed training.
"""
import os
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, get_linear_schedule_with_warmup

from config import Config
from model import TextClassificationDataset, create_model, create_tokenizer


# ============================================================================
# Distributed Training Utilities (Single-Node Multi-GPU)
# ============================================================================

def setup_logging(rank: int = 0, log_dir: str = './logs') -> None:
    """
    Setup logging for training.

    Args:
        rank: Process rank (0 for main process)
        log_dir: Directory to store log files
    """
    # Only log from main process to avoid duplicate logs
    if rank == 0:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create log file path with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f'training_{timestamp}.log'

        # Configure logging with both console and file output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='w')
            ],
            force=True
        )
        logging.info(f"Logging to {log_file}")
    else:
        logging.basicConfig(level=logging.WARNING)


def setup_distributed() -> tuple[int, int]:
    """
    Initialize distributed training for single-node multi-GPU.

    Returns:
        Tuple of (rank, world_size)
    """
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    return rank, world_size


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """Check if current process is main process."""
    return rank == 0


class DistributedMetrics:
    """Track and aggregate metrics across distributed processes."""

    def __init__(self, world_size: int = 1):
        self.world_size = world_size
        self.reset()

    def reset(self):
        """Reset metrics."""
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0

    def update(self, loss: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor):
        """Update metrics with batch results."""
        self.total_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(outputs, dim=1)
        self.total_correct += (predictions == labels).sum().item()
        self.total_samples += labels.size(0)

    def compute_average(self) -> Tuple[float, float]:
        """Compute average loss and accuracy across all processes."""
        if self.world_size > 1 and torch.distributed.is_initialized():
            # Aggregate across all processes
            metrics = torch.tensor([self.total_loss, self.total_correct, self.total_samples],
                                   device=torch.cuda.current_device())
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            total_loss, total_correct, total_samples = metrics.tolist()
        else:
            total_loss = self.total_loss
            total_correct = self.total_correct
            total_samples = self.total_samples

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy


def auto_detect_gpus(config: Config) -> Config:
    """
    Auto-detect and configure GPUs for training.

    Args:
        config: Configuration instance

    Returns:
        Updated configuration with GPU settings
    """
    # Check if distributed training should be enabled
    # Note: Distributed training only works with CUDA (NVIDIA GPUs)
    # MPS (Apple Silicon) does not support distributed training

    if torch.cuda.device_count() > 1 and not config.force_single_gpu:
        # Multi-GPU CUDA training
        config.distributed = True
        config.world_size = torch.cuda.device_count()
        config.rank = int(os.environ.get('RANK', 0))
        config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        logging.info(f"Detected {config.world_size} CUDA GPUs - enabling distributed training")
    else:
        # Single device training (single GPU, MPS, or CPU)
        config.distributed = False
        config.world_size = 1
        config.rank = 0
        config.local_rank = 0

        if torch.cuda.device_count() == 1:
            logging.info("Using single CUDA GPU training")
        elif torch.backends.mps.is_available():
            logging.info("Using Apple Silicon GPU (MPS) - single device only")
            logging.info("Note: Multi-GPU training is not supported on Apple Silicon")
        else:
            logging.info("Using CPU")

    return config


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[int, float]:
    """
    Load checkpoint to resume training.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        optimizer: Optimizer instance to load state into
        device: Device to load checkpoint on

    Returns:
        Tuple of (start_epoch, best_accuracy)
    """
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model state dict (handle DDP wrapper)
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Get training state
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
    best_accuracy = checkpoint.get('best_accuracy', 0.0)

    logging.info(f"Resuming from epoch {start_epoch}, best accuracy: {best_accuracy:.4f}")

    return start_epoch, best_accuracy


# ============================================================================
# Main Training Functions
# ============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(config: Config) -> Tuple[List[str], np.ndarray]:
    """
    Load training data from parquet file.

    Args:
        config: Config instance with file paths and column names

    Returns:
        Tuple of (texts, labels) where labels is a numpy array

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing
    """
    data_path = Path(config.train_file)
    # Load parquet file
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {data_path}: {e}") from e

    # Validate required columns
    required_cols = [config.text_column, config.label_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Extract texts and labels
    texts = df[config.text_column].astype(str).tolist()

    # Handle labels - convert to int64, handling various formats
    label_series = df[config.label_column]

    # Check if labels are sequences (lists/arrays) and extract first element if needed
    if isinstance(label_series.iloc[0], (list, np.ndarray)):
        labels = np.array([int(label[0]) for label in label_series], dtype=np.int64)
    else:
        labels = label_series.astype(np.int64).to_numpy()

    logging.info(f"Loaded {len(texts)} samples from {data_path}")
    return texts, labels


def load_validation_data(config: Config) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    Load validation data from parquet file if available.

    Args:
        config: Config instance with file paths and column names

    Returns:
        Tuple of (texts, labels) where labels is a numpy array, or None if file doesn't exist
    """
    val_path = Path(config.val_file)

    if not val_path.exists():
        logging.info(f"Validation file {val_path} not found. Will use train/val split.")
        return None

    try:
        df = pd.read_parquet(val_path)
        texts = df[config.text_column].astype(str).tolist()

        # Handle labels - convert to int64, handling various formats
        label_series = df[config.label_column]

        # Check if labels are sequences (lists/arrays) and extract first element if needed
        if isinstance(label_series.iloc[0], (list, np.ndarray)):
            labels = np.array([int(label[0]) for label in label_series], dtype=np.int64)
        else:
            labels = label_series.astype(np.int64).to_numpy()

        logging.info(f"Loaded {len(texts)} validation samples from {val_path}")
        return texts, labels
    except Exception as e:
        logging.warning(f"Failed to load validation data: {e}. Will use train/val split.")
        return None


def create_data_loader(
    texts: List[str],
    labels: np.ndarray,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
    distributed: bool = False
) -> DataLoader:
    """
    Create a PyTorch DataLoader for training or validation.

    Args:
        texts: List of text strings
        labels: Numpy array of integer labels
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        distributed: Whether to use distributed sampler

    Returns:
        Configured DataLoader instance
    """
    # Convert numpy array to list for dataset
    labels_list = labels.tolist()
    dataset = TextClassificationDataset(texts, labels_list, tokenizer, max_length)

    # Use DistributedSampler for multi-GPU training
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Don't shuffle when using DistributedSampler
    else:
        sampler = None

    # Configure DataLoader settings based on device
    # MPS doesn't support pin_memory, and it's only beneficial for CUDA
    use_pin_memory = torch.cuda.is_available()
    num_workers = 4 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0

    # Drop last incomplete batch during training to avoid BatchNorm issues with batch_size=1
    # Keep all data during validation/inference
    drop_last_batch = shuffle  # shuffle=True implies training mode

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        drop_last=drop_last_batch
    )


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    criterion: nn.Module,
    config: Config,
    metrics: Optional[DistributedMetrics] = None
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        data_loader: Training data loader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        criterion: Loss function
        config: Configuration instance
        metrics: Metrics tracker (created if None)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    if metrics is None:
        metrics = DistributedMetrics(config.world_size)
    metrics.reset()

    # Set sampler epoch for distributed training
    if hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(config.rank)

    pbar = tqdm(
        data_loader,
        desc="Training",
        disable=not is_main_process(config.rank)
    )

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.max_grad_norm
        )

        # Update weights
        optimizer.step()
        scheduler.step()

        # Update metrics
        metrics.update(loss, outputs, labels)

        # Log progress
        if is_main_process(config.rank) and \
           (batch_idx + 1) % config.log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return metrics.compute_average()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    criterion: nn.Module,
    config: Config,
    metrics: Optional[DistributedMetrics] = None
) -> Tuple[float, float, float, float, float]:
    """
    Evaluate model on validation/test set.

    Args:
        model: Model to evaluate
        data_loader: Validation data loader
        device: Device to evaluate on (cuda/cpu)
        criterion: Loss function
        config: Configuration instance
        metrics: Metrics tracker (created if None)

    Returns:
        Tuple of (avg_loss, accuracy, precision, recall, f1)
    """
    model.eval()
    if metrics is None:
        metrics = DistributedMetrics(config.world_size)
    metrics.reset()

    all_predictions = []
    all_labels = []

    pbar = tqdm(
        data_loader,
        desc="Evaluating",
        disable=not is_main_process(config.rank)
    )

    with torch.no_grad():
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            # Update metrics
            metrics.update(loss, outputs, labels)

            # Store predictions and labels for detailed metrics (only on main process)
            if is_main_process(config.rank):
                pred_classes = torch.argmax(outputs, dim=1)
                all_predictions.extend(pred_classes.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss, accuracy = metrics.compute_average()

    # Calculate detailed metrics only on main process
    if is_main_process(config.rank) and all_predictions:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0

    return avg_loss, accuracy, precision, recall, f1


def train_model() -> None:
    """
    Main training pipeline.

    Handles configuration, data loading, model initialization,
    training loop, validation, and checkpointing.
    """
    # Initialize configuration
    config = Config()

    # Auto-detect and configure GPUs
    config = auto_detect_gpus(config)

    # Setup logging
    setup_logging(config.rank, config.log_dir)

    # Setup distributed training if enabled (multi-GPU)
    if config.distributed:
        rank, world_size = setup_distributed()
        config.rank = rank
        config.world_size = world_size

        # Set device based on local rank
        config.device = f"cuda:{config.local_rank}"
        torch.cuda.set_device(config.local_rank)
        logging.info(f"Distributed training: rank {rank}/{world_size} on {config.device}")
    else:
        # Single GPU/CPU training - prioritize Apple Silicon MPS
        if torch.backends.mps.is_available():
            config.device = "mps"
            logging.info("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            config.device = "cuda:0"
            logging.info("Using NVIDIA CUDA GPU")
        else:
            config.device = "cpu"
            logging.info("Using CPU")

    # Set seed for reproducibility (different seed per rank in distributed)
    set_seed(config.seed + config.rank)

    # Log configuration
    if is_main_process(config.rank):
        logging.info("=" * 70)
        logging.info("Starting training with configuration:")
        logging.info(f"  Model: {config.model_name}")
        logging.info(f"  Num classes: {config.num_classes}")
        logging.info(f"  Max length: {config.max_length}")
        logging.info(f"  Batch size: {config.batch_size}")
        logging.info(f"  Learning rate: {config.learning_rate}")
        logging.info(f"  Epochs: {config.num_epochs}")
        logging.info(f"  Device: {config.device}")
        logging.info(f"  Distributed: {config.distributed}")
        if config.distributed:
            logging.info(f"  World size: {config.world_size}")
        logging.info("=" * 70)

    # Load data
    texts, labels = load_data(config)
    tokenizer = create_tokenizer(config.model_name)

    # Check for separate validation file
    val_data = load_validation_data(config)

    if val_data is not None:
        # Use separate validation file
        train_texts, train_labels = texts, labels
        val_texts, val_labels = val_data
        logging.info("Using separate validation file")
    else:
        # Split training data for validation
        total_size = len(texts)
        train_size = int((1 - config.val_split) * total_size)

        indices = np.arange(total_size)
        np.random.shuffle(indices)

        train_texts = [texts[i] for i in indices[:train_size]]
        train_labels = labels[indices[:train_size]]
        val_texts = [texts[i] for i in indices[train_size:]]
        val_labels = labels[indices[train_size:]]

        if is_main_process(config.rank):
            logging.info(
                f"Split data: {len(train_texts)} train, {len(val_texts)} validation"
            )

    # Create data loaders
    train_loader = create_data_loader(
        train_texts, train_labels, tokenizer,
        config.max_length, config.batch_size,
        shuffle=True, distributed=config.distributed
    )
    val_loader = create_data_loader(
        val_texts, val_labels, tokenizer,
        config.max_length, config.batch_size,
        shuffle=False, distributed=config.distributed
    )

    # Create model
    model = create_model(config)

    # Freeze BERT encoder - only train classification head
    model.freeze_bert_encoder()

    model.to(config.device)

    if is_main_process(config.rank):
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")
        logging.info(f"BERT encoder frozen - training classification head only")

    # Wrap model with DDP for distributed training
    if config.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            find_unused_parameters=False
        )
        logging.info(f"Model wrapped with DistributedDataParallel")

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Setup learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Create metrics trackers
    train_metrics = DistributedMetrics(config.world_size)
    val_metrics = DistributedMetrics(config.world_size)

    # Create save directory only on main process
    if is_main_process(config.rank):
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming training
    start_epoch = 0
    best_accuracy = 0.0

    # Determine checkpoint to load
    checkpoint_to_load = None
    if config.resume_from_checkpoint:
        # User specified a checkpoint
        checkpoint_to_load = config.resume_from_checkpoint
    else:
        # Auto-load best_model.pt if it exists
        default_checkpoint = Path(config.save_dir) / 'best_model.pt'
        if default_checkpoint.exists():
            checkpoint_to_load = str(default_checkpoint)
            if is_main_process(config.rank):
                logging.info(f"Found existing checkpoint: {checkpoint_to_load}")

    # Load checkpoint if we have one
    if checkpoint_to_load and is_main_process(config.rank):
        try:
            start_epoch, best_accuracy = load_checkpoint(
                checkpoint_to_load,
                model,
                optimizer,
                config.device
            )
            logging.info(f"Successfully loaded checkpoint, resuming from epoch {start_epoch}")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
            logging.warning("Starting training from scratch")
            start_epoch = 0
            best_accuracy = 0.0
    elif is_main_process(config.rank):
        logging.info("No checkpoint found, starting training from scratch")

    try:
        # Training loop
        for epoch in range(start_epoch, config.num_epochs):
            if is_main_process(config.rank):
                logging.info("")
                logging.info("=" * 70)
                logging.info(f"Epoch {epoch + 1}/{config.num_epochs}")
                logging.info("=" * 70)

            # Training phase
            train_loss, train_accuracy = train_epoch(
                model, train_loader, optimizer, scheduler,
                config.device, criterion, config,
                metrics=train_metrics
            )

            # Validation phase
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(
                model, val_loader, config.device,
                criterion, config, val_metrics
            )

            # Log results (only on main process)
            if is_main_process(config.rank):
                logging.info("")
                logging.info("Epoch Results:")
                logging.info("-" * 70)
                logging.info(f"{'Metric':<20} {'Train':<15} {'Validation':<15}")
                logging.info("-" * 70)
                logging.info(f"{'Loss':<20} {train_loss:<15.4f} {val_loss:<15.4f}")
                logging.info(f"{'Accuracy':<20} {train_accuracy:<15.4f} {val_accuracy:<15.4f}")
                logging.info(f"{'Precision':<20} {'-':<15} {val_precision:<15.4f}")
                logging.info(f"{'Recall':<20} {'-':<15} {val_recall:<15.4f}")
                logging.info(f"{'F1 Score':<20} {'-':<15} {val_f1:<15.4f}")
                logging.info("-" * 70)

            # Save best model (only on main process)
            if val_accuracy > best_accuracy and is_main_process(config.rank):
                best_accuracy = val_accuracy
                checkpoint_path = Path(config.save_dir) / 'best_model.pt'

                # Extract model state dict (handle DDP wrapper)
                model_state_dict = (
                    model.module.state_dict() if hasattr(model, 'module')
                    else model.state_dict()
                )

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'model_config': config,
                }, checkpoint_path)

                logging.info("")
                logging.info(f"✓ New best model saved (accuracy: {best_accuracy:.4f})")

        if is_main_process(config.rank):
            logging.info("")
            logging.info("=" * 70)
            logging.info(f"Training completed successfully!")
            logging.info(f"Best validation accuracy: {best_accuracy:.4f}")
            logging.info("=" * 70)

    except KeyboardInterrupt:
        if is_main_process(config.rank):
            logging.warning("")
            logging.warning("Training interrupted by user")
    except Exception as e:
        if is_main_process(config.rank):
            logging.error("")
            logging.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup distributed training
        if config.distributed:
            cleanup_distributed()


if __name__ == "__main__":
    train_model()