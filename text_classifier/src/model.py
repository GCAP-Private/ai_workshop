"""
Text classification model module.

This module provides the BERT-based classifier architecture, dataset handling,
and model checkpoint management utilities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from config import Config


class BERTClassifier(nn.Module):
    """
    BERT-based multi-label text classifier with CNN feature extraction.

    Architecture:
        BERT → Conv1D → MaxPooling → BatchNorm → Dense → Classifier

    Uses a pretrained BERT model followed by CNN layers for feature extraction
    and a classification head. Supports multi-label classification where each
    sample can belong to multiple classes simultaneously. Use with BCEWithLogitsLoss
    for training.

    Args:
        config: Config instance containing model hyperparameters

    Attributes:
        config: Stored model configuration
        bert: Pretrained BERT encoder
        conv1d: 1D convolutional layer for feature extraction
        maxpool: Adaptive max pooling layer
        batch_norm: Batch normalization layer
        dropout: Dropout layer for regularization
        dense: Dense hidden layer
        classifier: Linear classification head (outputs logits for each class)
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(
            config.model_name,
            output_hidden_states=False,
            output_attentions=False
        )

        # CNN layers for feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=256,
            kernel_size=5,
            padding=3
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        # Use track_running_stats=True but handle batch_size=1 in forward pass
        self.batch_norm = nn.BatchNorm1d(256, track_running_stats=True)

        # Regularization and classification layers
        self.dropout = nn.Dropout(config.dropout)
        self.dense = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, config.num_classes)

        # Initialize weights
        self._init_weights(self.conv1d)
        self._init_weights(self.dense)
        self._init_weights(self.classifier)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for CNN and classification layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            token_type_ids: Token type IDs of shape (batch_size, seq_length)

        Returns:
            Logits of shape (batch_size, num_classes) for multi-label classification
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get CLS token output (batch_size, hidden_size)
        # CLS token is the first token in the sequence (index 0)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Expand dimensions for Conv1d: (batch_size, hidden_size, 1)
        cls_output = cls_output.unsqueeze(-1)

        # Apply Conv1D + ReLU
        conv_output = torch.relu(self.conv1d(cls_output))

        # Apply MaxPooling: (batch_size, channels, 1)
        pooled_output = self.maxpool(conv_output)

        # Flatten: (batch_size, channels)
        pooled_output = pooled_output.squeeze(-1)

        # Apply Batch Normalization (skip if batch_size=1 during training)
        if self.training and pooled_output.size(0) == 1:
            # Skip batch norm for single sample batch during training
            normalized_output = pooled_output
        else:
            normalized_output = self.batch_norm(pooled_output)

        # Apply Dropout
        dropped_output = self.dropout(normalized_output)

        # Apply Dense layer + ReLU
        dense_output = torch.relu(self.dense(dropped_output))

        # Apply final classification layer
        logits = self.classifier(dense_output)

        return logits

    def freeze_bert_encoder(self) -> None:
        """Freeze BERT encoder parameters for feature extraction."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self) -> None:
        """Unfreeze BERT encoder parameters for fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True


class TextClassificationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for multi-label text classification.

    Handles tokenization and encoding of text data for model training/inference.

    Args:
        texts: List of text strings
        labels: List of label arrays/lists where each element is an array of class indices (optional for inference)
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length for tokenization
        num_classes: Number of classes for multi-label classification

    Attributes:
        texts: Stored text data
        labels: Stored label data
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        num_classes: Number of classes
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[List[int]]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        num_classes: int
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes

        # Validate inputs
        if labels is not None and len(texts) != len(labels):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts but {len(labels)} labels"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing input_ids, attention_mask, and label tensors
        """
        text = str(self.texts[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Prepare return dictionary
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        # Add label if available - labels are already one-hot encoded
        if self.labels is not None:
            # Labels are already in one-hot format (7-dimensional arrays)
            label_array = self.labels[idx]
            # Convert to tensor
            if isinstance(label_array, (list, tuple)):
                label_vector = torch.tensor(label_array, dtype=torch.float32)
            else:
                label_vector = torch.from_numpy(label_array).float()

            item['label'] = label_vector

        return item


def create_model(config: Config) -> BERTClassifier:
    """
    Create a new BERT classifier instance.

    Args:
        config: Config containing model hyperparameters

    Returns:
        Initialized BERTClassifier model
    """
    return BERTClassifier(config)


def create_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Create a tokenizer for the specified pretrained model.

    Args:
        model_name: Name or path of the pretrained model

    Returns:
        Initialized tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_trained_model(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None
) -> Tuple[BERTClassifier, PreTrainedTokenizer, Config]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint file
        device: Device to load model on (cuda/cpu). Auto-detected if None.

    Returns:
        Tuple of (model, tokenizer, model_config)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint is corrupted or incompatible
    """
    checkpoint_path = Path(checkpoint_path)

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    # Extract model config
    model_config = checkpoint.get('model_config')
    if model_config is None:
        raise RuntimeError("Checkpoint missing 'model_config' key")

    # Create and load model
    model = BERTClassifier(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Create tokenizer
    tokenizer = create_tokenizer(model_config.model_name)

    return model, tokenizer, model_config


def save_model_checkpoint(
    model: Union[BERTClassifier, nn.Module],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    config: Config,
    save_path: Union[str, Path]
) -> None:
    """
    Save model checkpoint to disk.

    Args:
        model: Model instance to save
        optimizer: Optimizer instance
        epoch: Current training epoch
        accuracy: Best accuracy achieved
        config: Model configuration
        save_path: Path where checkpoint will be saved
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract model state dict (handle DDP wrapper)
    model_state_dict = (
        model.module.state_dict() if hasattr(model, 'module')
        else model.state_dict()
    )

    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': accuracy,
        'model_config': config,  # Keep key name for backward compatibility
    }

    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")