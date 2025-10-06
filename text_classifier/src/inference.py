import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from model import TextClassificationDataset, load_trained_model


class TextClassifierInference:
    def __init__(self, model_path: str, device: str = None):
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.label_map = None  # Will be set after loading model

        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        # Use the load_trained_model utility
        self.model, self.tokenizer, self.model_config = load_trained_model(
            self.model_path, self.device
        )

        # Set hard-coded label map for the project
        if self.label_map is None:
            self.label_map = {
                1: "antitrust",
                2: "civil_rights",
                3: "crime",
                4: "govt_regulation",
                5: "labor_movement",
                6: "politics",
                7: "protests"
            }

        print(f"Model loaded successfully from {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Number of classes: {self.model_config.num_classes}")
        print(f"Label map: {self.label_map}")

    def predict_single(self, text: str, threshold: float = 0.5) -> Dict[str, Union[str, List, float]]:
        """
        Predict labels for a single text (multi-label classification).

        Args:
            text: Input text to classify
            threshold: Probability threshold for positive class (default: 0.5)

        Returns:
            Dictionary with predicted classes, labels, and probabilities
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.model_config.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # For multi-label classification, apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs)
            # Get predicted classes where probability > threshold
            # Model outputs 7 values (one per category)
            predicted_indices = (probabilities[0] > threshold).nonzero(as_tuple=True)[0].cpu().numpy().tolist()

            # Map indices to category names
            category_names = ['antitrust', 'civil_rights', 'crime', 'govt_regulation',
                            'labor_movement', 'politics', 'protests']
            predicted_labels = [category_names[idx] for idx in predicted_indices]

        return {
            'text': text,
            'predicted_indices': predicted_indices,
            'predicted_labels': predicted_labels,
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'threshold': threshold
        }

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Union[str, float, List[float]]]]:
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []

            for text in batch_texts:
                result = self.predict_single(text)
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def predict_from_file(self, input_file: str, output_file: str = None, text_column: str = "text"):
        # Auto-detect file format
        input_path = Path(input_file)
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)

        texts = df[text_column].tolist()

        print(f"Processing {len(texts)} texts from {input_file}")
        results = self.predict_batch(texts)

        # Add predictions to dataframe (multi-label format)
        df['predicted_indices'] = [r['predicted_indices'] for r in results]
        df['predicted_labels'] = [r['predicted_labels'] for r in results]
        df['probabilities'] = [r['probabilities'] for r in results]

        # Save results
        if output_file:
            output_path = Path(output_file)
            if output_path.suffix == '.parquet':
                df.to_parquet(output_file, index=False)
            else:
                df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        return df

    def set_label_map(self, label_map: Dict[int, str]):
        self.label_map = label_map

    def evaluate_on_test_data(
        self,
        test_file: str,
        text_column: str = "text",
        label_column: str = "label"
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset and compute metrics.

        Args:
            test_file: Path to test data file (parquet or csv)
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Dictionary containing test metrics
        """
        # Load test data
        test_path = Path(test_file)
        if test_path.suffix == '.parquet':
            df = pd.read_parquet(test_file)
        else:
            df = pd.read_csv(test_file)

        texts = df[text_column].astype(str).tolist()

        # Handle labels - already in one-hot encoded format (7-dimensional arrays)
        label_series = df[label_column]
        # Labels are already one-hot encoded
        true_labels = np.array([np.array(label, dtype=np.float32) for label in label_series])

        print(f"\nEvaluating on {len(texts)} test samples...")
        print(f"Label shape: {true_labels.shape}")

        # Create test dataloader for efficient batch processing
        labels_list = true_labels.tolist()
        dataset = TextClassificationDataset(
            texts, labels_list, self.tokenizer, self.model_config.max_length, self.model_config.num_classes
        )

        test_loader = DataLoader(
            dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=4 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0,
            drop_last=False
        )

        # Evaluate
        self.model.eval()
        all_predictions = []
        all_true_labels = []
        total_loss = 0.0
        criterion = torch.nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

                # Get predictions for multi-label classification
                predictions = (torch.sigmoid(outputs) > 0.5).float()

                # Accumulate results
                total_loss += loss.item() * input_ids.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

        predictions = np.array(all_predictions)
        true_labels = np.array(all_true_labels)
        avg_loss = total_loss / len(dataset)

        # Compute metrics for multi-label classification
        # Use 'samples' average which computes metrics for each sample
        metrics = {
            'test_loss': avg_loss,
            'exact_match_accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='samples', zero_division=0),
            'recall': recall_score(true_labels, predictions, average='samples', zero_division=0),
            'f1_score': f1_score(true_labels, predictions, average='samples', zero_division=0),
        }

        # Print results
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Test Loss:              {metrics['test_loss']:.4f}")
        print(f"Exact Match Accuracy:   {metrics['exact_match_accuracy']:.4f}")
        print(f"Precision (samples):    {metrics['precision']:.4f}")
        print(f"Recall (samples):       {metrics['recall']:.4f}")
        print(f"F1 Score (samples):     {metrics['f1_score']:.4f}")

        # Per-class metrics for multi-label classification
        print("\nPer-Class Metrics (binary for each label):")
        print("-"*70)
        category_names = ['antitrust', 'civil_rights', 'crime', 'govt_regulation',
                         'labor_movement', 'politics', 'protests']

        for i in range(self.model_config.num_classes):
            class_true = true_labels[:, i]
            class_pred = predictions[:, i]

            # Compute metrics for this class
            class_precision = precision_score(class_true, class_pred, zero_division=0)
            class_recall = recall_score(class_true, class_pred, zero_division=0)
            class_f1 = f1_score(class_true, class_pred, zero_division=0)
            support = int(class_true.sum())

            class_name = category_names[i]
            print(f"{class_name:20s} - Precision: {class_precision:.4f}, Recall: {class_recall:.4f}, F1: {class_f1:.4f}, Support: {support}")

        print("="*70)

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Text Classification Inference")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pt",
                       help="Path to the trained model checkpoint")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--input_file", type=str, help="CSV file with texts to classify")
    parser.add_argument("--output_file", type=str, help="Output CSV file for batch predictions")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Column name for text in input file")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--test", action="store_true",
                       help="Run evaluation on test dataset")
    parser.add_argument("--test_file", type=str, default="./data/test_data.parquet",
                       help="Path to test data file")
    parser.add_argument("--label_column", type=str, default="label",
                       help="Column name for labels in test file")

    args = parser.parse_args()

    # Initialize classifier
    classifier = TextClassifierInference(args.model_path, args.device)

    # Single text prediction
    if args.text:
        result = classifier.predict_single(args.text)
        print(json.dumps(result, indent=2))

    # Batch prediction from file
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} not found")
            return

        df = classifier.predict_from_file(
            args.input_file,
            args.output_file,
            args.text_column
        )

        # Print summary
        print(f"\nPrediction Summary:")
        print(df['predicted_label'].value_counts())
        print(f"Average confidence: {df['confidence'].mean():.4f}")

    # Test evaluation
    elif args.test:
        if not os.path.exists(args.test_file):
            print(f"Error: Test file {args.test_file} not found")
            return

        classifier.evaluate_on_test_data(
            args.test_file,
            args.text_column,
            args.label_column
        )

    else:
        print("Please specify --text, --input_file, or --test mode")
        print("Use --help for more options")


if __name__ == "__main__":
    main()