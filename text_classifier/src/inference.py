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

    def predict_single(self, text: str) -> Dict[str, Union[str, float, List[float]]]:
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
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'text': text,
            'predicted_class': predicted_class,
            'predicted_label': self.label_map.get(predicted_class, f"class_{predicted_class}"),
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
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

        # Add predictions to dataframe
        df['predicted_class'] = [r['predicted_class'] for r in results]
        df['predicted_label'] = [r['predicted_label'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]

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

        # Handle labels - convert to int64, handling various formats
        label_series = df[label_column]
        if isinstance(label_series.iloc[0], (list, np.ndarray)):
            true_labels = np.array([int(label[0]) for label in label_series], dtype=np.int64)
        else:
            true_labels = label_series.astype(np.int64).to_numpy()

        print(f"\nEvaluating on {len(texts)} test samples...")
        print(f"Label distribution: {np.bincount(true_labels)}")

        # Create test dataloader for efficient batch processing
        labels_list = true_labels.tolist()
        dataset = TextClassificationDataset(
            texts, labels_list, self.tokenizer, self.model_config.max_length
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
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

                # Get predictions
                predictions = torch.argmax(outputs, dim=1)

                # Accumulate results
                total_loss += loss.item() * input_ids.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

        predictions = np.array(all_predictions)
        true_labels = np.array(all_true_labels)
        avg_loss = total_loss / len(dataset)

        # Compute metrics
        average = 'binary' if self.model_config.num_classes == 2 else 'macro'

        metrics = {
            'test_loss': avg_loss,
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average=average, zero_division=0),
            'recall': recall_score(true_labels, predictions, average=average, zero_division=0),
            'f1_score': f1_score(true_labels, predictions, average=average, zero_division=0),
        }

        # Print results
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Test Loss:      {metrics['test_loss']:.4f}")
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1 Score:       {metrics['f1_score']:.4f}")

        # Per-class metrics
        print("\nPer-Class Metrics:")
        print("-"*70)
        class_names = [f"Class {i}" for i in range(self.model_config.num_classes)]
        report = classification_report(
            true_labels, predictions, target_names=class_names, zero_division=0
        )
        print(report)

        # Confusion matrix
        print("\nConfusion Matrix:")
        print("-"*70)
        cm = confusion_matrix(true_labels, predictions)

        # Print header
        header = "True\\Pred  " + "  ".join([f"Class {i:2d}" for i in range(self.model_config.num_classes)])
        print(header)
        print("-" * len(header))

        # Print matrix rows
        for i in range(self.model_config.num_classes):
            row = f"Class {i:2d}   " + "  ".join([f"{cm[i, j]:8d}" for j in range(self.model_config.num_classes)])
            print(row)

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