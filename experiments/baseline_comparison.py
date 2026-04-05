#!/usr/bin/env python3
"""
SPRT vs Empirical Threshold Tuning Comparison

This experiment addresses the reviewer's question:
"Given that prior adjustments break Wald's theoretical bounds, what is the
concrete advantage of using SPRT over simply tuning a raw confidence threshold?"

Experiment Design:
1. Split data into train/val/test (60/20/20)
2. On validation set:
   - Find confidence threshold that achieves target FPR (e.g., 5%)
   - Record SPRT's empirical FPR with (α=0.05, β=0.10)
3. On test set:
   - Compare both methods' FPR, FNR, F1
   - Measure generalization gap (val → test)

Author: Qiu Jianmin, Han Jinguang
Date: March 2026
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

# Configuration
CONFIG = {
    "n_samples": 300,
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "target_fpr": 0.05,
    "alpha": 0.05,
    "beta": 0.10,
    "prior": 0.5,
    "t_min": 5,
    "max_length": 128,
}


def create_imbalanced_dataset(n_samples: int = 300, toxicity_rate: float = 0.08, seed: int = 42):
    """Create a dataset mimicking CivilComments (8% toxicity)"""
    np.random.seed(seed)

    texts, labels = [], []

    # Safe samples
    safe_templates = [
        "I think this product is excellent and would recommend it.",
        "The weather today is beautiful, perfect for a walk.",
        "Thank you for your help with this project.",
        "This article provides helpful information.",
        "I completely agree with your perspective.",
        "The service was outstanding and exceeded expectations.",
        "Great work on this presentation, very informative.",
        "I appreciate your thoughtful feedback on this matter.",
        "This is a well-designed solution to the problem.",
        "The team did an excellent job on this project.",
    ]

    # Toxic samples
    toxic_templates = [
        "This is absolutely disgusting and unacceptable behavior.",
        "People like you are the problem with society today.",
        "I hate everything about this stupid idea.",
        "This is the worst thing I have ever seen, garbage.",
        "You are completely wrong and your opinion is worthless.",
    ]

    n_toxic = int(n_samples * toxicity_rate)
    n_safe = n_samples - n_toxic

    for _ in range(n_safe):
        base = np.random.choice(safe_templates)
        ext = np.random.choice([" Furthermore, this is excellent.", " Additionally, very well done.", " Moreover, great work."])
        texts.append(base + ext)
        labels.append(0)

    for _ in range(n_toxic):
        base = np.random.choice(toxic_templates)
        ext = np.random.choice([" This is completely unacceptable.", " Really terrible behavior.", " Awful."])
        texts.append(base + ext)
        labels.append(1)

    # Shuffle
    indices = np.random.permutation(len(texts))
    return [texts[i] for i in indices], [labels[i] for i in indices]


def get_classifier_probs(model, tokenizer, texts, device, max_length=128):
    """Get classifier probabilities for a list of texts"""
    probs = []
    for text in tqdm(texts, desc="Getting probs"):
        inputs = tokenizer(
            text, return_tensors='pt',
            truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
        probs.append(prob)
    return np.array(probs)


def find_threshold_for_fpr(probs, labels, target_fpr):
    """Find confidence threshold that achieves target FPR on validation set"""
    safe_probs = probs[np.array(labels) == 0]

    # Sort safe sample probabilities
    sorted_probs = np.sort(safe_probs)

    # Find threshold such that (1 - percentile) of safe samples are above threshold
    # FPR = P(pred toxic | safe) = P(prob > threshold | safe)
    # To achieve target_fpr, we need threshold at (1 - target_fpr) percentile
    idx = int((1 - target_fpr) * len(sorted_probs))
    threshold = sorted_probs[min(idx, len(sorted_probs) - 1)]

    return threshold


class SPRTDetector:
    """Simple SPRT detector for comparison"""

    def __init__(self, model, tokenizer, alpha=0.05, beta=0.10,
                 prior=0.5, t_min=5, device="cuda", max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.prior = prior
        self.t_min = t_min
        self.device = device
        self.max_length = max_length

        # Thresholds
        self.A = np.log((1 - beta) / alpha) + np.log(prior / (1 - prior))
        self.B = np.log(beta / (1 - alpha)) + np.log(prior / (1 - prior))
        self.epsilon = 1e-10

    def predict(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = min(len(tokens), self.max_length)

        decision = "SAFE"
        stopping_time = n_tokens

        for t in range(self.t_min, n_tokens + 1):
            prefix = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
            inputs = self.tokenizer(
                prefix, return_tensors='pt',
                truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            S_t = np.log(prob / (1 - prob + self.epsilon))

            if S_t >= self.A:
                decision = "UNSAFE"
                stopping_time = t
                break
            elif S_t <= self.B:
                decision = "SAFE"
                stopping_time = t
                break

        return decision, stopping_time


class ConfidenceThresholdDetector:
    """Simple confidence threshold detector for comparison"""

    def __init__(self, threshold=0.5, t_min=5):
        self.threshold = threshold
        self.t_min = t_min

    def predict(self, text, model, tokenizer, device, max_length=128):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = min(len(tokens), max_length)

        # For early exit with confidence threshold, we check at each step
        # and exit when confidence exceeds threshold
        decision = "SAFE"
        stopping_time = n_tokens

        for t in range(self.t_min, n_tokens + 1):
            prefix = tokenizer.decode(tokens[:t], skip_special_tokens=True)
            inputs = tokenizer(
                prefix, return_tensors='pt',
                truncation=True, max_length=max_length
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            # Early exit if confident
            if prob >= self.threshold:
                decision = "UNSAFE"
                stopping_time = t
                break
            elif prob <= (1 - self.threshold):
                decision = "SAFE"
                stopping_time = t
                break

        return decision, stopping_time


def evaluate_predictions(y_true, y_pred):
    """Calculate metrics"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'fnr': fnr,
        'accuracy': accuracy_score(y_true, y_pred)
    }


def main():
    print("=" * 70)
    print("SPRT vs Empirical Threshold Tuning Comparison")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    model.eval()

    # Warmup
    print("Warming up...")
    dummy = tokenizer("Warmup", return_tensors='pt').to(device)
    for _ in range(5):
        with torch.no_grad():
            model(**dummy)

    # Create dataset
    print("\nCreating dataset...")
    texts, labels = create_imbalanced_dataset(
        n_samples=CONFIG['n_samples'],
        toxicity_rate=0.08,
        seed=42
    )

    # Split into train/val/test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.4, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    print(f"Val toxicity rate: {sum(val_labels)/len(val_labels)*100:.1f}%")
    print(f"Test toxicity rate: {sum(test_labels)/len(test_labels)*100:.1f}%")

    # Get probabilities on validation set
    print("\n--- Phase 1: Validation Set Tuning ---")
    val_probs = get_classifier_probs(model, tokenizer, val_texts, str(device))

    # Find threshold for target FPR
    target_fpr = CONFIG['target_fpr']
    emp_threshold = find_threshold_for_fpr(val_probs, val_labels, target_fpr)
    print(f"\nTarget FPR: {target_fpr:.2%}")
    print(f"Empirical threshold found: {emp_threshold:.4f}")

    # Evaluate threshold on validation set
    val_pred_threshold = (val_probs >= emp_threshold).astype(int)
    val_metrics_threshold = evaluate_predictions(val_labels, val_pred_threshold)
    print(f"Validation FPR (threshold): {val_metrics_threshold['fpr']:.2%}")

    # Evaluate SPRT on validation set
    print("\nEvaluating SPRT on validation set...")
    sprt = SPRTDetector(model, tokenizer, alpha=CONFIG['alpha'], beta=CONFIG['beta'],
                       prior=CONFIG['prior'], device=str(device))

    val_pred_sprt = []
    for text in tqdm(val_texts, desc="SPRT on val"):
        decision, _ = sprt.predict(text)
        val_pred_sprt.append(1 if decision == "UNSAFE" else 0)

    val_metrics_sprt = evaluate_predictions(val_labels, val_pred_sprt)
    print(f"Validation FPR (SPRT): {val_metrics_sprt['fpr']:.2%}")

    # Test set evaluation
    print("\n--- Phase 2: Test Set Evaluation ---")

    # Get test probabilities
    test_probs = get_classifier_probs(model, tokenizer, test_texts, str(device))

    # Method 1: Empirical threshold
    test_pred_threshold = (test_probs >= emp_threshold).astype(int)
    test_metrics_threshold = evaluate_predictions(test_labels, test_pred_threshold)

    # Method 2: SPRT
    print("\nEvaluating SPRT on test set...")
    test_pred_sprt = []
    test_stops_sprt = []
    for text in tqdm(test_texts, desc="SPRT on test"):
        decision, stop = sprt.predict(text)
        test_pred_sprt.append(1 if decision == "UNSAFE" else 0)
        test_stops_sprt.append(stop)

    test_metrics_sprt = evaluate_predictions(test_labels, test_pred_sprt)
    avg_stop_sprt = np.mean(test_stops_sprt)

    # Full-text baseline
    test_pred_full = (test_probs >= 0.5).astype(int)
    test_metrics_full = evaluate_predictions(test_labels, test_pred_full)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- Validation Set ---")
    print(f"{'Method':<30} {'FPR':>8} {'FNR':>8} {'F1':>8}")
    print(f"{'Empirical Threshold':<30} {val_metrics_threshold['fpr']:>8.2%} {val_metrics_threshold['fnr']:>8.2%} {val_metrics_threshold['f1']:>8.3f}")
    print(f"{'SPRT (α=0.05, β=0.10)':<30} {val_metrics_sprt['fpr']:>8.2%} {val_metrics_sprt['fnr']:>8.2%} {val_metrics_sprt['f1']:>8.3f}")

    print("\n--- Test Set ---")
    print(f"{'Method':<30} {'FPR':>8} {'FNR':>8} {'F1':>8} {'Lat.Red':>8}")
    print(f"{'Full-text Baseline':<30} {test_metrics_full['fpr']:>8.2%} {test_metrics_full['fnr']:>8.2%} {test_metrics_full['f1']:>8.3f} {'0%':>8}")
    print(f"{'Empirical Threshold':<30} {test_metrics_threshold['fpr']:>8.2%} {test_metrics_threshold['fnr']:>8.2%} {test_metrics_threshold['f1']:>8.3f} {'N/A':>8}")
    print(f"{'SPRT (α=0.05, β=0.10)':<30} {test_metrics_sprt['fpr']:>8.2%} {test_metrics_sprt['fnr']:>8.2%} {test_metrics_sprt['f1']:>8.3f} {f'{(1-avg_stop_sprt/128)*100:.1f}%':>8}")

    # Generalization gap
    print("\n--- Generalization Gap (Val → Test) ---")
    fpr_gap_threshold = test_metrics_threshold['fpr'] - val_metrics_threshold['fpr']
    fpr_gap_sprt = test_metrics_sprt['fpr'] - val_metrics_sprt['fpr']

    print(f"Empirical Threshold FPR gap: {fpr_gap_threshold:+.2%}")
    print(f"SPRT FPR gap: {fpr_gap_sprt:+.2%}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'validation': {
            'empirical_threshold': emp_threshold,
            'threshold_metrics': val_metrics_threshold,
            'sprt_metrics': val_metrics_sprt,
        },
        'test': {
            'full_text_metrics': test_metrics_full,
            'threshold_metrics': test_metrics_threshold,
            'sprt_metrics': test_metrics_sprt,
            'sprt_avg_stopping_time': avg_stop_sprt,
        },
        'generalization_gap': {
            'threshold_fpr_gap': fpr_gap_threshold,
            'sprt_fpr_gap': fpr_gap_sprt,
        }
    }

    with open('./baseline_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to baseline_comparison_results.json")

    return results


if __name__ == "__main__":
    main()
