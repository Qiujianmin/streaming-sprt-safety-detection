#!/usr/bin/env python3
"""
Comprehensive Experiments for SPRT Streaming Safety Detection
Addresses all reviewer concerns for ACM TIST Major Revision

Experiments:
1. Multiple baseline comparisons (Fixed-Window, Confidence-based, etc.)
2. Multi-dataset validation (CivilComments, BeaverTails, PKU-SafeRLHF, Qwen3GuardTest)
3. Calibration ablation (Temperature Scaling, Isotonic Regression)
4. Jailbreak/Late-toxicity testing (using Qwen3GuardTest position annotations)

Author: Qiu Jianmin, Han Jinguang
Date: March 2026
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar
from datasets import load_dataset, load_from_disk

# ============================================================
# Configuration
# ============================================================

EXPERIMENT_CONFIG = {
    "alpha": 0.05,
    "beta": 0.10,
    "t_min": 5,
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 3,
    "priors": [0.50, 0.10, 0.05, 0.01],
    "window_sizes": [5, 10, 20],
    "confidence_thresholds": [0.9, 0.95],
}

# ============================================================
# 1. Data Loading
# ============================================================

class SafetyDataset(Dataset):
    """Unified safety dataset wrapper"""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_civilcomments(train_size=20000, val_size=2000, test_size=5000):
    """Load CivilComments dataset"""
    print("Loading CivilComments...")
    train_data = load_dataset("google/civil_comments", split=f"train[:{train_size}]")
    val_data = load_dataset("google/civil_comments", split=f"validation[:{val_size}]")
    test_data = load_dataset("google/civil_comments", split=f"test[:{test_size}]")

    def preprocess(examples):
        texts, labels = [], []
        for text, toxicity in zip(examples['text'], examples['toxicity']):
            if text and text.strip():
                texts.append(text.strip())
                labels.append(1 if (toxicity or 0) >= 0.5 else 0)
        return {'text': texts, 'label': labels}

    train = train_data.map(lambda x: preprocess(x), batched=True, remove_columns=train_data.column_names)
    val = val_data.map(lambda x: preprocess(x), batched=True, remove_columns=val_data.column_names)
    test = test_data.map(lambda x: preprocess(x), batched=True, remove_columns=test_data.column_names)

    return train, val, test, "CivilComments"


def load_beavertails(data_path, test_size=5000):
    """Load BeaverTails dataset"""
    print(f"Loading BeaverTails...")
    # Try loading from HuggingFace first
    try:
        dataset = load_dataset("PKU-Alignment/BeaverTails")
        # Check available keys
        keys = list(dataset.keys())
        print(f"HuggingFace keys: {keys}")
        if 'train' in dataset:
            train_data = dataset['train']
            test_data = dataset['test']
        elif '30k_train' in dataset:
            train_data = dataset['30k_train']
            test_data = dataset['30k_test']
        else:
            # Use first key for train, last for test
            train_data = dataset[keys[0]]
            test_data = dataset[keys[-1]]
        print(f"Loaded from HuggingFace: train={len(train_data)}, test={len(test_data)}")
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        # Try local path as fallback
        try:
            dataset = load_from_disk(os.path.join(data_path, "beavertails", "data"))
            train_data = dataset['30k_train']
            test_data = dataset['30k_test']
            print(f"Loaded from local: train={len(train_data)}, test={len(test_data)}")
        except Exception as e2:
            print(f"Failed to load from local: {e2}")
            raise

    # Extract text and labels
    # is_safe: True = safe, False = unsafe
    train_texts = [f"{item['prompt']} {item['response']}" for item in train_data]
    train_labels = [0 if item['is_safe'] else 1 for item in train_data]  # 0=safe, 1=unsafe

    test_texts = [f"{item['prompt']} {item['response']}" for item in test_data]
    test_labels = [0 if item['is_safe'] else 1 for item in test_data]

    # Split train into train/val
    val_size = min(2000, len(train_texts) // 5)
    val_texts = train_texts[:val_size]
    val_labels = train_labels[:val_size]
    train_texts = train_texts[val_size:]
    train_labels = train_labels[val_size:]

    # Limit test size
    test_texts = test_texts[:test_size]
    test_labels = test_labels[:test_size]

    train = {'text': train_texts, 'label': train_labels}
    val = {'text': val_texts, 'label': val_labels}
    test = {'text': test_texts, 'label': test_labels}

    return train, val, test, "BeaverTails"


def load_pku_saferlhf(data_path, test_size=3000):
    """Load PKU-SafeRLHF dataset"""
    print(f"Loading PKU-SafeRLHF from {data_path}...")
    dataset = load_from_disk(os.path.join(data_path, "pku_saferlhf_30k", "data"))

    train_data = dataset['train']
    test_data = dataset['test']

    # Extract safe and unsafe responses
    texts = []
    labels = []

    for item in train_data:
        # Response 0
        texts.append(item['response_0'])
        labels.append(0 if item['is_response_0_safe'] else 1)
        # Response 1
        texts.append(item['response_1'])
        labels.append(0 if item['is_response_1_safe'] else 1)

    # Split into train/val/test
    val_size = min(2000, len(texts) // 5)
    test_size_actual = min(test_size, len(texts) // 5)

    val = {'text': texts[:val_size], 'label': labels[:val_size]}
    test = {'text': texts[val_size:val_size+test_size_actual], 'label': labels[val_size:val_size+test_size_actual]}
    train = {'text': texts[val_size+test_size_actual:], 'label': labels[val_size+test_size_actual:]}

    return train, val, test, "PKU-SafeRLHF"


def load_qwen_guardtest(data_path):
    """Load Qwen3GuardTest dataset - has position annotations for late-toxicity testing"""
    print(f"Loading Qwen3GuardTest from HuggingFace...")
    # Load directly from HuggingFace to avoid compatibility issues
    try:
        dataset = load_dataset("Qwen/Qwen3GuardTest")
        loc_data = dataset['response_loc']
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}, trying local path...")
        dataset = load_from_disk(os.path.join(data_path, "qwen3guardtest", "data"))
        loc_data = dataset['response_loc']

    # response_loc has unsafe_start_index and unsafe_end_index
    loc_data = dataset['response_loc']

    texts = []
    labels = []
    unsafe_positions = []  # (start, end) for late-toxicity analysis

    for item in loc_data:
        # Extract assistant response
        message = item['message']
        for msg in message:
            if msg['role'] == 'assistant':
                texts.append(msg['content'])
                labels.append(1 if item['label'].lower() == 'unsafe' else 0)
                # Store position info
                unsafe_positions.append((item.get('unsafe_start_index', -1),
                                        item.get('unsafe_end_index', -1)))
                break

    # Split
    val_size = min(200, len(texts) // 5)
    val = {'text': texts[:val_size], 'label': labels[:val_size]}
    test = {'text': texts[val_size:], 'label': labels[val_size:],
            'unsafe_positions': unsafe_positions[val_size:]}
    train = {'text': texts[val_size:val_size*2], 'label': labels[val_size:val_size*2]}  # Small train for fine-tuning

    return train, val, test, "Qwen3GuardTest"


# ============================================================
# 2. Calibration Methods
# ============================================================

class TemperatureScaling(nn.Module):
    """Temperature Scaling for calibration"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits, labels):
        """Find optimal temperature using NLL minimization"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature.item()


class IsotonicCalibrator:
    """Isotonic Regression for calibration"""
    def __init__(self):
        self.regressor = IsotonicRegression(out_of_bounds='clip')

    def fit(self, probs, labels):
        """Fit isotonic regression on probabilities"""
        self.regressor.fit(probs, labels)
        return self

    def calibrate(self, probs):
        """Apply calibration"""
        return self.regressor.transform(probs)


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(labels[in_bin])
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return ece


# ============================================================
# 3. Detection Methods (Baselines + SPRT)
# ============================================================

class FullTextClassifier:
    """Standard full-text classification baseline"""
    def __init__(self, model, tokenizer, device='cuda', max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def predict_batch(self, texts, labels=None):
        decisions = []
        probs = []

        for text in tqdm(texts, desc="Full-text"):
            inputs = self.tokenizer(
                text, return_tensors='pt',
                truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            decisions.append("UNSAFE" if prob >= 0.5 else "SAFE")
            probs.append(prob)

        results = {
            'predictions': decisions,
            'probabilities': probs,
            'avg_stopping_time': self.max_length,  # Full text
            'method': 'Full-text'
        }

        if labels is not None:
            results.update(self._compute_metrics(decisions, labels))

        return results

    def _compute_metrics(self, decisions, labels):
        y_pred = [1 if d == "UNSAFE" else 0 for d in decisions]
        y_true = labels

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'fpr': fpr, 'fnr': fnr,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }


class SPRTDetector:
    """SPRT streaming detector with prior adjustment"""
    def __init__(self, model, tokenizer, alpha=0.05, beta=0.10, t_min=5,
                 prior_toxicity=0.50, device='cuda', max_length=128,
                 one_sided=False, calibrator=None):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.t_min = t_min
        self.prior_toxicity = prior_toxicity
        self.device = device
        self.max_length = max_length
        self.one_sided = one_sided  # Only detect UNSAFE early
        self.calibrator = calibrator
        self.epsilon = 1e-10

        # Wald thresholds
        self.A = np.log((1 - beta) / alpha)
        self.B = np.log(beta / (1 - alpha))

        # Prior adjustment
        if prior_toxicity != 0.50:
            prior_log_odds = np.log(prior_toxicity / (1 - prior_toxicity))
            self.A += prior_log_odds
            self.B += prior_log_odds

    def predict_batch(self, texts, labels=None, return_trajectories=False):
        decisions = []
        stopping_times = []
        trajectories = [] if return_trajectories else None

        for text in tqdm(texts, desc=f"SPRT(π={self.prior_toxicity})"):
            decision, stop_time, traj = self._predict_single(text, return_trajectories)
            decisions.append(decision)
            stopping_times.append(stop_time)
            if return_trajectories:
                trajectories.append(traj)

        results = {
            'predictions': decisions,
            'stopping_times': stopping_times,
            'avg_stopping_time': np.mean(stopping_times),
            'median_stopping_time': np.median(stopping_times),
            'method': f'SPRT(π={self.prior_toxicity})'
        }

        if labels is not None:
            results.update(self._compute_metrics(decisions, labels))

        if return_trajectories:
            results['trajectories'] = trajectories

        return results

    def _predict_single(self, text, return_trajectory=False):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = min(len(tokens), self.max_length)

        decision = "SAFE"
        stop_time = n_tokens
        trajectory = [] if return_trajectory else None

        for t in range(self.t_min, n_tokens + 1):
            current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
            inputs = self.tokenizer(
                current_text, return_tensors='pt',
                truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            # Apply calibration if available
            if self.calibrator is not None:
                prob = self.calibrator.calibrate(np.array([prob]))[0]

            S_t = np.log(prob / (1 - prob + self.epsilon))

            if return_trajectory:
                trajectory.append({'t': t, 'p': prob, 'S': S_t})

            if S_t >= self.A:
                decision = "UNSAFE"
                stop_time = t
                break
            elif not self.one_sided and S_t <= self.B:
                # Only stop for SAFE if not one-sided
                decision = "SAFE"
                stop_time = t
                break

        return decision, stop_time, trajectory

    def _compute_metrics(self, decisions, labels):
        y_pred = [1 if d == "UNSAFE" else 0 for d in decisions]
        y_true = labels

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'fpr': fpr, 'fnr': fnr,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }


class FixedWindowDetector:
    """Fixed-window baseline: check at regular intervals"""
    def __init__(self, model, tokenizer, window_size=10, threshold=0.5,
                 device='cuda', max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.threshold = threshold
        self.device = device
        self.max_length = max_length

    def predict_batch(self, texts, labels=None):
        decisions = []
        stopping_times = []
        check_counts = []

        for text in tqdm(texts, desc=f"Fixed-W{self.window_size}"):
            decision, stop_time, checks = self._predict_single(text)
            decisions.append(decision)
            stopping_times.append(stop_time)
            check_counts.append(checks)

        results = {
            'predictions': decisions,
            'stopping_times': stopping_times,
            'avg_stopping_time': np.mean(stopping_times),
            'avg_checks': np.mean(check_counts),
            'method': f'Fixed-W{self.window_size}'
        }

        if labels is not None:
            results.update(self._compute_metrics(decisions, labels))

        return results

    def _predict_single(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = min(len(tokens), self.max_length)

        decision = "SAFE"
        stop_time = n_tokens
        checks = 0

        # Check at window intervals
        for t in range(self.window_size, n_tokens + 1, self.window_size):
            checks += 1
            current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
            inputs = self.tokenizer(
                current_text, return_tensors='pt',
                truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            if prob >= self.threshold:
                decision = "UNSAFE"
                stop_time = t
                break

        return decision, stop_time, checks

    def _compute_metrics(self, decisions, labels):
        y_pred = [1 if d == "UNSAFE" else 0 for d in decisions]
        y_true = labels

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'fpr': fpr, 'fnr': fnr,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }


class ConfidenceBasedDetector:
    """Confidence-based early stopping: stop when confidence exceeds threshold"""
    def __init__(self, model, tokenizer, confidence_threshold=0.9, t_min=5,
                 device='cuda', max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold
        self.t_min = t_min
        self.device = device
        self.max_length = max_length

    def predict_batch(self, texts, labels=None):
        decisions = []
        stopping_times = []

        for text in tqdm(texts, desc=f"Conf-{self.confidence_threshold}"):
            decision, stop_time = self._predict_single(text)
            decisions.append(decision)
            stopping_times.append(stop_time)

        results = {
            'predictions': decisions,
            'stopping_times': stopping_times,
            'avg_stopping_time': np.mean(stopping_times),
            'method': f'Confidence(th={self.confidence_threshold})'
        }

        if labels is not None:
            results.update(self._compute_metrics(decisions, labels))

        return results

    def _predict_single(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = min(len(tokens), self.max_length)

        decision = "SAFE"
        stop_time = n_tokens

        for t in range(self.t_min, n_tokens + 1):
            current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
            inputs = self.tokenizer(
                current_text, return_tensors='pt',
                truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            # Check confidence (distance from 0.5)
            if prob >= self.confidence_threshold:
                decision = "UNSAFE"
                stop_time = t
                break
            elif prob <= (1 - self.confidence_threshold):
                decision = "SAFE"
                stop_time = t
                break

        return decision, stop_time

    def _compute_metrics(self, decisions, labels):
        y_pred = [1 if d == "UNSAFE" else 0 for d in decisions]
        y_true = labels

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'fpr': fpr, 'fnr': fnr,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }


class CumulativeLogitDetector:
    """INCORRECT method: sum logit over time (double-counting) - for comparison"""
    def __init__(self, model, tokenizer, alpha=0.05, beta=0.10, t_min=5,
                 device='cuda', max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.t_min = t_min
        self.device = device
        self.max_length = max_length
        self.epsilon = 1e-10

        # Standard Wald thresholds
        self.A = np.log((1 - beta) / alpha)
        self.B = np.log(beta / (1 - alpha))

    def predict_batch(self, texts, labels=None):
        decisions = []
        stopping_times = []

        for text in tqdm(texts, desc="Cumulative-Logit(WRONG)"):
            decision, stop_time = self._predict_single(text)
            decisions.append(decision)
            stopping_times.append(stop_time)

        results = {
            'predictions': decisions,
            'stopping_times': stopping_times,
            'avg_stopping_time': np.mean(stopping_times),
            'method': 'Cumulative-Logit(WRONG)'
        }

        if labels is not None:
            results.update(self._compute_metrics(decisions, labels))

        return results

    def _predict_single(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = min(len(tokens), self.max_length)

        decision = "SAFE"
        stop_time = n_tokens
        cumulative_S = 0.0  # WRONG: accumulate over time

        for t in range(1, n_tokens + 1):
            current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
            inputs = self.tokenizer(
                current_text, return_tensors='pt',
                truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            # WRONG: Add to cumulative sum (double-counting!)
            S_t = np.log(prob / (1 - prob + self.epsilon))
            cumulative_S += S_t

            if t >= self.t_min:
                if cumulative_S >= self.A:
                    decision = "UNSAFE"
                    stop_time = t
                    break
                elif cumulative_S <= self.B:
                    decision = "SAFE"
                    stop_time = t
                    break

        return decision, stop_time

    def _compute_metrics(self, decisions, labels):
        y_pred = [1 if d == "UNSAFE" else 0 for d in decisions]
        y_true = labels

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'fpr': fpr, 'fnr': fnr,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }


# ============================================================
# 4. Late-Toxicity Analysis (Jailbreak Testing)
# ============================================================

def analyze_late_toxicity(detector, texts, labels, unsafe_positions, tokenizer):
    """Analyze performance on late-toxicity samples (toxicity appears late in sequence)"""
    results = {
        'early_toxic': {'count': 0, 'detected': 0, 'avg_stop': []},
        'late_toxic': {'count': 0, 'detected': 0, 'avg_stop': []},
        'full_safe': {'count': 0, 'false_positive': 0, 'avg_stop': []},
    }

    for text, label, (start, end) in zip(texts, labels, unsafe_positions):
        if start < 0:  # No position info
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)
        toxicity_position = start  # Normalized position

        decision, stop_time, _ = detector._predict_single(text, return_trajectory=False)

        if label == 1:  # Toxic
            relative_pos = toxicity_position / n_tokens if n_tokens > 0 else 0

            if relative_pos < 0.3:  # Early toxicity
                results['early_toxic']['count'] += 1
                if decision == "UNSAFE":
                    results['early_toxic']['detected'] += 1
                results['early_toxic']['avg_stop'].append(stop_time)
            else:  # Late toxicity
                results['late_toxic']['count'] += 1
                if decision == "UNSAFE":
                    results['late_toxic']['detected'] += 1
                results['late_toxic']['avg_stop'].append(stop_time)
        else:  # Safe
            results['full_safe']['count'] += 1
            if decision == "UNSAFE":
                results['full_safe']['false_positive'] += 1
            results['full_safe']['avg_stop'].append(stop_time)

    # Compute averages
    for key in results:
        if results[key]['avg_stop']:
            results[key]['avg_stop'] = np.mean(results[key]['avg_stop'])
        else:
            results[key]['avg_stop'] = 0

    # Compute detection rates
    if results['early_toxic']['count'] > 0:
        results['early_toxic']['detection_rate'] = results['early_toxic']['detected'] / results['early_toxic']['count']
    if results['late_toxic']['count'] > 0:
        results['late_toxic']['detection_rate'] = results['late_toxic']['detected'] / results['late_toxic']['count']
    if results['full_safe']['count'] > 0:
        results['full_safe']['fpr'] = results['full_safe']['false_positive'] / results['full_safe']['count']

    return results


# ============================================================
# 5. Main Experiment Runner
# ============================================================

def run_comprehensive_experiments(
    dataset_name='civilcomments',
    data_path=None,
    output_dir='./results',
    device='cuda',
    config=None
):
    """Run all experiments for a given dataset"""

    config = config or EXPERIMENT_CONFIG
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Comprehensive SPRT Experiments - {dataset_name}")
    print("=" * 70)

    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    if dataset_name == 'civilcomments':
        train, val, test, ds_name = load_civilcomments()
    elif dataset_name == 'beavertails':
        train, val, test, ds_name = load_beavertails(data_path)
    elif dataset_name == 'pku_saferlhf':
        train, val, test, ds_name = load_pku_saferlhf(data_path)
    elif dataset_name == 'qwen3guardtest':
        train, val, test, ds_name = load_qwen_guardtest(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nDataset: {ds_name}")
    print(f"Train: {len(train['text'])}, Val: {len(val['text'])}, Test: {len(test['text'])}")
    print(f"Test toxicity rate: {sum(test['label'])/len(test['label'])*100:.1f}%")

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)

    # Create datasets
    train_dataset = SafetyDataset(train['text'], train['label'], tokenizer, config['max_length'])
    val_dataset = SafetyDataset(val['text'], val['label'], tokenizer, config['max_length'])
    test_dataset = SafetyDataset(test['text'], test['label'], tokenizer, config['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # Train model
    print("\n" + "=" * 70)
    print("Training Classifier")
    print("=" * 70)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    training_history = []

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())

        # Validation
        model.eval()
        val_preds, val_true = [], []
        val_logits = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels']

                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]

                val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                val_true.extend(labels.numpy())
                val_logits.extend(logits.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(val_true, val_preds, average='binary', zero_division=0)

        print(f"Epoch {epoch+1}: Loss={np.mean(train_loss):.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': np.mean(train_loss),
            'val_acc': val_acc,
            'val_f1': val_f1
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))

    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    model.eval()

    # ============================================================
    # Calibration
    # ============================================================
    print("\n" + "=" * 70)
    print("Calibration")
    print("=" * 70)

    # Get validation logits for calibration
    val_logits_tensor = torch.tensor(val_logits).to(device)
    val_labels_tensor = torch.tensor(val_true).to(device)
    val_probs_arr = np.array(val_probs)
    val_labels_arr = np.array(val_true)

    # Compute ECE before calibration
    ece_before = compute_ece(val_probs_arr, val_labels_arr)
    print(f"ECE before calibration: {ece_before:.4f}")

    # Temperature Scaling
    temp_scaling = TemperatureScaling().to(device)
    temp_scaling.calibrate(val_logits_tensor, val_labels_tensor)
    print(f"Learned temperature: {temp_scaling.temperature.item():.4f}")

    # Compute ECE after temperature scaling
    with torch.no_grad():
        calibrated_logits = temp_scaling(val_logits_tensor)
        calibrated_probs = torch.softmax(calibrated_logits, dim=-1)[:, 1].cpu().numpy()
    ece_temp = compute_ece(calibrated_probs, val_labels_arr)
    print(f"ECE after Temperature Scaling: {ece_temp:.4f}")

    # Isotonic Regression
    isotonic = IsotonicCalibrator()
    isotonic.fit(val_probs_arr, val_labels_arr)
    isotonic_probs = isotonic.calibrate(val_probs_arr)
    ece_isotonic = compute_ece(isotonic_probs, val_labels_arr)
    print(f"ECE after Isotonic Regression: {ece_isotonic:.4f}")

    # ============================================================
    # Run All Methods
    # ============================================================
    print("\n" + "=" * 70)
    print("Running All Detection Methods")
    print("=" * 70)

    test_texts = test['text']
    test_labels = test['label']

    all_results = {}

    # 1. Full-text baseline
    print("\n[1/6] Full-text baseline...")
    full_text = FullTextClassifier(model, tokenizer, device, config['max_length'])
    all_results['Full-text'] = full_text.predict_batch(test_texts, test_labels)

    # 2. SPRT with different priors
    print("\n[2/6] SPRT with different priors...")
    for prior in config['priors']:
        sprt = SPRTDetector(
            model, tokenizer,
            alpha=config['alpha'], beta=config['beta'],
            t_min=config['t_min'], prior_toxicity=prior,
            device=device, max_length=config['max_length']
        )
        all_results[f'SPRT(π={prior})'] = sprt.predict_batch(test_texts, test_labels)

    # 3. Fixed-Window baselines
    print("\n[3/6] Fixed-Window baselines...")
    for w in config['window_sizes']:
        fw = FixedWindowDetector(
            model, tokenizer, window_size=w,
            device=device, max_length=config['max_length']
        )
        all_results[f'Fixed-W{w}'] = fw.predict_batch(test_texts, test_labels)

    # 4. Confidence-based
    print("\n[4/6] Confidence-based baselines...")
    for th in config['confidence_thresholds']:
        conf = ConfidenceBasedDetector(
            model, tokenizer, confidence_threshold=th,
            t_min=config['t_min'], device=device, max_length=config['max_length']
        )
        all_results[f'Confidence(th={th})'] = conf.predict_batch(test_texts, test_labels)

    # 5. Cumulative-Logit (WRONG - for comparison)
    print("\n[5/6] Cumulative-Logit (incorrect method)...")
    cumlogit = CumulativeLogitDetector(
        model, tokenizer,
        alpha=config['alpha'], beta=config['beta'],
        t_min=config['t_min'], device=device, max_length=config['max_length']
    )
    all_results['Cumulative-Logit(WRONG)'] = cumlogit.predict_batch(test_texts, test_labels)

    # 6. SPRT with calibration
    print("\n[6/6] SPRT with calibration...")
    # Temperature-scaled SPRT
    sprt_temp = SPRTDetector(
        model, tokenizer,
        alpha=config['alpha'], beta=config['beta'],
        t_min=config['t_min'], prior_toxicity=0.01,
        device=device, max_length=config['max_length'],
        calibrator=isotonic  # Use isotonic as it's better
    )
    all_results['SPRT(π=0.01, calibrated)'] = sprt_temp.predict_batch(test_texts, test_labels)

    # 7. One-sided SPRT (only detect UNSAFE early) for jailbreak testing
    sprt_one_sided = SPRTDetector(
        model, tokenizer,
        alpha=config['alpha'], beta=config['beta'],
        t_min=config['t_min'], prior_toxicity=0.01,
        device=device, max_length=config['max_length'],
        one_sided=True
    )
    all_results['SPRT(π=0.01, one-sided)'] = sprt_one_sided.predict_batch(test_texts, test_labels)

    # ============================================================
    # Late-Toxicity Analysis (if position info available)
    # ============================================================
    late_toxicity_results = None
    if 'unsafe_positions' in test:
        print("\n" + "=" * 70)
        print("Late-Toxicity Analysis (Jailbreak Testing)")
        print("=" * 70)

        sprt_detector = SPRTDetector(
            model, tokenizer,
            alpha=config['alpha'], beta=config['beta'],
            t_min=config['t_min'], prior_toxicity=0.01,
            device=device, max_length=config['max_length']
        )

        late_toxicity_results = analyze_late_toxicity(
            sprt_detector, test_texts, test_labels,
            test['unsafe_positions'], tokenizer
        )

        early_rate = late_toxicity_results['early_toxic'].get('detection_rate', 0)
        late_rate = late_toxicity_results['late_toxic'].get('detection_rate', 0)
        safe_fpr = late_toxicity_results['full_safe'].get('fpr', 0)
        print(f"Early toxic detection rate: {early_rate:.2%}" if early_rate else "Early toxic detection rate: N/A")
        print(f"Late toxic detection rate: {late_rate:.2%}" if late_rate else "Late toxic detection rate: N/A")
        print(f"Safe samples FPR: {safe_fpr:.2%}" if safe_fpr else "Safe samples FPR: N/A")

    # ============================================================
    # Save Results
    # ============================================================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    # Prepare results for JSON serialization
    results_to_save = {
        'dataset': ds_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'training_history': training_history,
        'calibration': {
            'ece_before': ece_before,
            'ece_temperature': ece_temp,
            'ece_isotonic': ece_isotonic,
            'temperature': temp_scaling.temperature.item()
        },
        'test_info': {
            'n_samples': len(test_texts),
            'toxicity_rate': sum(test_labels) / len(test_labels)
        },
        'results': {}
    }

    for method, res in all_results.items():
        results_to_save['results'][method] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in res.items()
            if k not in ['predictions', 'trajectories']
        }

    if late_toxicity_results:
        results_to_save['late_toxicity_analysis'] = late_toxicity_results

    # Save to file
    output_file = os.path.join(output_dir, f'{dataset_name}_comprehensive_results.json')
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Results")
    print("=" * 70)
    print(f"\n{'Method':<35} {'F1':>8} {'Acc':>8} {'FPR':>6} {'FNR':>6} {'AvgTok':>8}")
    print("-" * 73)

    for method, res in all_results.items():
        if 'f1' in res:
            print(f"{method:<35} {res['f1']:>8.4f} {res['accuracy']:>8.4f} "
                  f"{res['fpr']:>6.3f} {res['fnr']:>6.3f} {res.get('avg_stopping_time', config['max_length']):>8.1f}")

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)

    return all_results


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive SPRT Experiments')
    parser.add_argument('--dataset', type=str, default='civilcomments',
                        choices=['civilcomments', 'beavertails', 'pku_saferlhf', 'qwen3guardtest'],
                        help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='../llm_security_datasets',
                        help='Path to local datasets')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    run_comprehensive_experiments(
        dataset_name=args.dataset,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device
    )
