#!/usr/bin/env python3
"""
Cascade-Guard: Heterogeneous Two-Stage Streaming Safety Detection

Architecture:
  Layer 1 (Fast Screening): Fine-tuned RoBERTa with Wald prior-adjusted threshold
    → Flags potentially unsafe prefixes during streaming (high recall, high FPR)
  Layer 2 (Precise Confirmation): Llama Guard on the flagged prefix
    → Only confirms if Llama Guard also declares unsafe (low FPR)

Key insight: The two stages use DIFFERENT models (RoBERTa vs Llama Guard),
so the cascade does NOT collapse to a single threshold. The models have
independent error modes, making the AND combination genuinely beneficial.

Usage:
  python cascade_guard_experiments.py --dataset beavertails --device cuda
  python cascade_guard_experiments.py --dataset all --device cuda

Author: Qiu Jianmin, Han Jinguang
Date: April 2026
"""

import os
import sys

# Set HF mirror for China-based servers (AutoDL)
if os.environ.get("HF_ENDPOINT", "") == "":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "autodl" in os.environ.get("HF_HOME", ""):
    pass  # Already set
elif os.path.exists("/root/autodl-tmp"):
    os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"

import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Note: WaldThresholdDetector import removed - all methods defined inline

# ============================================================
# Configuration
# ============================================================

CASCADE_CONFIG = {
    # Stage 1 (RoBERTa) settings
    "alpha": 0.05,
    "beta": 0.10,
    "t_min": 5,
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 3,
    "priors": [0.50, 0.10, 0.05, 0.01],

    # Stage 2 (Llama Guard) settings
    "llama_guard_model": "meta-llama/LlamaGuard-3-8B",
    "llama_guard_max_length": 4096,

    # Confidence baseline sweep
    "confidence_thresholds": [0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95],
}


# ============================================================
# Metrics Utility
# ============================================================

def compute_metrics(y_true, y_pred):
    """Compute binary classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    return {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "fpr": float(fpr), "fnr": float(fnr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ============================================================
# Stage 2: Llama Guard Classifier
# ============================================================

class LlamaGuardClassifier:
    """
    Wrapper for Llama Guard model as Stage 2 confirmation classifier.

    Llama Guard outputs:
    - "safe" → content is safe
    - "unsafe" → content is unsafe (with optional category)
    """

    def __init__(self, model_name_or_path: str, device: str = "cuda", max_length: int = 4096):
        self.device = device
        self.max_length = max_length
        self.model_name = model_name_or_path

        print(f"Loading Llama Guard from {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if device == "cuda" and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
            )

        self.model.eval()
        print(f"Llama Guard loaded successfully.")

    def classify_single(self, text: str) -> Tuple[int, float, str]:
        """Classify a single text. Returns (label, confidence, raw_output)."""
        conversation = [{"role": "user", "content": text}]

        input_ids = self.tokenizer.apply_chat_template(
            conversation, return_tensors="pt", return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **input_ids, max_new_tokens=64, do_sample=False,
            )

        new_tokens = output[0][input_ids["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        is_unsafe = "unsafe" in raw_output.lower()

        return (1 if is_unsafe else 0), (1.0 if is_unsafe else 0.0), raw_output

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """Classify a batch of texts."""
        results = []
        for text in tqdm(texts, desc="Llama Guard"):
            start = time.time()
            label, conf, raw = self.classify_single(text)
            latency = (time.time() - start) * 1000
            results.append({
                "label": label, "confidence": conf,
                "raw_output": raw, "latency_ms": latency,
            })
        return results

    def benchmark_latency(self, n_samples: int = 50) -> Dict:
        """Measure average inference latency."""
        dummy_text = "This is a sample text for benchmarking latency measurement. " * 20
        latencies = []
        for _ in range(n_samples):
            start = time.time()
            self.classify_single(dummy_text[:512])
            latencies.append((time.time() - start) * 1000)
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "median_ms": float(np.median(latencies)),
            "n_samples": n_samples,
        }


# ============================================================
# Cascade-Guard: Two-Stage Architecture
# ============================================================

class CascadeGuard:
    """
    Two-stage cascade detection architecture.

    Stage 1: RoBERTa with Wald prior-adjusted threshold (fast screening)
    Stage 2: Llama Guard on flagged prefixes (precise confirmation)

    The cascade only declares UNSAFE if BOTH stages agree.
    Since the stages use different models, this does NOT reduce to a single threshold.
    """

    def __init__(
        self,
        stage1_model, stage1_tokenizer,
        stage2: LlamaGuardClassifier,
        alpha: float = 0.05, beta: float = 0.10,
        prior: float = 0.01, t_min: int = 5,
        max_length: int = 128, device: str = "cuda",
    ):
        self.s1_model = stage1_model
        self.s1_tokenizer = stage1_tokenizer
        self.stage2 = stage2
        self.alpha = alpha
        self.beta = beta
        self.prior = prior
        self.t_min = t_min
        self.max_length = max_length
        self.device = device

        # Compute Wald thresholds with prior adjustment
        self.A = np.log((1 - beta) / alpha) + np.log(prior / (1 - prior))
        self.B = np.log(beta / (1 - alpha)) + np.log(prior / (1 - prior))
        self.epsilon = 1e-10

    def predict_single(self, text: str) -> Dict:
        """Run cascade on a single text."""
        t0 = time.time()
        tokens = self.s1_tokenizer.encode(text, add_special_tokens=True)
        n_tokens = min(len(tokens), self.max_length)

        stage1_prob = 0.5
        stage1_decision = "SAFE"
        stop_time = n_tokens
        s1_calls = 0

        # Stage 1: Streaming detection with Wald threshold
        for t in range(self.t_min, n_tokens + 1):
            prefix_text = self.s1_tokenizer.decode(tokens[:t], skip_special_tokens=True)
            inputs = self.s1_tokenizer(
                prefix_text, return_tensors='pt',
                truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.s1_model(**inputs)
                p_t = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
            s1_calls += 1

            # Compute evidence score (logit)
            S_t = np.log((p_t + self.epsilon) / (1 - p_t + self.epsilon))

            if S_t >= self.A:
                stage1_decision = "UNSAFE"
                stage1_prob = p_t
                stop_time = t
                break
            elif S_t <= self.B:
                stage1_decision = "SAFE"
                stage1_prob = p_t
                stop_time = t
                break

        # Stage 2: If Stage 1 says UNSAFE, confirm with Llama Guard
        stage2_triggered = stage1_decision == "UNSAFE"
        stage2_label = None
        stage2_latency_ms = 0.0
        cascade_decision = stage1_decision

        if stage2_triggered:
            prefix_text = self.s1_tokenizer.decode(
                tokens[:stop_time], skip_special_tokens=True
            )
            t1 = time.time()
            stage2_label, _, _ = self.stage2.classify_single(prefix_text)
            stage2_latency_ms = (time.time() - t1) * 1000

            if stage2_label == 0:  # Llama Guard says SAFE → overrule Stage 1
                cascade_decision = "SAFE"
            else:  # Both agree UNSAFE
                cascade_decision = "UNSAFE"

        total_latency_ms = (time.time() - t0) * 1000

        return {
            "decision": cascade_decision,
            "stage1_decision": stage1_decision,
            "stage2_triggered": stage2_triggered,
            "stage2_label": stage2_label,
            "stopping_time": stop_time,
            "stage1_prob": stage1_prob,
            "s1_calls": s1_calls,
            "stage2_latency_ms": stage2_latency_ms,
            "total_latency_ms": total_latency_ms,
        }

    def predict_batch(self, texts: List[str], labels: Optional[List[int]] = None) -> Dict:
        """Run cascade on a batch of texts."""
        decisions = []
        stage1_decisions = []
        stage2_triggereds = []
        stopping_times = []
        s1_calls_list = []
        s2_latencies = []
        total_latencies = []

        for text in tqdm(texts, desc=f"Cascade(π={self.prior})"):
            result = self.predict_single(text)
            decisions.append(result["decision"])
            stage1_decisions.append(result["stage1_decision"])
            stage2_triggereds.append(result["stage2_triggered"])
            stopping_times.append(result["stopping_time"])
            s1_calls_list.append(result["s1_calls"])
            s2_latencies.append(result["stage2_latency_ms"])
            total_latencies.append(result["total_latency_ms"])

        results = {
            "avg_stopping_time": float(np.mean(stopping_times)),
            "n_stage2_triggered": sum(stage2_triggereds),
            "stage2_trigger_rate": float(np.mean(stage2_triggereds)),
            "avg_s1_calls": float(np.mean(s1_calls_list)),
            "avg_s2_latency_ms": float(np.mean([l for l in s2_latencies if l > 0])) if any(l > 0 for l in s2_latencies) else 0,
            "avg_total_latency_ms": float(np.mean(total_latencies)),
            "median_total_latency_ms": float(np.median(total_latencies)),
            "method": f"Cascade(π={self.prior})",
        }

        if labels is not None:
            y_pred = [1 if d == "UNSAFE" else 0 for d in decisions]
            results.update(compute_metrics(labels, y_pred))

        return results


# ============================================================
# Inline Streaming Baselines (no cross-module imports)
# ============================================================

class StreamingSPRTBaseline:
    """Single-stage SPRT streaming baseline (Wald threshold on RoBERTa)."""

    def __init__(self, model, tokenizer, alpha=0.05, beta=0.10, prior=0.5,
                 t_min=5, max_length=128, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.prior = prior
        self.t_min = t_min
        self.max_length = max_length
        self.device = device
        self.A = np.log((1 - beta) / alpha) + np.log(prior / (1 - prior))
        self.B = np.log(beta / (1 - alpha)) + np.log(prior / (1 - prior))
        self.epsilon = 1e-10

    def predict_batch(self, texts, labels=None):
        decisions = []
        stopping_times = []
        for text in tqdm(texts, desc=f"SPRT(π={self.prior})"):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            n_tokens = min(len(tokens), self.max_length)
            decision = "SAFE"
            stop_time = n_tokens
            for t in range(self.t_min, n_tokens + 1):
                prefix = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                inputs = self.tokenizer(
                    prefix, return_tensors='pt',
                    truncation=True, max_length=self.max_length
                ).to(self.device)
                with torch.no_grad():
                    p_t = torch.softmax(self.model(**inputs).logits, dim=-1)[0, 1].item()
                S_t = np.log((p_t + self.epsilon) / (1 - p_t + self.epsilon))
                if S_t >= self.A:
                    decision = "UNSAFE"
                    stop_time = t
                    break
                elif S_t <= self.B:
                    decision = "SAFE"
                    stop_time = t
                    break
            decisions.append(decision)
            stopping_times.append(stop_time)

        results = {"avg_stopping_time": float(np.mean(stopping_times))}
        if labels is not None:
            y_pred = [1 if d == "UNSAFE" else 0 for d in decisions]
            results.update(compute_metrics(labels, y_pred))
        return results


class StreamingConfidenceBaseline:
    """Single-stage confidence threshold streaming baseline."""

    def __init__(self, model, tokenizer, threshold=0.9, t_min=5,
                 max_length=128, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.t_min = t_min
        self.max_length = max_length
        self.device = device

    def predict_batch(self, texts, labels=None):
        decisions = []
        stopping_times = []
        for text in tqdm(texts, desc=f"Conf(τ={self.threshold})"):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            n_tokens = min(len(tokens), self.max_length)
            decision = "SAFE"
            stop_time = n_tokens
            for t in range(self.t_min, n_tokens + 1):
                prefix = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                inputs = self.tokenizer(
                    prefix, return_tensors='pt',
                    truncation=True, max_length=self.max_length
                ).to(self.device)
                with torch.no_grad():
                    p_t = torch.softmax(self.model(**inputs).logits, dim=-1)[0, 1].item()
                if p_t >= self.threshold:
                    decision = "UNSAFE"
                    stop_time = t
                    break
                elif p_t >= (1 - self.threshold):
                    decision = "SAFE"
                    stop_time = t
                    break
            decisions.append(decision)
            stopping_times.append(stop_time)

        results = {"avg_stopping_time": float(np.mean(stopping_times))}
        if labels is not None:
            y_pred = [1 if d == "UNSAFE" else 0 for d in decisions]
            results.update(compute_metrics(labels, y_pred))
        return results


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset(dataset_name, data_path=None):
    """Load a dataset and return train/val/test splits."""
    from datasets import load_dataset

    if dataset_name == 'civilcomments':
        train = load_dataset("google/civil_comments", split="train[:20000]")
        val = load_dataset("google/civil_comments", split="validation[:2000]")
        test = load_dataset("google/civil_comments", split="test[:5000]")

        def preprocess(examples):
            texts, labels = [], []
            for text, toxicity in zip(examples['text'], examples['toxicity']):
                if text and text.strip():
                    texts.append(text.strip())
                    labels.append(1 if (toxicity or 0) >= 0.5 else 0)
            return {'text': texts, 'label': labels}

        train = train.map(lambda x: preprocess(x), batched=True, remove_columns=train.column_names)
        val = val.map(lambda x: preprocess(x), batched=True, remove_columns=val.column_names)
        test = test.map(lambda x: preprocess(x), batched=True, remove_columns=test.column_names)
        ds_name = "CivilComments"

    elif dataset_name == 'beavertails':
        ds = load_dataset("PKU-Alignment/BeaverTails")
        # BeaverTails uses '30k_test' split, not 'test'
        split_name = "30k_test" if "30k_test" in ds else "test"
        test_data = ds[split_name]
        texts = [item["prompt"] + " " + item["response"]
                 for item in test_data if item.get("prompt") and item.get("response")]
        labels = [1 if not item.get("is_safe", True) else 0 for item in test_data
                  if item.get("prompt") and item.get("response")]
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels)
        train = {'text': train_texts, 'label': train_labels}
        val = {'text': val_texts, 'label': val_labels}
        test = {'text': test_texts[:3021], 'label': test_labels[:3021]}
        ds_name = "BeaverTails"

    elif dataset_name == 'pku_saferlhf':
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF")
        all_texts, all_labels = [], []
        for item in ds["train"]:
            prompt = item.get("prompt", "")
            response = item.get("response_0", "")
            # Field name is is_response_0_safe (not is_safe_0)
            is_safe = item.get("is_response_0_safe", True)
            if prompt and response:
                all_texts.append(prompt + " " + response)
                all_labels.append(0 if is_safe else 1)
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.3, random_state=42)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1, random_state=42)
        train = {'text': train_texts, 'label': train_labels}
        val = {'text': val_texts, 'label': val_labels}
        test = {'text': test_texts[:3000], 'label': test_labels[:3000]}
        ds_name = "PKU-SafeRLHF"

    elif dataset_name == 'qwen3guardtest':
        ds = load_dataset("Qwen/Qwen3GuardTest")
        # Qwen3GuardTest has splits: thinking, response_loc, thinking_loc
        # Each split has 'message' field (chat-format list of dicts), all labeled Unsafe
        all_texts, all_labels = [], []
        for split_name in ds:
            for item in ds[split_name]:
                messages = item.get("message", [])
                # Extract text from chat format: concatenate user + assistant content
                parts = []
                for msg in messages:
                    content = msg.get("content", "")
                    if content:
                        parts.append(content)
                text = " ".join(parts).strip()
                if text:
                    all_texts.append(text)
                    all_labels.append(1)  # All adversarial/unsafe
        print(f"Qwen3GuardTest: loaded {len(all_texts)} samples from all splits")
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=42)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.125, random_state=42)
        train = {'text': train_texts, 'label': train_labels}
        val = {'text': val_texts, 'label': val_labels}
        test = {'text': test_texts[:651], 'label': test_labels[:651]}
        ds_name = "Qwen3GuardTest"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train, val, test, ds_name


# ============================================================
# RoBERTa Training
# ============================================================

def train_roberta(train_data, val_data, device='cuda', config=None):
    """Train a RoBERTa safety classifier."""
    config = config or CASCADE_CONFIG
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)

    class SafetyDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            return {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    train_ds = SafetyDataset(train_data['text'], train_data['label'], tokenizer, config['max_length'])
    val_ds = SafetyDataset(val_data['text'], val_data['label'], tokenizer, config['max_length'])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_f1 = 0
    best_state = None
    for epoch in range(config['epochs']):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                outputs = model(**inputs)
                val_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                val_true.extend(batch['labels'].numpy())

        val_f1 = f1_score(val_true, val_preds, average='binary', zero_division=0)
        print(f"Epoch {epoch+1}: Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, tokenizer


# ============================================================
# Main Experiment Runner
# ============================================================

def run_cascade_experiments(
    dataset_name='beavertails',
    llama_guard_path=None,
    data_path=None,
    output_dir='./results',
    device='cuda',
    config=None,
):
    """Run complete Cascade-Guard experiments for a dataset."""
    config = config or CASCADE_CONFIG
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Cascade-Guard Experiments - {dataset_name}")
    print("=" * 70)

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 1. Load dataset
    train, val, test, ds_name = load_dataset(dataset_name, data_path)
    test_texts = test['text']
    test_labels = test['label']
    n_samples = len(test_texts)
    tox_rate = sum(test_labels) / len(test_labels)

    print(f"\nDataset: {ds_name}")
    print(f"Test: {n_samples} samples, Toxicity: {tox_rate:.1%}")

    # 2. Train Stage 1 (RoBERTa)
    print("\n" + "=" * 70)
    print("Stage 1: Training RoBERTa Classifier")
    print("=" * 70)
    s1_model, s1_tokenizer = train_roberta(train, val, device, config)
    s1_model.eval()

    # 3. Load Stage 2 (Llama Guard)
    print("\n" + "=" * 70)
    print("Stage 2: Loading Llama Guard")
    print("=" * 70)
    lg_path = llama_guard_path or config['llama_guard_model']
    stage2 = LlamaGuardClassifier(lg_path, device, config['llama_guard_max_length'])

    # Benchmark Stage 2 latency
    print("\nBenchmarking Llama Guard latency...")
    lg_latency = stage2.benchmark_latency()
    print(f"Llama Guard: {lg_latency['mean_ms']:.1f} +/- {lg_latency['std_ms']:.1f} ms/call")

    # 4. Run all methods
    print("\n" + "=" * 70)
    print("Running All Methods")
    print("=" * 70)

    all_results = {}

    # 4a. Full-text baselines
    print("\n[1/5] Full-text baselines...")

    # RoBERTa full-text
    print("  RoBERTa full-text...")
    ft_probs = []
    for text in tqdm(test_texts, desc="RoBERTa-FT"):
        inputs = s1_tokenizer(text, return_tensors='pt', truncation=True, max_length=config['max_length']).to(device_obj)
        with torch.no_grad():
            prob = torch.softmax(s1_model(**inputs).logits, dim=-1)[0, 1].item()
        ft_probs.append(prob)
    ft_preds = [1 if p >= 0.5 else 0 for p in ft_probs]

    all_results['RoBERTa-FT'] = compute_metrics(test_labels, ft_preds)
    all_results['RoBERTa-FT']['avg_stopping_time'] = 128
    all_results['RoBERTa-FT']['savings'] = 0

    # Llama Guard full-text
    print("  Llama Guard full-text...")
    lg_results = stage2.classify_batch(test_texts)
    lg_preds = [r['label'] for r in lg_results]
    lg_latencies = [r['latency_ms'] for r in lg_results]

    all_results['LlamaGuard-FT'] = compute_metrics(test_labels, lg_preds)
    all_results['LlamaGuard-FT']['avg_stopping_time'] = 128
    all_results['LlamaGuard-FT']['savings'] = 0
    all_results['LlamaGuard-FT']['avg_latency_ms'] = float(np.mean(lg_latencies))

    # 4b. Single-stage streaming baselines
    print("\n[2/5] Single-stage SPRT streaming baselines...")

    for prior in config['priors']:
        sprt = StreamingSPRTBaseline(
            s1_model, s1_tokenizer,
            alpha=config['alpha'], beta=config['beta'],
            prior=prior, t_min=config['t_min'],
            max_length=config['max_length'], device=device_obj,
        )
        res = sprt.predict_batch(test_texts, test_labels)
        avg_tok = res['avg_stopping_time']
        key = f'SPRT(pi={prior})'
        all_results[key] = {
            **res,
            'savings': round((1 - avg_tok / 128) * 100, 1)
        }
        print(f"  {key}: F1={res['f1']:.3f} FPR={res['fpr']:.3f} Savings={all_results[key]['savings']}%")

    print("\n[3/5] Confidence baseline sweep...")
    for th in config['confidence_thresholds']:
        conf = StreamingConfidenceBaseline(
            s1_model, s1_tokenizer, threshold=th,
            t_min=config['t_min'], max_length=config['max_length'],
            device=device_obj,
        )
        res = conf.predict_batch(test_texts, test_labels)
        avg_tok = res['avg_stopping_time']
        key = f'Conf(tau={th})'
        all_results[key] = {
            **res,
            'savings': round((1 - avg_tok / 128) * 100, 1)
        }
        print(f"  {key}: F1={res['f1']:.3f} FPR={res['fpr']:.3f} Savings={all_results[key]['savings']}%")

    # 4c. Cascade-Guard (different priors)
    print("\n[4/5] Cascade-Guard (RoBERTa screening + Llama Guard confirmation)...")

    for prior in config['priors']:
        cascade = CascadeGuard(
            stage1_model=s1_model,
            stage1_tokenizer=s1_tokenizer,
            stage2=stage2,
            alpha=config['alpha'], beta=config['beta'],
            prior=prior, t_min=config['t_min'],
            max_length=config['max_length'], device=device_obj,
        )
        res = cascade.predict_batch(test_texts, test_labels)
        avg_tok = res['avg_stopping_time']
        key = f'Cascade(pi={prior})'
        all_results[key] = {
            **res,
            'savings': round((1 - avg_tok / 128) * 100, 1)
        }
        print(f"  {key}: F1={res['f1']:.3f} FPR={res['fpr']:.3f} Savings={all_results[key]['savings']}% "
              f"S2-calls={res['n_stage2_triggered']}/{n_samples}")

    # 4d. Cost analysis (reviewer requirement: efficiency metrics)
    print("\n[5/5] Computational cost analysis...")

    # Model FLOPs estimates ( reviewer requirement )
    roberta_flops_per_call = 0.5e9    # ~0.5 GFLOPs per RoBERTa-base forward pass
    llama_guard_flops_per_call = 16e9  # ~16 GFLOPs per Llama Guard 3 8B forward pass (bfloat16)
    llm_ms_per_token = 50  # estimated LLM generation latency

    cost_analysis = {
        "roberta_per_call_ms": 6.05,
        "llama_guard_per_call_ms": lg_latency['mean_ms'],
        "roberta_flops_per_call": roberta_flops_per_call,
        "llama_guard_flops_per_call": llama_guard_flops_per_call,
        "efficiency": {},
    }

    for method_name, res in all_results.items():
        avg_tok = res.get('avg_stopping_time', 128)
        avg_s1 = res.get('avg_s1_calls', avg_tok)  # fallback

        if 'Cascade' in method_name:
            s2_rate = res.get('stage2_trigger_rate', 0.5)
            # FLOPs: s1 calls * roberta + s2_rate * llama_guard
            total_flops = avg_s1 * roberta_flops_per_call + s2_rate * llama_guard_flops_per_call
            lg_calls_pct = s2_rate * 100
            # Latency model: LLM gen + S1 calls + S2 calls
            total_ms = avg_tok * llm_ms_per_token + avg_s1 * 6.05 + s2_rate * lg_latency['mean_ms']
        elif 'SPRT' in method_name or 'Conf' in method_name:
            total_flops = avg_s1 * roberta_flops_per_call
            lg_calls_pct = 0
            total_ms = avg_tok * llm_ms_per_token + avg_s1 * 6.05
        elif 'LlamaGuard' in method_name:
            total_flops = llama_guard_flops_per_call
            lg_calls_pct = 100
            total_ms = 128 * llm_ms_per_token + lg_latency['mean_ms']
        else:  # Full-text RoBERTa
            total_flops = roberta_flops_per_call
            lg_calls_pct = 0
            total_ms = 128 * llm_ms_per_token + 6.05

        lg_total = llama_guard_flops_per_call  # Llama Guard-only FLOPs baseline
        flops_savings = (1 - total_flops / lg_total) * 100 if lg_total > 0 else 0

        ft_total_ms = 128 * llm_ms_per_token + lg_latency['mean_ms']  # vs LG-only

        cost_analysis['efficiency'][method_name] = {
            "total_flops_gflops": round(total_flops / 1e9, 2),
            "flops_savings_vs_lg_only_pct": round(flops_savings, 1),
            "lg_calls_pct": round(lg_calls_pct, 1),
            "estimated_total_ms": round(total_ms, 1),
            "speedup_vs_lg_only": round(ft_total_ms / total_ms, 2),
            "token_savings_pct": round((1 - avg_tok / 128) * 100, 1),
        }

    # 5. Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    results_to_save = {
        'dataset': ds_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'test_info': {
            'n_samples': n_samples,
            'toxicity_rate': tox_rate,
        },
        'stage2_latency': lg_latency,
        'cost_analysis': cost_analysis,
        'results': {},
    }

    for method_name, res in all_results.items():
        results_to_save['results'][method_name] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer, np.bool_)) else
                int(v) if isinstance(v, (np.integer,)) else v)
            for k, v in res.items()
        }

    output_file = os.path.join(output_dir, f'{dataset_name}_cascade_guard_results.json')
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"Results saved to: {output_file}")

    # 6. Print summary
    print("\n" + "=" * 70)
    print("Summary Results")
    print("=" * 70)
    print(f"\n{'Method':<30} {'F1':>6} {'FPR':>6} {'FNR':>6} {'Savings':>8} {'LG-Calls':>10} {'FLOPs-Sav':>10}")
    print("-" * 80)

    for method_name, res in all_results.items():
        if 'f1' in res:
            s2 = res.get('n_stage2_triggered', 0)
            eff = cost_analysis['efficiency'].get(method_name, {})
            flops_sav = eff.get('flops_savings_vs_lg_only_pct', 0)
            print(f"{method_name:<30} {res['f1']:>6.3f} {res['fpr']:>6.3f} {res['fnr']:>6.3f} "
                  f"{res.get('savings', 0):>7.1f}% {str(s2):>10} {flops_sav:>9.1f}%")

    return all_results


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cascade-Guard Experiments')
    parser.add_argument('--dataset', type=str, default='beavertails',
                        choices=['civilcomments', 'beavertails', 'pku_saferlhf', 'qwen3guardtest', 'all'])
    parser.add_argument('--llama_guard_path', type=str, default=None,
                        help='Path to Llama Guard model (default: meta-llama/LlamaGuard-3-8B)')
    parser.add_argument('--data_path', type=str, default='../llm_security_datasets')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if args.dataset == 'all':
        for ds in ['civilcomments', 'beavertails', 'pku_saferlhf', 'qwen3guardtest']:
            run_cascade_experiments(
                dataset_name=ds,
                llama_guard_path=args.llama_guard_path,
                data_path=args.data_path,
                output_dir=args.output_dir,
                device=args.device,
            )
    else:
        run_cascade_experiments(
            dataset_name=args.dataset,
            llama_guard_path=args.llama_guard_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            device=args.device,
        )
