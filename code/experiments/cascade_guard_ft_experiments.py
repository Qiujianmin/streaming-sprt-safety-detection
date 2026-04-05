#!/usr/bin/env python3
"""
Cascade-Guard with Prefix-Fine-Tuned Stage 2: Comparative Experiments

Compares:
  - Cascade-ZS: Zero-shot Llama Guard (original)
  - Cascade-FT: Prefix-fine-tuned Llama Guard (QLoRA adapter)

This addresses Reviewer Major Comment 2: "The cascade fails because
Llama Guard is trained on full texts, not prefixes."

Usage:
  python cascade_guard_ft_experiments.py --adapter_path /tmp/llama_guard_prefix_finetuned/final_adapter
"""

import os
import sys

# Set HF mirror for China-based servers (AutoDL)
if os.environ.get("HF_ENDPOINT", "") == "":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if os.path.exists("/root/autodl-tmp"):
    os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"

import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# Metrics Utility
# ============================================================

def compute_metrics(y_true, y_pred):
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
# Llama Guard Classifier (supports both ZS and FT)
# ============================================================

class LlamaGuardClassifier:
    """Wrapper for Llama Guard model (zero-shot or fine-tuned)."""

    def __init__(self, model_name_or_path: str, adapter_path: str = None,
                 device: str = "cuda", max_length: int = 4096):
        self.device = device
        self.max_length = max_length
        self.model_name = model_name_or_path
        self.adapter_path = adapter_path

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

        # Load QLoRA adapter if provided
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading QLoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
            print("Adapter merged successfully.")

        self.model.eval()
        tag = "FT" if adapter_path else "ZS"
        print(f"Llama Guard ({tag}) loaded.")

    def classify_single(self, text: str):
        """Classify a single text. Returns (label, raw_output)."""
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

        return (1 if is_unsafe else 0), raw_output


# ============================================================
# Cascade-Guard Runner
# ============================================================

def run_cascade_on_dataset(
    dataset_name: str,
    stage2_classifier: LlamaGuardClassifier,
    s1_model, s1_tokenizer,
    prior: float = 0.01,
    alpha: float = 0.05,
    beta: float = 0.10,
    t_min: int = 5,
    max_length: int = 128,
    device: str = "cuda",
    prefix_failures_path: str = None,
):
    """Run cascade on a dataset with a given Stage 2 classifier."""

    # Load dataset (reuse from parent experiment code)
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    if dataset_name == 'civilcomments':
        test = load_dataset("google/civil_comments", split="test[:5000]")
        test_texts = [t.strip() for t in test['text'] if t and t.strip()]
        test_labels = [1 if (tox or 0) >= 0.5 else 0 for t, tox in zip(test['text'], test['toxicity']) if t and t.strip()]
    elif dataset_name == 'beavertails':
        ds = load_dataset("PKU-Alignment/BeaverTails")
        split_name = "30k_test" if "30k_test" in ds else "test"
        test_data = ds[split_name]
        texts = [item["prompt"] + " " + item["response"]
                 for item in test_data if item.get("prompt") and item.get("response")]
        labels = [1 if not item.get("is_safe", True) else 0 for item in test_data
                  if item.get("prompt") and item.get("response")]
        _, test_texts, _, test_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels)
        test_texts = test_texts[:907]
        test_labels = test_labels[:907]
    elif dataset_name == 'pku_saferlhf':
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF")
        all_texts, all_labels = [], []
        for item in ds["train"]:
            prompt = item.get("prompt", "")
            response = item.get("response_0", "")
            is_safe = item.get("is_response_0_safe", True)
            if prompt and response:
                all_texts.append(prompt + " " + response)
                all_labels.append(0 if is_safe else 1)
        _, test_texts, _, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.3, random_state=42)
        test_texts = test_texts[:3000]
        test_labels = test_labels[:3000]
    elif dataset_name == 'qwen3guardtest':
        ds = load_dataset("Qwen/Qwen3GuardTest")
        all_texts, all_labels = [], []
        for split_name in ds:
            for item in ds[split_name]:
                messages = item.get("message", [])
                parts = []
                for msg in messages:
                    content = msg.get("content", "")
                    if content:
                        parts.append(content)
                text = " ".join(parts).strip()
                if text:
                    all_texts.append(text)
                    all_labels.append(1)
        _, test_texts, _, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=42)
        test_texts = test_texts[:489]
        test_labels = test_labels[:489]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Compute Wald thresholds
    A = np.log((1 - beta) / alpha) + np.log(prior / (1 - prior))
    B = np.log(beta / (1 - alpha)) + np.log(prior / (1 - prior))
    epsilon = 1e-10

    tag = "FT" if stage2_classifier.adapter_path else "ZS"
    predictions = []
    stopping_times = []
    stage2_triggers = 0
    prefix_failures = []  # Cases where S1 correct but S2 overrules

    for idx, (text, true_label) in enumerate(tqdm(
        zip(test_texts, test_labels),
        total=len(test_texts),
        desc=f"Cascade-{tag}({dataset_name},π={prior})"
    )):
        tokens = s1_tokenizer.encode(text, add_special_tokens=True)
        n_tokens = min(len(tokens), max_length)

        # Stage 1: RoBERTa streaming
        stage1_decision = "SAFE"
        stage1_prob = 0.5
        stop_time = n_tokens

        for t in range(t_min, n_tokens + 1):
            prefix_text = s1_tokenizer.decode(tokens[:t], skip_special_tokens=True)
            inputs = s1_tokenizer(
                prefix_text, return_tensors='pt',
                truncation=True, max_length=max_length
            ).to(device)

            with torch.no_grad():
                p_t = torch.softmax(s1_model(**inputs).logits, dim=-1)[0, 1].item()

            S_t = np.log((p_t + epsilon) / (1 - p_t + epsilon))

            if S_t >= A:
                stage1_decision = "UNSAFE"
                stage1_prob = p_t
                stop_time = t
                break
            elif S_t <= B:
                stage1_decision = "SAFE"
                stage1_prob = p_t
                stop_time = t
                break

        # Stage 2: Llama Guard confirmation
        cascade_decision = stage1_decision
        if stage1_decision == "UNSAFE":
            prefix_text = s1_tokenizer.decode(tokens[:stop_time], skip_special_tokens=True)
            s2_label, s2_raw = stage2_classifier.classify_single(prefix_text)
            stage2_triggers += 1

            if s2_label == 0:  # Stage 2 says SAFE → overrule
                cascade_decision = "SAFE"
                # Track prefix failure: S1 said UNSAFE (correct) but S2 overruled
                if true_label == 1:
                    prefix_failures.append({
                        'index': idx,
                        'prefix_text': prefix_text[:200],
                        'prefix_tokens': stop_time,
                        's1_prob': round(stage1_prob, 4),
                        's2_raw': s2_raw[:100],
                        'true_label': true_label,
                        'dataset': dataset_name,
                    })
            else:
                cascade_decision = "UNSAFE"
        else:
            s2_label = None
            s2_raw = None

        predictions.append(1 if cascade_decision == "UNSAFE" else 0)
        stopping_times.append(stop_time)

    # Compute metrics
    metrics = compute_metrics(test_labels, predictions)
    metrics['avg_stopping_time'] = float(np.mean(stopping_times))
    metrics['n_stage2_triggered'] = stage2_triggers
    metrics['stage2_trigger_rate'] = stage2_triggers / len(test_texts)
    metrics['savings'] = round((1 - np.mean(stopping_times) / 128) * 100, 1)
    metrics['n_prefix_failures'] = len(prefix_failures)

    # Save prefix failures if path provided
    if prefix_failures_path:
        with open(prefix_failures_path, 'w') as f:
            for pf in prefix_failures:
                f.write(json.dumps(pf) + '\n')

    return metrics, prefix_failures


# ============================================================
# RoBERTa Training
# ============================================================

def train_roberta(train_data, val_data, device='cuda'):
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
            enc = self.tokenizer(self.texts[idx], max_length=self.max_length,
                                padding='max_length', truncation=True, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

    train_ds = SafetyDataset(train_data['text'], train_data['label'], tokenizer)
    val_ds = SafetyDataset(val_data['text'], val_data['label'], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_f1 = 0
    best_state = None
    for epoch in range(3):
        model.train()
        for batch in train_loader:
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
                logits = model(**inputs).logits
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                val_true.extend(batch['labels'].numpy())

        f1 = f1_score(val_true, val_preds, average='binary', zero_division=0)
        print(f"  Epoch {epoch+1}/3: val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return model, tokenizer


def load_dataset_splits(dataset_name):
    """Load train/val/test splits for a dataset."""
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

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

    elif dataset_name == 'beavertails':
        ds = load_dataset("PKU-Alignment/BeaverTails")
        split_name = "30k_test" if "30k_test" in ds else "test"
        test_data = ds[split_name]
        texts = [item["prompt"] + " " + item["response"]
                 for item in test_data if item.get("prompt") and item.get("response")]
        labels = [1 if not item.get("is_safe", True) else 0 for item in test_data
                  if item.get("prompt") and item.get("response")]
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels)
        train = {'text': train_texts, 'label': train_labels}
        val = {'text': val_texts, 'label': val_labels}
        test = {'text': test_texts[:907], 'label': test_labels[:907]}

    elif dataset_name == 'pku_saferlhf':
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF")
        all_texts, all_labels = [], []
        for item in ds["train"]:
            prompt = item.get("prompt", "")
            response = item.get("response_0", "")
            is_safe = item.get("is_response_0_safe", True)
            if prompt and response:
                all_texts.append(prompt + " " + response)
                all_labels.append(0 if is_safe else 1)
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.3, random_state=42)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1, random_state=42)
        train = {'text': train_texts, 'label': train_labels}
        val = {'text': val_texts, 'label': val_labels}
        test = {'text': test_texts[:3000], 'label': test_labels[:3000]}

    elif dataset_name == 'qwen3guardtest':
        ds = load_dataset("Qwen/Qwen3GuardTest")
        all_texts, all_labels = [], []
        for split_name in ds:
            for item in ds[split_name]:
                messages = item.get("message", [])
                parts = []
                for msg in messages:
                    content = msg.get("content", "")
                    if content:
                        parts.append(content)
                text = " ".join(parts).strip()
                if text:
                    all_texts.append(text)
                    all_labels.append(1)
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=42)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.125, random_state=42)
        train = {'text': train_texts, 'label': train_labels}
        val = {'text': val_texts, 'label': val_labels}
        test = {'text': test_texts[:489], 'label': test_labels[:489]}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train, val, test


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Cascade-Guard FT vs ZS Comparison')
    parser.add_argument('--adapter_path', type=str,
                        default='/tmp/llama_guard_prefix_finetuned/final_adapter',
                        help='Path to QLoRA adapter for prefix-fine-tuned Llama Guard')
    parser.add_argument('--llama_guard_path', type=str,
                        default='/root/autodl-tmp/llama_guard/LLM-Research/Llama-Guard-3-8B',
                        help='Path to base Llama Guard model')
    parser.add_argument('--output_dir', type=str,
                        default='/tmp/cascade_ft_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['beavertails', 'pku_saferlhf', 'qwen3guardtest'],
                        help='Datasets to evaluate')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # Best priors per dataset (from previous experiments)
    best_priors = {
        'beavertails': 0.1,
        'pku_saferlhf': 0.01,
        'qwen3guardtest': 0.05,
        'civilcomments': 0.01,
    }

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        # Load dataset and train Stage 1
        train, val, test = load_dataset_splits(ds_name)
        print(f"Train: {len(train['text'])}, Test: {len(test['text'])}")

        print("Training Stage 1 RoBERTa...")
        s1_model, s1_tokenizer = train_roberta(train, val, device)
        s1_model.eval()

        prior = best_priors.get(ds_name, 0.05)

        # ---- Zero-shot Llama Guard ----
        print("\n--- Zero-shot Llama Guard (Stage 2) ---")
        lg_path = args.llama_guard_path
        if not os.path.exists(lg_path):
            lg_path = "meta-llama/LlamaGuard-3-8B"

        stage2_zs = LlamaGuardClassifier(lg_path, adapter_path=None, device=device)

        zs_metrics, zs_failures = run_cascade_on_dataset(
            ds_name, stage2_zs, s1_model, s1_tokenizer,
            prior=prior, device=device,
            prefix_failures_path=os.path.join(args.output_dir, f'{ds_name}_zs_failures.jsonl'),
        )
        print(f"  ZS: F1={zs_metrics['f1']:.3f} FPR={zs_metrics['fpr']:.3f} "
              f"FNR={zs_metrics['fnr']:.3f} Failures={zs_metrics['n_prefix_failures']}")

        # Free ZS model memory
        del stage2_zs
        torch.cuda.empty_cache()

        # ---- Fine-tuned Llama Guard ----
        print("\n--- Fine-tuned Llama Guard (Stage 2) ---")
        if os.path.exists(args.adapter_path):
            stage2_ft = LlamaGuardClassifier(lg_path, adapter_path=args.adapter_path, device=device)

            ft_metrics, ft_failures = run_cascade_on_dataset(
                ds_name, stage2_ft, s1_model, s1_tokenizer,
                prior=prior, device=device,
                prefix_failures_path=os.path.join(args.output_dir, f'{ds_name}_ft_failures.jsonl'),
            )
            print(f"  FT: F1={ft_metrics['f1']:.3f} FPR={ft_metrics['fpr']:.3f} "
                  f"FNR={ft_metrics['fnr']:.3f} Failures={ft_metrics['n_prefix_failures']}")

            # Compare
            delta_f1 = ft_metrics['f1'] - zs_metrics['f1']
            delta_failures = zs_metrics['n_prefix_failures'] - ft_metrics['n_prefix_failures']
            print(f"\n  ΔF1={delta_f1:+.3f} ΔFailures={delta_failures:+d}")
        else:
            print(f"  Adapter not found at {args.adapter_path}, skipping FT.")
            ft_metrics = None
            ft_failures = []

        # Save results
        ds_result = {
            'dataset': ds_name,
            'prior': prior,
            'zero_shot': zs_metrics,
            'fine_tuned': ft_metrics,
        }
        all_results[ds_name] = ds_result

        result_path = os.path.join(args.output_dir, f'{ds_name}_cascade_comparison.json')
        with open(result_path, 'w') as f:
            json.dump(ds_result, f, indent=2, default=str)
        print(f"  Saved to {result_path}")

        # Free memory
        if 'stage2_ft' in locals():
            del stage2_ft
        del s1_model, s1_tokenizer
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Cascade-ZS vs Cascade-FT")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'ZS-F1':>8} {'FT-F1':>8} {'ΔF1':>8} "
          f"{'ZS-Fail':>8} {'FT-Fail':>8} {'ΔFail':>8}")
    print("-" * 80)

    for ds_name, res in all_results.items():
        zs = res['zero_shot']
        ft = res['fine_tuned']
        if ft:
            print(f"{ds_name:<20} {zs['f1']:>8.3f} {ft['f1']:>8.3f} {ft['f1']-zs['f1']:>+8.3f} "
                  f"{zs['n_prefix_failures']:>8d} {ft['n_prefix_failures']:>8d} "
                  f"{zs['n_prefix_failures']-ft['n_prefix_failures']:>+8d}")
        else:
            print(f"{ds_name:<20} {zs['f1']:>8.3f} {'N/A':>8} {'N/A':>8} "
                  f"{zs['n_prefix_failures']:>8d} {'N/A':>8} {'N/A':>8}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, 'combined_cascade_comparison.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_path}")


if __name__ == '__main__':
    main()
