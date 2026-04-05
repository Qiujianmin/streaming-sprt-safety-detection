#!/usr/bin/env python3
"""
Run Experiments 5-8 only (fixing the crash in exp5)
Loads pre-computed probabilities from the first run.
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# ============================================================
# Configuration (same as original)
# ============================================================
CONFIG = {
    "model_name": "cardiffnlp/twitter-roberta-base-offensive",
    "dataset": "civil_comments",
    "max_samples": 2000,
    "max_length": 128,
    "t_min": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])


# ============================================================
# Model & Data Loading (copied from original)
# ============================================================
def load_model():
    print(f"Loading model: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"])
    model.to(CONFIG["device"])
    model.eval()
    return model, tokenizer


def load_data(tokenizer, max_samples=2000):
    print(f"Loading dataset: {CONFIG['dataset']}...")
    try:
        dataset = load_dataset("civil_comments", trust_remote_code=True)
        test_data = dataset["test"]
    except Exception as e:
        print(f"Failed to load civil_comments: {e}")
        try:
            dataset = load_dataset("google/civil_comments_data")
            test_data = dataset["test"]
        except Exception as e2:
            print(f"Also failed: {e2}")
            return create_synthetic_dataset(max_samples)

    texts, labels = [], []
    for item in tqdm(test_data, desc="Processing dataset"):
        text = item.get("text", "")
        toxicity = item.get("toxicity", item.get("target", 0))
        if isinstance(toxicity, str):
            continue
        label = 1 if toxicity >= 0.5 else 0
        if len(text.strip()) > 20:
            texts.append(text)
            labels.append(label)

    print(f"Total samples: {len(texts)}, Toxic: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    if len(texts) > max_samples:
        indices = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    print(f"Subsampled: {len(texts)}, Toxic: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    return texts, labels


def create_synthetic_dataset(n=2000, toxicity_rate=0.08):
    print("Creating synthetic dataset...")
    safe_templates = [
        "I think this product is excellent and would recommend it to anyone.",
        "The weather today is beautiful, perfect for a walk in the park.",
        "Thank you for your help with this project, your guidance was valuable.",
        "This article provides helpful information about recent discoveries.",
        "I completely agree with your perspective on this important issue.",
        "The service was outstanding and exceeded all my expectations.",
        "Great work on this presentation, very informative and well organized.",
        "I appreciate your thoughtful feedback on this matter.",
        "This is a well-designed solution to the problem we discussed.",
        "The team did an excellent job on this project.",
    ]
    toxic_templates = [
        "This is absolutely disgusting and these people should be ashamed.",
        "People like you are the problem with society today.",
        "I hate everything about this stupid idea and the idiots behind it.",
        "This is the worst thing I have ever seen, complete garbage.",
        "You are completely worthless and your opinion means nothing.",
        "These parasites are destroying our community.",
        "I cannot believe how pathetic and useless this group is.",
        "This trash needs to be stopped before more damage is done.",
        "What a joke these people are, completely incompetent.",
        "The scum behind this should be locked up forever.",
    ]
    texts, labels = [], []
    n_toxic = int(n * toxicity_rate)
    for _ in range(n - n_toxic):
        texts.append(np.random.choice(safe_templates))
        labels.append(0)
    for _ in range(n_toxic):
        texts.append(np.random.choice(toxic_templates))
        labels.append(1)
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    return texts, labels


# ============================================================
# Detection Methods (copied from original)
# ============================================================
def get_classifier_prob(model, tokenizer, text, device, max_length=128):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        p_toxic = probs[-1].item()
    return p_toxic


def get_prefix_probs(model, tokenizer, text, device, max_length=128, chunk_k=1, t_min=5):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = min(len(tokens), max_length)
    probs = {}
    for t in range(t_min, n_tokens + 1, chunk_k):
        prefix = tokenizer.decode(tokens[:t], skip_special_tokens=True)
        p = get_classifier_prob(model, tokenizer, prefix, device, max_length)
        probs[t] = p
    return probs, n_tokens


class WaldThresholdDetector:
    def __init__(self, alpha=0.05, beta=0.10, prior=0.5, t_min=5):
        self.alpha = alpha
        self.beta = beta
        self.prior = prior
        self.t_min = t_min
        self.epsilon = 1e-10
        self.A = np.log((1 - beta) / alpha) + np.log(prior / (1 - prior))
        self.B = np.log(beta / (1 - alpha)) + np.log(prior / (1 - prior))

    def decide_from_probs(self, probs_dict, max_length=128):
        for t in sorted(probs_dict.keys()):
            if t < self.t_min:
                continue
            p_t = probs_dict[t]
            S_t = np.log((p_t + self.epsilon) / (1 - p_t + self.epsilon))
            if S_t >= self.A:
                return "UNSAFE", t
            elif S_t <= self.B:
                return "SAFE", t
        return "SAFE", max_length


class ConfidenceDetector:
    def __init__(self, threshold=0.95, t_min=5):
        self.threshold = threshold
        self.t_min = t_min

    def decide_from_probs(self, probs_dict, max_length=128):
        for t in sorted(probs_dict.keys()):
            if t < self.t_min:
                continue
            p_t = probs_dict[t]
            if p_t >= self.threshold:
                return "UNSAFE", t
            elif p_t <= (1 - self.threshold):
                return "SAFE", t
        return "SAFE", max_length


# ============================================================
# Evaluation Utilities
# ============================================================
def evaluate_predictions(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {
        'f1': round(float(f1), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'fpr': round(float(fpr), 4),
        'fnr': round(float(fnr), 4),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return round(float(ece / len(probs)), 4)


def run_detector_on_precomputed(detector, all_probs, max_length=128):
    decisions, stopping_times = [], []
    for i in sorted(all_probs.keys()):
        probs = all_probs[i]['probs']
        decision, stop = detector.decide_from_probs(probs, max_length)
        decisions.append(1 if decision == "UNSAFE" else 0)
        stopping_times.append(stop)
    return decisions, stopping_times


def precompute_all_probs(model, tokenizer, texts, device, max_length=128, t_min=5):
    print(f"\nPre-computing prefix probabilities for {len(texts)} texts...")
    all_probs = {}
    full_probs = []
    for i, text in enumerate(tqdm(texts, desc="Computing probs")):
        probs, n_tok = get_prefix_probs(model, tokenizer, text, device, max_length, chunk_k=1, t_min=t_min)
        all_probs[i] = {'probs': probs, 'n_tokens': n_tok}
        full_p = get_classifier_prob(model, tokenizer, text, device, max_length)
        full_probs.append(full_p)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(texts)}]")
    return all_probs, np.array(full_probs)


# ============================================================
# Experiment 5: Calibration Sensitivity (FIXED)
# ============================================================
def exp5_calibration_sensitivity(all_probs, labels):
    print("\n" + "=" * 60)
    print("Experiment 5: Calibration Sensitivity")
    print("=" * 60)

    target_eces = [0.0, 0.05, 0.10, 0.15, 0.20]
    results = []

    for target_ece in target_eces:
        perturbed_probs = {}
        for i in all_probs:
            perturbed_probs[i] = {'probs': {}, 'n_tokens': all_probs[i]['n_tokens']}
            for t, p in all_probs[i]['probs'].items():
                if target_ece == 0:
                    perturbed_probs[i]['probs'][t] = p
                else:
                    noise = np.random.normal(0, target_ece * 0.5)
                    p_perturbed = np.clip(p + noise, 0.001, 0.999)
                    perturbed_probs[i]['probs'][t] = p_perturbed

        det = WaldThresholdDetector(alpha=0.05, beta=0.10, prior=0.01)
        preds, stops = run_detector_on_precomputed(det, perturbed_probs)
        metrics = evaluate_predictions(labels, preds)

        # FIX: Guard against empty dicts
        full_perturbed = []
        for i in sorted(perturbed_probs.keys()):
            keys = list(perturbed_probs[i]['probs'].keys())
            if keys:
                full_perturbed.append(perturbed_probs[i]['probs'][max(keys)])
            else:
                full_perturbed.append(0.5)
        full_perturbed = np.array(full_perturbed)
        actual_ece = compute_ece(full_perturbed, np.array(labels))

        row = {
            'target_ece': target_ece,
            'actual_ece': actual_ece,
            'f1': metrics['f1'],
            'fpr': metrics['fpr'],
            'fnr': metrics['fnr'],
            'accuracy': metrics['accuracy'],
        }
        results.append(row)
        print(f"  TargetECE={target_ece:.0%}: ActualECE={actual_ece:.4f}, "
              f"F1={metrics['f1']:.4f}, FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}")
    return results


# ============================================================
# Experiment 6: Wall-Clock Latency
# ============================================================
def exp6_latency(model, tokenizer, texts, device):
    print("\n" + "=" * 60)
    print("Experiment 6: Wall-Clock Latency Measurement")
    print("=" * 60)

    n_measure = min(100, len(texts))
    measure_texts = texts[:n_measure]

    # Warmup
    for _ in range(5):
        get_classifier_prob(model, tokenizer, measure_texts[0], device)

    # Single classification latency
    times = []
    for text in measure_texts:
        start = time.perf_counter()
        get_classifier_prob(model, tokenizer, text, device)
        times.append(time.perf_counter() - start)
    single_cls_ms = round(np.mean(times) * 1000, 2)
    single_cls_std = round(np.std(times) * 1000, 2)

    # Streaming detection latency
    det = WaldThresholdDetector(alpha=0.05, beta=0.10, prior=0.01)
    stream_times = []
    n_calls_list = []
    for text in measure_texts:
        start = time.perf_counter()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = min(len(tokens), 128)
        n_calls = 0
        for t in range(5, n_tokens + 1):
            prefix = tokenizer.decode(tokens[:t], skip_special_tokens=True)
            p_t = get_classifier_prob(model, tokenizer, prefix, device)
            n_calls += 1
            S_t = np.log((p_t + 1e-10) / (1 - p_t + 1e-10))
            if S_t >= det.A or S_t <= det.B:
                break
        stream_times.append(time.perf_counter() - start)
        n_calls_list.append(n_calls)

    avg_stream_ms = round(np.mean(stream_times) * 1000, 2)
    avg_calls = round(np.mean(n_calls_list), 1)

    llm_ms_per_token = 50
    full_text_llm_ms = 128 * llm_ms_per_token
    sprt_llm_ms = avg_calls * llm_ms_per_token

    results = {
        'classifier_per_call_ms': single_cls_ms,
        'classifier_std_ms': single_cls_std,
        'sprt_avg_calls': avg_calls,
        'sprt_total_classify_ms': round(avg_calls * single_cls_ms, 1),
        'llm_ms_per_token': llm_ms_per_token,
        'full_text_total_ms': round(full_text_llm_ms + single_cls_ms, 1),
        'sprt_total_ms_estimate': round(sprt_llm_ms + avg_calls * single_cls_ms, 1),
        'speedup_estimate': round(
            (full_text_llm_ms + single_cls_ms) / (sprt_llm_ms + avg_calls * single_cls_ms), 2
        ),
    }
    print(f"  Classifier per call: {single_cls_ms}ms (std: {single_cls_std}ms)")
    print(f"  SPRT avg calls: {avg_calls}")
    print(f"  Estimated speedup: {results['speedup_estimate']}x")
    return results


# ============================================================
# Experiment 7: Chunking Strategy
# ============================================================
def exp7_chunking(model, tokenizer, texts, labels, device):
    print("\n" + "=" * 60)
    print("Experiment 7: Chunking Strategy")
    print("=" * 60)

    n_eval = min(500, len(texts))
    eval_texts = texts[:n_eval]
    eval_labels = labels[:n_eval]

    results = []
    for k in [1, 3, 5, 10]:
        decisions, stops, n_calls_list = [], [], []
        det = WaldThresholdDetector(alpha=0.05, beta=0.10, prior=0.01)

        for text in tqdm(eval_texts, desc=f"Chunk k={k}"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            n_tokens = min(len(tokens), 128)
            decision, stop, n_calls = "SAFE", n_tokens, 0

            for t in range(5, n_tokens + 1, k):
                prefix = tokenizer.decode(tokens[:t], skip_special_tokens=True)
                p_t = get_classifier_prob(model, tokenizer, prefix, device)
                n_calls += 1
                S_t = np.log((p_t + 1e-10) / (1 - p_t + 1e-10))
                if S_t >= det.A:
                    decision, stop = "UNSAFE", t
                    break
                elif S_t <= det.B:
                    decision, stop = "SAFE", t
                    break

            decisions.append(1 if decision == "UNSAFE" else 0)
            stops.append(stop)
            n_calls_list.append(n_calls)

        metrics = evaluate_predictions(eval_labels, decisions)
        row = {
            'k': k,
            'f1': metrics['f1'],
            'fpr': metrics['fpr'],
            'fnr': metrics['fnr'],
            'accuracy': metrics['accuracy'],
            'avg_calls': round(float(np.mean(n_calls_list)), 1),
            'avg_stop': round(float(np.mean(stops)), 1),
            'token_savings': round(1 - np.mean(stops) / 128, 4),
        }
        results.append(row)
        print(f"  k={k}: F1={metrics['f1']:.4f}, FPR={metrics['fpr']:.4f}, "
              f"AvgCalls={row['avg_calls']}, Savings={row['token_savings']:.1%}")
    return results


# ============================================================
# Experiment 8: Error Rate Ablation
# ============================================================
def exp8_error_rates(all_probs, labels):
    print("\n" + "=" * 60)
    print("Experiment 8: Error Rate Ablation")
    print("=" * 60)

    configs = [(0.01, 0.05), (0.05, 0.10), (0.10, 0.15), (0.20, 0.20)]
    results = []

    for alpha, beta in configs:
        A_raw = np.log((1 - beta) / alpha)
        B_raw = np.log(beta / (1 - alpha))
        for prior in [0.50, 0.01]:
            det = WaldThresholdDetector(alpha=alpha, beta=beta, prior=prior)
            preds, stops = run_detector_on_precomputed(det, all_probs)
            metrics = evaluate_predictions(labels, preds)
            avg_stop = np.mean(stops)
            row = {
                'alpha': alpha, 'beta': beta, 'prior': prior,
                'A_raw': round(float(A_raw), 2),
                'B_raw': round(float(B_raw), 2),
                'A_adjusted': round(float(det.A), 2),
                'B_adjusted': round(float(det.B), 2),
                'fpr': metrics['fpr'], 'fnr': metrics['fnr'],
                'f1': metrics['f1'], 'accuracy': metrics['accuracy'],
                'avg_tokens': round(float(avg_stop), 1),
            }
            results.append(row)
            print(f"  (α={alpha},β={beta},π={prior}): FPR={metrics['fpr']:.4f}, "
                  f"FNR={metrics['fnr']:.4f}, F1={metrics['f1']:.4f}, AvgTok={avg_stop:.1f}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Running Experiments 5-8 (with fix)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {CONFIG['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    model, tokenizer = load_model()
    texts, labels = load_data(tokenizer, max_samples=CONFIG['max_samples'])
    labels_array = np.array(labels)

    # Load existing results if available
    existing_path = '/root/autodl-tmp/experiment_results.json'
    existing = {}
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        print(f"Loaded existing results from {existing_path}")
        print(f"  Has exp1: {'exp1_wald_bounds' in existing}")
        print(f"  Has exp2: {'exp2_pareto' in existing}")
        print(f"  Has exp3: {'exp3_prior_ablation' in existing}")

    # Pre-compute probabilities
    all_probs, full_probs_arr = precompute_all_probs(
        model, tokenizer, texts, CONFIG['device'],
        max_length=CONFIG['max_length'], t_min=CONFIG['t_min']
    )

    # Run experiments 5-8
    results = existing.copy() if existing else {}
    results['exp5_calibration_sensitivity'] = exp5_calibration_sensitivity(all_probs, labels_array)
    results['exp6_latency'] = exp6_latency(model, tokenizer, texts, CONFIG['device'])
    results['exp7_chunking'] = exp7_chunking(model, tokenizer, texts, labels_array, CONFIG['device'])
    results['exp8_error_rates'] = exp8_error_rates(all_probs, labels_array)
    results['exp5_8_timestamp'] = datetime.now().isoformat()

    # Save
    with open(existing_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nAll results saved to {existing_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Exp 5-8)")
    print("=" * 60)
    print("\nExp5 (Calibration Sensitivity):")
    for r in results['exp5_calibration_sensitivity']:
        print(f"  TargetECE={r['target_ece']:.0%}: F1={r['f1']:.4f}, FPR={r['fpr']:.4f}")
    print("\nExp6 (Latency):")
    lat = results['exp6_latency']
    print(f"  Classifier: {lat['classifier_per_call_ms']}ms/call")
    print(f"  SPRT avg calls: {lat['sprt_avg_calls']}")
    print(f"  Speedup: {lat['speedup_estimate']}x")
    print("\nExp7 (Chunking):")
    for r in results['exp7_chunking']:
        print(f"  k={r['k']}: F1={r['f1']:.4f}, FPR={r['fpr']:.4f}, Savings={r['token_savings']:.1%}")
    print("\nExp8 (Error Rates):")
    for r in results['exp8_error_rates']:
        print(f"  (α={r['alpha']},β={r['beta']},π={r['prior']}): F1={r['f1']:.4f}")

    return results


if __name__ == "__main__":
    main()
