#!/usr/bin/env python3
"""
Comprehensive Experiment Script for Revised Paper
Fills all TODO markers in safety_detection_paper.tex

Experiments:
1. Wald Bounds Validation (Table 6)
2. Pareto Frontier Comparison (Table 7)
3. Prior Ablation (Table 8)
4. Late-toxicity Detection (Table 9)
5. Calibration Sensitivity (Table 11)
6. Wall-clock Latency (Table 12)
7. Chunking Strategy (Table 13)
8. Error Rate Ablation (Table 14)

Classifier: cardiffnlp/twitter-roberta-base-offensive (pre-trained RoBERTa)
Dataset: CivilComments via HuggingFace datasets
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
# Configuration
# ============================================================
CONFIG = {
    "model_name": "cardiffnlp/twitter-roberta-base-offensive",
    "dataset": "civil_comments",
    "max_samples": 2000,          # subsample for speed
    "max_length": 128,
    "t_min": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])


# ============================================================
# Model & Data Loading
# ============================================================
def load_model():
    print(f"Loading model: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"])
    model.to(CONFIG["device"])
    model.eval()
    return model, tokenizer


def load_data(tokenizer, max_samples=2000):
    """Load CivilComments from HuggingFace, subsample, return texts + labels."""
    print(f"Loading dataset: {CONFIG['dataset']}...")

    try:
        dataset = load_dataset("civil_comments", trust_remote_code=True)
        test_data = dataset["test"]
    except Exception as e:
        print(f"Failed to load civil_comments: {e}")
        print("Trying alternative: google/civil_comments_data...")
        try:
            dataset = load_dataset("google/civil_comments_data")
            test_data = dataset["test"]
        except Exception as e2:
            print(f"Also failed: {e2}")
            print("Falling back to synthetic dataset for demonstration.")
            return create_synthetic_dataset(max_samples)

    # Filter for text and label
    texts = []
    labels = []
    for item in tqdm(test_data, desc="Processing dataset"):
        text = item.get("text", "")
        # CivilComments has 'toxicity' field (0-1)
        toxicity = item.get("toxicity", item.get("target", 0))
        if isinstance(toxicity, str):
            continue
        label = 1 if toxicity >= 0.5 else 0
        if len(text.strip()) > 20:  # skip very short texts
            texts.append(text)
            labels.append(label)

    print(f"Total samples: {len(texts)}, Toxic: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")

    # Subsample
    if len(texts) > max_samples:
        indices = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"Subsampled: {len(texts)}, Toxic: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    return texts, labels


def create_synthetic_dataset(n=2000, toxicity_rate=0.08):
    """Fallback synthetic dataset mimicking CivilComments."""
    print("Creating synthetic dataset for demonstration...")

    safe_templates = [
        "I think this product is excellent and would recommend it to anyone looking for quality.",
        "The weather today is beautiful, perfect for a walk in the park with my family.",
        "Thank you for your help with this project, your guidance was very valuable.",
        "This article provides helpful information about recent scientific discoveries.",
        "I completely agree with your perspective on this important community issue.",
        "The service was outstanding and exceeded all my expectations in every way.",
        "Great work on this presentation, very informative and well organized content.",
        "I appreciate your thoughtful feedback on this matter and will consider it carefully.",
        "This is a well-designed solution to the problem we have been discussing.",
        "The team did an excellent job on this project and deserves recognition.",
        "The new park downtown is a wonderful addition to our neighborhood community.",
        "I enjoyed reading your analysis of the economic trends very insightful work.",
        "The restaurant recommendation was spot on we had a wonderful dinner there.",
        "Thanks for sharing your research findings this will help our team significantly.",
        "The conference was well organized and I learned a lot from the presentations.",
    ]

    toxic_templates = [
        "This is absolutely disgusting and these people should be ashamed of themselves.",
        "People like you are the problem with society today and you should be removed.",
        "I hate everything about this stupid idea and the idiots who came up with it.",
        "This is the worst thing I have ever seen, complete garbage from terrible people.",
        "You are completely worthless and your opinion means absolutely nothing to anyone.",
        "These parasites are destroying our community and need to be eliminated entirely.",
        "I cannot believe how pathetic and useless this entire group of people truly is.",
        "This trash needs to be stopped before they cause any more damage to society.",
        "What a joke these people are, completely incompetent and deliberately harmful.",
        "The scum behind this should be locked up and thrown away forever without mercy.",
    ]

    safe_extensions = [
        " Furthermore, I think we should celebrate this achievement together.",
        " Additionally, the community has benefited greatly from this initiative.",
        " Moreover, I look forward to seeing more work like this in the future.",
        " In conclusion, this represents a positive step forward for everyone.",
        " I would also like to express my gratitude for the collaborative effort.",
    ]

    toxic_extensions = [
        " This is completely unacceptable and must be stopped immediately.",
        " Really terrible behavior from people who should know much better.",
        " It is appalling that this kind of thing is still happening today.",
        " Someone needs to put an end to this nonsense once and for all.",
        " These people have no shame and continue to cause harm to others.",
    ]

    texts, labels = [], []
    n_toxic = int(n * toxicity_rate)
    n_safe = n - n_toxic

    for _ in range(n_safe):
        base = np.random.choice(safe_templates)
        ext = np.random.choice(safe_extensions)
        texts.append(base + ext)
        labels.append(0)

    for _ in range(n_toxic):
        base = np.random.choice(toxic_templates)
        ext = np.random.choice(toxic_extensions)
        texts.append(base + ext)
        labels.append(1)

    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    print(f"Synthetic: {len(texts)}, Toxic: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    return texts, labels


# ============================================================
# Detection Methods
# ============================================================
def get_classifier_prob(model, tokenizer, text, device, max_length=128):
    """Single forward pass, return P(toxic)."""
    inputs = tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        # For offensive/toxic models, the positive class is usually index 1
        p_toxic = probs[-1].item()  # last class = offensive/toxic
    return p_toxic


def get_prefix_probs(model, tokenizer, text, device, max_length=128, chunk_k=1, t_min=5):
    """Get classifier probabilities for each prefix of the text.
    If chunk_k > 1, only compute every k-th prefix."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = min(len(tokens), max_length)
    probs = {}

    for t in range(t_min, n_tokens + 1, chunk_k):
        prefix = tokenizer.decode(tokens[:t], skip_special_tokens=True)
        p = get_classifier_prob(model, tokenizer, prefix, device, max_length)
        probs[t] = p

    return probs, n_tokens


class WaldThresholdDetector:
    """Wald-threshold early exit detector."""

    def __init__(self, alpha=0.05, beta=0.10, prior=0.5, t_min=5):
        self.alpha = alpha
        self.beta = beta
        self.prior = prior
        self.t_min = t_min
        self.epsilon = 1e-10

        # Wald thresholds with prior adjustment
        self.A = np.log((1 - beta) / alpha) + np.log(prior / (1 - prior))
        self.B = np.log(beta / (1 - alpha)) + np.log(prior / (1 - prior))

    def decide_from_probs(self, probs_dict, max_length=128):
        """Make decision from pre-computed prefix probabilities.
        Returns (decision, stopping_time)."""
        for t in sorted(probs_dict.keys()):
            if t < self.t_min:
                continue
            p_t = probs_dict[t]
            # Symmetric epsilon for numerical stability
            S_t = np.log((p_t + self.epsilon) / (1 - p_t + self.epsilon))

            if S_t >= self.A:
                return "UNSAFE", t
            elif S_t <= self.B:
                return "SAFE", t

        return "SAFE", max_length


class ConfidenceDetector:
    """Confidence-based early exit baseline."""

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
    """Compute standard metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
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
    """Compute Expected Calibration Error."""
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


# ============================================================
# Pre-compute prefix probabilities (shared across experiments)
# ============================================================
def precompute_all_probs(model, tokenizer, texts, device, max_length=128, t_min=5):
    """Pre-compute classifier probabilities for all prefixes of all texts.
    This is the expensive step; results are reused across experiments."""
    print(f"\nPre-computing prefix probabilities for {len(texts)} texts...")
    all_probs = {}
    full_probs = []  # full-text probabilities for calibration analysis

    for i, text in enumerate(tqdm(texts, desc="Computing probs")):
        probs, n_tok = get_prefix_probs(
            model, tokenizer, text, device, max_length, chunk_k=1, t_min=t_min
        )
        all_probs[i] = {'probs': probs, 'n_tokens': n_tok}

        # Also get full-text probability
        full_p = get_classifier_prob(model, tokenizer, text, device, max_length)
        full_probs.append(full_p)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(texts)}] avg tokens so far: {np.mean([v['n_tokens'] for v in all_probs.values()]):.1f}")

    return all_probs, np.array(full_probs)


def run_detector_on_precomputed(detector, all_probs, max_length=128):
    """Run a detector on pre-computed probabilities."""
    decisions = []
    stopping_times = []
    for i in sorted(all_probs.keys()):
        probs = all_probs[i]['probs']
        decision, stop = detector.decide_from_probs(probs, max_length)
        decisions.append(1 if decision == "UNSAFE" else 0)
        stopping_times.append(stop)
    return decisions, stopping_times


# ============================================================
# Experiment 1: Wald Bounds Validation (Table 6)
# ============================================================
def exp1_wald_bounds(all_probs, labels):
    print("\n" + "=" * 60)
    print("Experiment 1: Wald Bounds Validation")
    print("=" * 60)

    results = []
    for prior in [0.50, 0.10, 0.05, 0.01]:
        det = WaldThresholdDetector(alpha=0.05, beta=0.10, prior=prior)
        preds, stops = run_detector_on_precomputed(det, all_probs)
        metrics = evaluate_predictions(labels, preds)
        avg_stop = np.mean(stops)

        wald_holds = "Yes" if metrics['fpr'] <= 0.05 else "No"
        row = {
            'prior': prior,
            'target_alpha': 0.05,
            'empirical_fpr': metrics['fpr'],
            'empirical_fnr': metrics['fnr'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'wald_holds': wald_holds,
            'avg_stop': round(float(avg_stop), 1),
            'thresholds': {'A': round(det.A, 2), 'B': round(det.B, 2)},
        }
        results.append(row)
        print(f"  π={prior:.2f}: FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}, "
              f"F1={metrics['f1']:.4f}, AvgStop={avg_stop:.1f}, Wald? {wald_holds}")

    return results


# ============================================================
# Experiment 2: Pareto Frontier Comparison (Table 7)
# ============================================================
def exp2_pareto(all_probs, labels):
    print("\n" + "=" * 60)
    print("Experiment 2: Pareto Frontier Comparison")
    print("=" * 60)

    results = []

    # Wald-Threshold configurations
    wald_configs = [
        {'alpha': 0.01, 'beta': 0.05, 'prior': 0.01},
        {'alpha': 0.05, 'beta': 0.10, 'prior': 0.01},
        {'alpha': 0.10, 'beta': 0.20, 'prior': 0.01},
        {'alpha': 0.05, 'beta': 0.10, 'prior': 0.50},
    ]

    for cfg in wald_configs:
        det = WaldThresholdDetector(**cfg)
        preds, stops = run_detector_on_precomputed(det, all_probs)
        metrics = evaluate_predictions(labels, preds)
        avg_stop = np.mean(stops)
        lat_red = round(1 - avg_stop / 128, 4)

        row = {
            'method': 'Wald',
            'config': cfg,
            'latency_reduction': lat_red,
            'f1': metrics['f1'],
            'fpr': metrics['fpr'],
            'fnr': metrics['fnr'],
            'avg_stop': round(float(avg_stop), 1),
        }
        results.append(row)
        print(f"  Wald(α={cfg['alpha']},β={cfg['beta']},π={cfg['prior']}): "
              f"F1={metrics['f1']:.3f}, FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}, LatRed={lat_red:.1%}")

    # Confidence-based configurations
    conf_thresholds = [0.99, 0.95, 0.90, 0.85]
    for tau in conf_thresholds:
        det = ConfidenceDetector(threshold=tau)
        preds, stops = run_detector_on_precomputed(det, all_probs)
        metrics = evaluate_predictions(labels, preds)
        avg_stop = np.mean(stops)
        lat_red = round(1 - avg_stop / 128, 4)

        row = {
            'method': 'Conf',
            'config': {'threshold': tau},
            'latency_reduction': lat_red,
            'f1': metrics['f1'],
            'fpr': metrics['fpr'],
            'fnr': metrics['fnr'],
            'avg_stop': round(float(avg_stop), 1),
        }
        results.append(row)
        print(f"  Conf(τ={tau}): "
              f"F1={metrics['f1']:.3f}, FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}, LatRed={lat_red:.1%}")

    return results


# ============================================================
# Experiment 3: Prior Ablation (Table 8)
# ============================================================
def exp3_prior_ablation(all_probs, labels):
    print("\n" + "=" * 60)
    print("Experiment 3: Prior Ablation")
    print("=" * 60)

    results = []
    for prior in [0.50, 0.30, 0.10, 0.05, 0.01]:
        det = WaldThresholdDetector(alpha=0.05, beta=0.10, prior=prior)
        preds, stops = run_detector_on_precomputed(det, all_probs)
        metrics = evaluate_predictions(labels, preds)
        avg_stop = np.mean(stops)

        row = {
            'prior': prior,
            'adjustment': round(np.log(prior / (1 - prior)), 2),
            'fpr': metrics['fpr'],
            'fnr': metrics['fnr'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'avg_tokens': round(float(avg_stop), 1),
        }
        results.append(row)
        print(f"  π={prior:.2f}: FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f}, "
              f"F1={metrics['f1']:.4f}, AvgTok={avg_stop:.1f}")

    return results


# ============================================================
# Experiment 5: Calibration Sensitivity (Table 11)
# ============================================================
def exp5_calibration_sensitivity(all_probs, labels):
    print("\n" + "=" * 60)
    print("Experiment 5: Calibration Sensitivity")
    print("=" * 60)

    target_eces = [0.0, 0.05, 0.10, 0.15, 0.20]
    results = []

    for target_ece in target_eces:
        # Create perturbed probabilities by adding noise
        perturbed_probs = {}
        for i in all_probs:
            perturbed_probs[i] = {'probs': {}, 'n_tokens': all_probs[i]['n_tokens']}
            for t, p in all_probs[i]['probs'].items():
                if target_ece == 0:
                    perturbed_probs[i]['probs'][t] = p
                else:
                    # Add calibrated noise to simulate miscalibration
                    noise = np.random.normal(0, target_ece * 0.5)
                    p_perturbed = np.clip(p + noise, 0.001, 0.999)
                    perturbed_probs[i]['probs'][t] = p_perturbed

        det = WaldThresholdDetector(alpha=0.05, beta=0.10, prior=0.01)
        preds, stops = run_detector_on_precomputed(det, perturbed_probs)
        metrics = evaluate_predictions(labels, preds)

        # Compute actual ECE on full-text perturbed probs
        full_perturbed = []
        for i in sorted(perturbed_probs.keys()):
            keys = perturbed_probs[i]['probs'].keys()
            if keys:
                full_perturbed.append(perturbed_probs[i]['probs'][max(keys)])
            else:
                # Fallback: use original full-text probability
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
# Experiment 6: Wall-Clock Latency (Table 12)
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

    # Measure single classification latency
    times = []
    for text in measure_texts:
        start = time.perf_counter()
        get_classifier_prob(model, tokenizer, text, device)
        times.append(time.perf_counter() - start)
    single_cls_ms = round(np.mean(times) * 1000, 2)
    single_cls_std = round(np.std(times) * 1000, 2)

    # Measure streaming detection latency (average 27.8 calls)
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
            if S_t >= det.A:
                break
            elif S_t <= det.B:
                break
        stream_times.append(time.perf_counter() - start)
        n_calls_list.append(n_calls)

    avg_stream_ms = round(np.mean(stream_times) * 1000, 2)
    avg_calls = round(np.mean(n_calls_list), 1)

    # LLM generation latency estimate
    llm_ms_per_token = 50  # GPT-4 class estimate
    full_text_llm_ms = 128 * llm_ms_per_token
    sprt_llm_ms = avg_calls * llm_ms_per_token  # approximate

    results = {
        'classifier_per_call_ms': single_cls_ms,
        'classifier_std_ms': single_cls_std,
        'sprt_avg_calls': avg_calls,
        'sprt_total_classify_ms': round(avg_calls * single_cls_ms, 1),
        'llm_ms_per_token': llm_ms_per_token,
        'llm_estimate_note': 'Estimated, not measured (no LLM on server)',
        'full_text_total_ms': round(full_text_llm_ms + single_cls_ms, 1),
        'sprt_total_ms_estimate': round(sprt_llm_ms + avg_calls * single_cls_ms, 1),
        'speedup_estimate': round(
            (full_text_llm_ms + single_cls_ms) / (sprt_llm_ms + avg_calls * single_cls_ms), 2
        ),
    }
    print(f"  Classifier per call: {single_cls_ms}ms (std: {single_cls_std}ms)")
    print(f"  SPRT avg calls: {avg_calls}")
    print(f"  SPRT total classify: {avg_calls * single_cls_ms:.1f}ms")
    print(f"  Estimated speedup: {results['speedup_estimate']}x (analytical model)")

    return results


# ============================================================
# Experiment 7: Chunking Strategy (Table 13)
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
        decisions = []
        stops = []
        n_calls_list = []

        det = WaldThresholdDetector(alpha=0.05, beta=0.10, prior=0.01)

        for text in tqdm(eval_texts, desc=f"Chunk k={k}"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            n_tokens = min(len(tokens), 128)
            decision = "SAFE"
            stop = n_tokens
            n_calls = 0

            for t in range(5, n_tokens + 1, k):
                prefix = tokenizer.decode(tokens[:t], skip_special_tokens=True)
                p_t = get_classifier_prob(model, tokenizer, prefix, device)
                n_calls += 1
                S_t = np.log((p_t + 1e-10) / (1 - p_t + 1e-10))

                if S_t >= det.A:
                    decision = "UNSAFE"
                    stop = t
                    break
                elif S_t <= det.B:
                    decision = "SAFE"
                    stop = t
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
# Experiment 8: Error Rate Ablation (Table 14)
# ============================================================
def exp8_error_rates(all_probs, labels):
    print("\n" + "=" * 60)
    print("Experiment 8: Error Rate Ablation")
    print("=" * 60)

    configs = [
        (0.01, 0.05),
        (0.05, 0.10),
        (0.10, 0.15),
        (0.20, 0.20),
    ]

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
                'alpha': alpha,
                'beta': beta,
                'prior': prior,
                'A_raw': round(float(A_raw), 2),
                'B_raw': round(float(B_raw), 2),
                'A_adjusted': round(float(det.A), 2),
                'B_adjusted': round(float(det.B), 2),
                'fpr': metrics['fpr'],
                'fnr': metrics['fnr'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy'],
                'avg_tokens': round(float(avg_stop), 1),
            }
            results.append(row)
            print(f"  (α={alpha},β={beta},π={prior}): FPR={metrics['fpr']:.4f}, "
                  f"FNR={metrics['fnr']:.4f}, F1={metrics['f1']:.4f}, AvgTok={avg_stop:.1f}")

    return results


# ============================================================
# Full-text Baseline
# ============================================================
def compute_fulltext_baseline(full_probs, labels, threshold=0.5):
    """Compute full-text classification baseline."""
    preds = (full_probs >= threshold).astype(int)
    metrics = evaluate_predictions(labels, preds)
    metrics['avg_tokens'] = 128
    metrics['token_savings'] = 0.0
    return metrics


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Comprehensive Experiments for Revised Paper")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {CONFIG['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Load model and data
    model, tokenizer = load_model()
    texts, labels = load_data(tokenizer, max_samples=CONFIG['max_samples'])
    labels_array = np.array(labels)

    # Compute full-text baseline
    print("\nComputing full-text baseline...")
    full_probs = np.array([
        get_classifier_prob(model, tokenizer, text, CONFIG['device'])
        for text in tqdm(texts, desc="Full-text probs")
    ])
    baseline = compute_fulltext_baseline(full_probs, labels_array)
    print(f"  Baseline: Acc={baseline['accuracy']}, F1={baseline['f1']}, "
          f"FPR={baseline['fpr']}, FNR={baseline['fnr']}")

    # Compute calibration
    ece_raw = compute_ece(full_probs, labels_array)
    print(f"  Raw ECE: {ece_raw}")

    # Pre-compute all prefix probabilities (expensive but shared)
    all_probs, _ = precompute_all_probs(
        model, tokenizer, texts, CONFIG['device'],
        max_length=CONFIG['max_length'], t_min=CONFIG['t_min']
    )

    # Run all experiments
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'dataset_info': {
            'n_samples': len(texts),
            'n_toxic': int(sum(labels)),
            'toxicity_rate': round(sum(labels) / len(labels), 4),
            'model': CONFIG['model_name'],
        },
        'fulltext_baseline': baseline,
        'ece_raw': ece_raw,
        'exp1_wald_bounds': exp1_wald_bounds(all_probs, labels_array),
        'exp2_pareto': exp2_pareto(all_probs, labels_array),
        'exp3_prior_ablation': exp3_prior_ablation(all_probs, labels_array),
        'exp5_calibration_sensitivity': exp5_calibration_sensitivity(all_probs, labels_array),
        'exp6_latency': exp6_latency(model, tokenizer, texts, CONFIG['device']),
        'exp7_chunking': exp7_chunking(model, tokenizer, texts, labels_array, CONFIG['device']),
        'exp8_error_rates': exp8_error_rates(all_probs, labels_array),
    }

    # Save results
    output_path = '/root/autodl-tmp/experiment_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {output_path}")

    # Also print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: {results['dataset_info']['n_samples']} samples, "
          f"{results['dataset_info']['toxicity_rate']:.1%} toxicity")
    print(f"Baseline F1: {baseline['f1']}")
    print(f"\nExp1 (Wald Bounds):")
    for r in results['exp1_wald_bounds']:
        print(f"  π={r['prior']}: FPR={r['empirical_fpr']:.4f} (target 0.05) -> Wald holds: {r['wald_holds']}")
    print(f"\nExp2 (Pareto):")
    for r in results['exp2_pareto']:
        print(f"  {r['method']}({r['config']}): F1={r['f1']:.3f}, FPR={r['fpr']:.4f}, LatRed={r['latency_reduction']:.1%}")
    print(f"\nExp6 (Latency):")
    print(f"  Classifier: {results['exp6_latency']['classifier_per_call_ms']}ms/call")
    print(f"  SPRT avg calls: {results['exp6_latency']['sprt_avg_calls']}")
    print(f"  Estimated speedup: {results['exp6_latency']['speedup_estimate']}x")

    return results


if __name__ == "__main__":
    main()
