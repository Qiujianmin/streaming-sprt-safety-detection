#!/usr/bin/env python3
"""
Supplemental Experiments for ACM TIST Major Revision

This script addresses the three main reviewer concerns:
1. Comparison with Confidence-based Early Exit
2. Wall-Clock Time and FLOPs Measurement
3. Calibration Sensitivity Analysis

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
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar

# ============================================================
# Experiment 1: Wall-Clock Time Measurement
# ============================================================

class LatencyMeasurer:
    """Measure actual wall-clock latency for different methods"""

    def __init__(self, model, tokenizer, device='cuda', max_length=128, n_warmup=100):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.n_warmup = n_warmup

        # Warm up GPU
        print(f"Warming up GPU with {n_warmup} inferences...")
        dummy_text = "This is a warmup sentence for the model."
        for _ in range(n_warmup):
            self._single_inference(dummy_text)
        torch.cuda.synchronize()
        print("Warmup complete.")

    def _single_inference(self, text):
        inputs = self.tokenizer(
            text, return_tensors='pt',
            truncation=True, max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

        return prob

    def measure_full_text(self, texts, n_runs=5):
        """Measure full-text classification latency"""
        times = []
        call_counts = []

        for run in range(n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            for text in texts:
                self._single_inference(text)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed / len(texts) * 1000)  # ms per sample
            call_counts.append(1)

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'classifier_calls_per_sample': 1.0,
            'method': 'Full-text'
        }

    def measure_sprt(self, texts, alpha=0.05, beta=0.10, t_min=5, prior=0.01, n_runs=5):
        """Measure SPRT latency with token-by-token checking"""
        A = np.log((1 - beta) / alpha)
        B = np.log(beta / (1 - alpha))

        if prior != 0.5:
            prior_log_odds = np.log(prior / (1 - prior))
            A += prior_log_odds
            B += prior_log_odds

        all_times = []
        all_stops = []
        all_calls = []

        for run in range(n_runs):
            run_times = []
            run_stops = []
            run_calls = []

            for text in texts:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                n_tokens = min(len(tokens), self.max_length)

                torch.cuda.synchronize()
                start = time.perf_counter()

                calls = 0
                stop_time = n_tokens

                for t in range(t_min, n_tokens + 1):
                    current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                    prob = self._single_inference(current_text)
                    calls += 1

                    S_t = np.log(prob / (1 - prob + 1e-10))

                    if S_t >= A or S_t <= B:
                        stop_time = t
                        break

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                run_times.append(elapsed * 1000)  # ms
                run_stops.append(stop_time)
                run_calls.append(calls)

            all_times.append(np.mean(run_times))
            all_stops.append(np.mean(run_stops))
            all_calls.append(np.mean(run_calls))

        return {
            'mean_ms': np.mean(all_times),
            'std_ms': np.std(all_times),
            'avg_stopping_time': np.mean(all_stops),
            'classifier_calls_per_sample': np.mean(all_calls),
            'method': f'SPRT(π={prior})'
        }

    def measure_sprt_chunked(self, texts, chunk_size=5, alpha=0.05, beta=0.10,
                              t_min=5, prior=0.01, n_runs=5):
        """Measure SPRT latency with chunking (check every k tokens)"""
        A = np.log((1 - beta) / alpha)
        B = np.log(beta / (1 - alpha))

        if prior != 0.5:
            prior_log_odds = np.log(prior / (1 - prior))
            A += prior_log_odds
            B += prior_log_odds

        all_times = []
        all_stops = []
        all_calls = []

        for run in range(n_runs):
            run_times = []
            run_stops = []
            run_calls = []

            for text in texts:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                n_tokens = min(len(tokens), self.max_length)

                torch.cuda.synchronize()
                start = time.perf_counter()

                calls = 0
                stop_time = n_tokens

                # Check at chunk intervals
                for t in range(t_min, n_tokens + 1, chunk_size):
                    current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                    prob = self._single_inference(current_text)
                    calls += 1

                    S_t = np.log(prob / (1 - prob + 1e-10))

                    if S_t >= A:
                        stop_time = t
                        break
                    elif S_t <= B:
                        stop_time = t
                        break

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                run_times.append(elapsed * 1000)
                run_stops.append(stop_time)
                run_calls.append(calls)

            all_times.append(np.mean(run_times))
            all_stops.append(np.mean(run_stops))
            all_calls.append(np.mean(run_calls))

        return {
            'mean_ms': np.mean(all_times),
            'std_ms': np.std(all_times),
            'avg_stopping_time': np.mean(all_stops),
            'classifier_calls_per_sample': np.mean(all_calls),
            'method': f'SPRT-chunked(k={chunk_size})'
        }

    def measure_confidence_based(self, texts, threshold=0.95, t_min=5, n_runs=5):
        """Measure confidence-based early exit latency"""
        all_times = []
        all_stops = []
        all_calls = []

        for run in range(n_runs):
            run_times = []
            run_stops = []
            run_calls = []

            for text in texts:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                n_tokens = min(len(tokens), self.max_length)

                torch.cuda.synchronize()
                start = time.perf_counter()

                calls = 0
                stop_time = n_tokens

                for t in range(t_min, n_tokens + 1):
                    current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                    prob = self._single_inference(current_text)
                    calls += 1

                    # Confidence-based: stop when confident
                    if prob >= threshold or prob <= (1 - threshold):
                        stop_time = t
                        break

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                run_times.append(elapsed * 1000)
                run_stops.append(stop_time)
                run_calls.append(calls)

            all_times.append(np.mean(run_times))
            all_stops.append(np.mean(run_stops))
            all_calls.append(np.mean(run_calls))

        return {
            'mean_ms': np.mean(all_times),
            'std_ms': np.std(all_times),
            'avg_stopping_time': np.mean(all_stops),
            'classifier_calls_per_sample': np.mean(all_calls),
            'method': f'Confidence(th={threshold})'
        }

    def measure_fixed_window(self, texts, window_size=10, n_runs=5):
        """Measure fixed-window latency"""
        all_times = []
        all_stops = []
        all_calls = []

        for run in range(n_runs):
            run_times = []
            run_stops = []
            run_calls = []

            for text in texts:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                n_tokens = min(len(tokens), self.max_length)

                torch.cuda.synchronize()
                start = time.perf_counter()

                calls = 0
                stop_time = n_tokens

                for t in range(window_size, n_tokens + 1, window_size):
                    current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                    prob = self._single_inference(current_text)
                    calls += 1

                    if prob >= 0.5:
                        stop_time = t
                        break

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                run_times.append(elapsed * 1000)
                run_stops.append(stop_time)
                run_calls.append(calls)

            all_times.append(np.mean(run_times))
            all_stops.append(np.mean(run_stops))
            all_calls.append(np.mean(run_calls))

        return {
            'mean_ms': np.mean(all_times),
            'std_ms': np.std(all_times),
            'avg_stopping_time': np.mean(all_stops),
            'classifier_calls_per_sample': np.mean(all_calls),
            'method': f'Fixed-W{window_size}'
        }


# ============================================================
# Experiment 2: FLOPs Estimation
# ============================================================

def estimate_flops(seq_length, model_params=125e6, hidden_dim=768, num_layers=12):
    """
    Estimate FLOPs for RoBERTa-base forward pass

    Approximation: FLOPs ≈ 2 × params × seq_length (for transformer)
    More accurate: Include attention (seq_length^2) and FFN components
    """
    # Simplified estimation
    # Attention: 4 × hidden_dim × seq_length^2 per layer
    # FFN: 8 × hidden_dim^2 × seq_length per layer

    attention_flops = 4 * hidden_dim * seq_length * seq_length * num_layers
    ffn_flops = 8 * hidden_dim * hidden_dim * seq_length * num_layers
    embedding_flops = 2 * model_params * seq_length * 0.1  # Approximate

    total_flops = attention_flops + ffn_flops + embedding_flops

    # Alternative simpler estimation
    simple_flops = 2 * model_params * seq_length

    return {
        'detailed_flops': total_flops,
        'simple_flops': simple_flops,
        'attention_flops': attention_flops,
        'ffn_flops': ffn_flops
    }


def compute_total_flops_for_method(method_name, avg_calls, avg_seq_length, max_length=128):
    """Compute total FLOPs for a method"""
    if 'Full-text' in method_name:
        flops = estimate_flops(max_length)
        return flops['detailed_flops']

    elif 'SPRT' in method_name:
        # Average over increasing prefix lengths
        total_flops = 0
        for i in range(int(avg_calls)):
            seq_len = int(max_length * (i + 1) / (avg_calls + 1))
            flops = estimate_flops(seq_len)
            total_flops += flops['detailed_flops']
        return total_flops

    elif 'Fixed' in method_name:
        # Check at fixed intervals
        total_flops = 0
        for i in range(int(avg_calls)):
            seq_len = max_length * (i + 1) // int(avg_calls + 1)
            flops = estimate_flops(seq_len)
            total_flops += flops['detailed_flops']
        return total_flops

    return 0


# ============================================================
# Experiment 3: Calibration Sensitivity Analysis
# ============================================================

class CalibrationSensitivityAnalyzer:
    """Analyze how calibration error affects SPRT performance"""

    def __init__(self, model, tokenizer, device='cuda', max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.epsilon = 1e-10

    def add_calibration_error(self, probs, target_ece, labels):
        """
        Add perturbation to simulate calibration error

        Strategy: Randomly shift probabilities towards/away from 0.5
        """
        np.random.seed(42)
        perturbed = probs.copy()

        for i in range(len(probs)):
            # Direction: towards incorrect prediction increases ECE
            if labels[i] == 1:
                # Should be high, reduce to simulate underconfidence
                shift = np.random.uniform(0, target_ece)
                perturbed[i] = max(0.01, probs[i] - shift)
            else:
                # Should be low, increase to simulate overconfidence
                shift = np.random.uniform(0, target_ece)
                perturbed[i] = min(0.99, probs[i] + shift)

        return perturbed

    def compute_ece(self, probs, labels, n_bins=10):
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

    def run_sprt_with_probs(self, texts, probs_per_text, labels, alpha=0.05, beta=0.10,
                            t_min=5, prior=0.01):
        """Run SPRT using pre-computed probabilities (with added calibration error)"""
        A = np.log((1 - beta) / alpha)
        B = np.log(beta / (1 - alpha))

        if prior != 0.5:
            prior_log_odds = np.log(prior / (1 - prior))
            A += prior_log_odds
            B += prior_log_odds

        decisions = []
        stopping_times = []

        for text_idx, text in enumerate(texts):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            n_tokens = min(len(tokens), self.max_length)

            decision = "SAFE"
            stop_time = n_tokens

            # Simulate SPRT with perturbed probabilities
            for t in range(t_min, n_tokens + 1):
                # Get pre-computed probability (with calibration error)
                prob = probs_per_text[text_idx]

                S_t = np.log(prob / (1 - prob + self.epsilon))

                if S_t >= A:
                    decision = "UNSAFE"
                    stop_time = t
                    break
                elif S_t <= B:
                    decision = "SAFE"
                    stop_time = t
                    break

            decisions.append(decision)
            stopping_times.append(stop_time)

        # Compute metrics
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
            'accuracy': accuracy,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'avg_stopping_time': np.mean(stopping_times)
        }

    def run_sensitivity_analysis(self, texts, labels, target_eces=[0.0, 0.05, 0.10, 0.15, 0.20]):
        """Run full sensitivity analysis across different ECE levels"""
        # First, get original probabilities
        print("Computing original probabilities...")
        original_probs = []
        for text in tqdm(texts):
            inputs = self.tokenizer(
                text, return_tensors='pt',
                truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            original_probs.append(prob)

        original_probs = np.array(original_probs)
        original_ece = self.compute_ece(original_probs, labels)

        results = {
            'original_ece': original_ece,
            'sensitivity': {}
        }

        for target_ece in target_eces:
            print(f"\nAnalyzing ECE = {target_ece:.2f}...")

            if target_ece == 0.0:
                # Use original probabilities
                perturbed_probs = original_probs
                actual_ece = original_ece
            else:
                # Add calibration error
                perturbed_probs = self.add_calibration_error(
                    original_probs, target_ece, labels
                )
                actual_ece = self.compute_ece(perturbed_probs, labels)

            # Run SPRT
            sprt_results = self.run_sprt_with_probs(
                texts, perturbed_probs, labels
            )

            results['sensitivity'][target_ece] = {
                'target_ece': target_ece,
                'actual_ece': actual_ece,
                **sprt_results
            }

            print(f"  Actual ECE: {actual_ece:.4f}")
            print(f"  F1: {sprt_results['f1']:.4f}, FPR: {sprt_results['fpr']:.4f}, "
                  f"FNR: {sprt_results['fnr']:.4f}")

        return results


# ============================================================
# Main Experiment Runner
# ============================================================

def run_supplemental_experiments(
    model_path,
    test_texts,
    test_labels,
    output_dir='./results/supplemental',
    device='cuda',
    n_samples=1000,
    n_runs=5
):
    """Run all supplemental experiments"""

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Supplemental Experiments for ACM TIST Major Revision")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Subsample if needed
    if len(test_texts) > n_samples:
        np.random.seed(42)
        indices = np.random.choice(len(test_texts), n_samples, replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]

    all_results = {}

    # ============================================================
    # Experiment 1: Wall-Clock Time Measurement
    # ============================================================
    print("\n" + "=" * 70)
    print("Experiment 1: Wall-Clock Time Measurement")
    print("=" * 70)

    measurer = LatencyMeasurer(model, tokenizer, device)

    latency_results = {}

    # Full-text
    print("\n[1/6] Measuring Full-text latency...")
    latency_results['Full-text'] = measurer.measure_full_text(test_texts, n_runs)

    # SPRT
    print("\n[2/6] Measuring SPRT latency...")
    latency_results['SPRT(π=0.01)'] = measurer.measure_sprt(test_texts, prior=0.01, n_runs=n_runs)

    # SPRT chunked
    print("\n[3/6] Measuring SPRT-chunked(k=5) latency...")
    latency_results['SPRT-chunked(k=5)'] = measurer.measure_sprt_chunked(test_texts, chunk_size=5, n_runs=n_runs)

    print("\n[4/6] Measuring SPRT-chunked(k=10) latency...")
    latency_results['SPRT-chunked(k=10)'] = measurer.measure_sprt_chunked(test_texts, chunk_size=10, n_runs=n_runs)

    # Confidence-based
    print("\n[5/6] Measuring Confidence-based latency...")
    latency_results['Confidence(th=0.95)'] = measurer.measure_confidence_based(test_texts, threshold=0.95, n_runs=n_runs)

    # Fixed-window
    print("\n[6/6] Measuring Fixed-W10 latency...")
    latency_results['Fixed-W10'] = measurer.measure_fixed_window(test_texts, window_size=10, n_runs=n_runs)

    all_results['latency'] = latency_results

    # Print latency summary
    print("\n" + "-" * 70)
    print("Latency Summary:")
    print("-" * 70)
    print(f"{'Method':<30} {'Mean (ms)':>12} {'Std (ms)':>10} {'Calls':>8}")
    print("-" * 70)
    for method, res in latency_results.items():
        print(f"{method:<30} {res['mean_ms']:>12.2f} {res['std_ms']:>10.2f} {res['classifier_calls_per_sample']:>8.1f}")

    # ============================================================
    # Experiment 2: FLOPs Estimation
    # ============================================================
    print("\n" + "=" * 70)
    print("Experiment 2: FLOPs Estimation")
    print("=" * 70)

    flops_results = {}
    for method, res in latency_results.items():
        avg_calls = res['classifier_calls_per_sample']
        avg_seq = res.get('avg_stopping_time', 128)

        flops = compute_total_flops_for_method(method, avg_calls, avg_seq)
        flops_results[method] = {
            'total_flops': flops,
            'flops_per_sample': flops,
            'relative_to_fulltext': flops / flops_results.get('Full-text', {}).get('total_flops', flops)
        }

    all_results['flops'] = flops_results

    print("\n" + "-" * 70)
    print("FLOPs Summary:")
    print("-" * 70)
    print(f"{'Method':<30} {'GFLOPs':>12} {'Relative':>10}")
    print("-" * 70)
    for method, res in flops_results.items():
        print(f"{method:<30} {res['total_flops']/1e9:>12.2f} {res['relative_to_fulltext']:>10.2f}x")

    # ============================================================
    # Experiment 3: Calibration Sensitivity
    # ============================================================
    print("\n" + "=" * 70)
    print("Experiment 3: Calibration Sensitivity Analysis")
    print("=" * 70)

    analyzer = CalibrationSensitivityAnalyzer(model, tokenizer, device)
    sensitivity_results = analyzer.run_sensitivity_analysis(
        test_texts, test_labels,
        target_eces=[0.0, 0.05, 0.10, 0.15, 0.20]
    )

    all_results['calibration_sensitivity'] = sensitivity_results

    print("\n" + "-" * 70)
    print("Calibration Sensitivity Summary:")
    print("-" * 70)
    print(f"{'Target ECE':>12} {'Actual ECE':>12} {'F1':>8} {'FPR':>8} {'FNR':>8}")
    print("-" * 70)
    for target_ece, res in sensitivity_results['sensitivity'].items():
        print(f"{target_ece:>12.2f} {res['actual_ece']:>12.4f} {res['f1']:>8.4f} "
              f"{res['fpr']:>8.4f} {res['fnr']:>8.4f}")

    # ============================================================
    # Save Results
    # ============================================================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    output_file = os.path.join(output_dir, 'supplemental_experiments_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")

    # Generate LaTeX tables
    print("\n" + "=" * 70)
    print("LaTeX Tables for Paper")
    print("=" * 70)

    # Table: Wall-Clock Latency
    print("\n% Table: Wall-Clock Latency Comparison")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Wall-Clock Latency Comparison}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Method & Latency (ms) & Calls & Speedup \\\\")
    print("\\midrule")
    baseline = latency_results['Full-text']['mean_ms']
    for method, res in latency_results.items():
        speedup = baseline / res['mean_ms']
        print(f"{method} & {res['mean_ms']:.1f} $\\pm$ {res['std_ms']:.1f} & "
              f"{res['classifier_calls_per_sample']:.1f} & {speedup:.2f}$\\times$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    return all_results


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supplemental Experiments')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--test_texts', type=str, nargs='+',
                        help='Test texts (or pass via stdin)')
    parser.add_argument('--test_labels', type=str, nargs='+',
                        help='Test labels (0/1)')
    parser.add_argument('--output_dir', type=str, default='./results/supplemental',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to test')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of runs for latency measurement')

    args = parser.parse_args()

    # For demo, create dummy data if not provided
    if args.test_texts is None:
        print("No test data provided, using dummy data for demonstration...")
        test_texts = ["This is a sample text."] * 100
        test_labels = [0, 1] * 50
    else:
        test_texts = args.test_texts
        test_labels = [int(l) for l in args.test_labels]

    run_supplemental_experiments(
        model_path=args.model_path,
        test_texts=test_texts,
        test_labels=test_labels,
        output_dir=args.output_dir,
        device=args.device,
        n_samples=args.n_samples,
        n_runs=args.n_runs
    )
