#!/usr/bin/env python3
"""
简化版补充实验脚本 - CPU兼容版本

运行Wall-Clock时间测量和校准敏感性分析
注意：此脚本在CPU上运行，速度较慢但可以完成

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
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============================================================
# 配置
# ============================================================

EXPERIMENT_CONFIG = {
    "n_samples": 200,  # CPU版本减少样本数
    "n_runs": 3,
    "alpha": 0.05,
    "beta": 0.10,
    "t_min": 5,
    "max_length": 128,
}


# ============================================================
# Wall-Clock 时间测量
# ============================================================

class LatencyMeasurer:
    """测量实际wall-clock延迟"""

    def __init__(self, model, tokenizer, device='cpu', max_length=128, n_warmup=20):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        # Warmup
        print(f"Warming up with {n_warmup} inferences...")
        dummy_text = "This is a warmup sentence for the model."
        for _ in range(n_warmup):
            self._single_inference(dummy_text)
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

    def measure_full_text(self, texts, n_runs=3):
        """测量完整文本分类延迟"""
        times = []

        for run in range(n_runs):
            start = time.perf_counter()

            for text in texts:
                self._single_inference(text)

            elapsed = time.perf_counter() - start
            times.append(elapsed / len(texts) * 1000)  # ms per sample

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'classifier_calls_per_sample': 1.0,
            'method': 'Full-text'
        }

    def measure_sprt(self, texts, alpha=0.05, beta=0.10, t_min=5, prior=0.01, n_runs=3):
        """测量SPRT延迟"""
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

            for text in tqdm(texts, desc=f"SPRT run {run+1}"):
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                n_tokens = min(len(tokens), self.max_length)

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
                              t_min=5, prior=0.01, n_runs=3):
        """测量分块SPRT延迟"""
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

            for text in tqdm(texts, desc=f"SPRT-chunked(k={chunk_size}) run {run+1}"):
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                n_tokens = min(len(tokens), self.max_length)

                start = time.perf_counter()

                calls = 0
                stop_time = n_tokens

                for t in range(t_min, n_tokens + 1, chunk_size):
                    current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                    prob = self._single_inference(current_text)
                    calls += 1

                    S_t = np.log(prob / (1 - prob + 1e-10))

                    if S_t >= A or S_t <= B:
                        stop_time = t
                        break

                elapsed = time.perf_counter() - start

                run_times.append(elapsed * 1000)
                run_stops.append(stop_time)
                run_calls.append(calls)

            all_times.append(np.mean(run_times))
            all_stops.append(np.mean(run_stops))
            all_calls.append(np.mean(all_calls))

        return {
            'mean_ms': np.mean(all_times),
            'std_ms': np.std(all_times),
            'avg_stopping_time': np.mean(all_stops),
            'classifier_calls_per_sample': np.mean(all_calls),
            'method': f'SPRT-chunked(k={chunk_size})'
        }

    def measure_confidence_based(self, texts, threshold=0.95, t_min=5, n_runs=3):
        """测量置信度早退延迟"""
        all_times = []
        all_stops = []
        all_calls = []

        for run in range(n_runs):
            run_times = []
            run_stops = []
            run_calls = []

            for text in tqdm(texts, desc=f"Confidence(th={threshold}) run {run+1}"):
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                n_tokens = min(len(tokens), self.max_length)

                start = time.perf_counter()

                calls = 0
                stop_time = n_tokens

                for t in range(t_min, n_tokens + 1):
                    current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                    prob = self._single_inference(current_text)
                    calls += 1

                    if prob >= threshold or prob <= (1 - threshold):
                        stop_time = t
                        break

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


# ============================================================
# 校准敏感性分析
# ============================================================

class CalibrationSensitivityAnalyzer:
    """分析校准误差对SPRT性能的影响"""

    def __init__(self, model, tokenizer, device='cpu', max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def add_calibration_error(self, probs, target_ece, labels):
        """添加扰动模拟校准误差"""
        np.random.seed(42)
        perturbed = probs.copy()

        for i in range(len(probs)):
            if labels[i] == 1:
                shift = np.random.uniform(0, target_ece * 0.5)
                perturbed[i] = max(0.01, probs[i] - shift)
            else:
                shift = np.random.uniform(0, target_ece * 0.5)
                perturbed[i] = min(0.99, probs[i] + shift)

        return perturbed

    def compute_ece(self, probs, labels, n_bins=10):
        """计算ECE"""
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

    def run_sprt_with_probs(self, probs_list, labels, alpha=0.05, beta=0.10, prior=0.01):
        """使用预计算概率运行SPRT"""
        A = np.log((1 - beta) / alpha)
        B = np.log(beta / (1 - alpha))

        if prior != 0.5:
            prior_log_odds = np.log(prior / (1 - prior))
            A += prior_log_odds
            B += prior_log_odds

        decisions = []
        stopping_times = []

        for prob in probs_list:
            decision = "SAFE"
            stop_time = 1

            S_t = np.log(prob / (1 - prob + 1e-10))

            if S_t >= A:
                decision = "UNSAFE"
            elif S_t <= B:
                decision = "SAFE"

            decisions.append(decision)
            stopping_times.append(stop_time)

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
        }

    def run_sensitivity_analysis(self, texts, labels, target_eces=[0.0, 0.05, 0.10, 0.15, 0.20]):
        """运行完整的敏感性分析"""
        # 获取原始概率
        print("Computing original probabilities...")
        original_probs = []

        for text in tqdm(texts, desc="Getting probs"):
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
                perturbed_probs = original_probs
                actual_ece = original_ece
            else:
                perturbed_probs = self.add_calibration_error(
                    original_probs, target_ece, labels
                )
                actual_ece = self.compute_ece(perturbed_probs, labels)

            sprt_results = self.run_sprt_with_probs(
                perturbed_probs, labels
            )

            results['sensitivity'][target_ece] = {
                'target_ece': target_ece,
                'actual_ece': actual_ece,
                **sprt_results
            }

            print(f"  Actual ECE: {actual_ece:.4f}")
            print(f"  F1: {sprt_results['f1']:.4f}, FPR: {sprt_results['fpr']:.4f}")

        return results


# ============================================================
# 主函数
# ============================================================

def create_dummy_data(n_samples=200):
    """创建模拟数据用于测试"""
    np.random.seed(42)

    # 创建模拟文本
    safe_templates = [
        "I think this is a great product and I love using it.",
        "The weather today is beautiful and sunny.",
        "Thank you for your help with this project.",
        "This article provides useful information about the topic.",
        "I agree with your point about the importance of education.",
    ]

    toxic_templates = [
        "This is absolutely disgusting and you should be ashamed.",
        "People like you are the problem with society today.",
        "I hate everything about this stupid idea.",
        "This is the worst thing I have ever seen in my life.",
        "You are completely wrong and your opinion is worthless.",
    ]

    texts = []
    labels = []

    for i in range(n_samples):
        if np.random.random() < 0.92:  # 8% toxicity rate like CivilComments
            texts.append(np.random.choice(safe_templates))
            labels.append(0)
        else:
            texts.append(np.random.choice(toxic_templates))
            labels.append(1)

    return texts, labels


def run_experiments(model_path, output_dir='./results/supplemental', device='cpu'):
    """运行所有补充实验"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Supplemental Experiments for ACM TIST Major Revision")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # 加载模型
    print("\nLoading model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from: {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}, using untrained model")

    model.to(device)
    model.eval()

    # 创建测试数据
    print("\nCreating test data...")
    test_texts, test_labels = create_dummy_data(EXPERIMENT_CONFIG['n_samples'])
    print(f"Test samples: {len(test_texts)}")
    print(f"Toxicity rate: {sum(test_labels)/len(test_labels)*100:.1f}%")

    all_results = {}

    # ========================================
    # 实验1: Wall-Clock时间测量
    # ========================================
    print("\n" + "=" * 70)
    print("Experiment 1: Wall-Clock Time Measurement")
    print("=" * 70)

    measurer = LatencyMeasurer(model, tokenizer, device)

    latency_results = {}

    print("\n[1/5] Measuring Full-text latency...")
    latency_results['Full-text'] = measurer.measure_full_text(
        test_texts, n_runs=EXPERIMENT_CONFIG['n_runs']
    )

    print("\n[2/5] Measuring SPRT latency...")
    latency_results['SPRT(π=0.01)'] = measurer.measure_sprt(
        test_texts, prior=0.01, n_runs=EXPERIMENT_CONFIG['n_runs']
    )

    print("\n[3/5] Measuring SPRT-chunked(k=5) latency...")
    latency_results['SPRT-chunked(k=5)'] = measurer.measure_sprt_chunked(
        test_texts, chunk_size=5, n_runs=EXPERIMENT_CONFIG['n_runs']
    )

    print("\n[4/5] Measuring Confidence-based latency...")
    latency_results['Confidence(th=0.95)'] = measurer.measure_confidence_based(
        test_texts, threshold=0.95, n_runs=EXPERIMENT_CONFIG['n_runs']
    )

    all_results['latency'] = latency_results

    # 打印延迟摘要
    print("\n" + "-" * 70)
    print("Latency Summary:")
    print("-" * 70)
    print(f"{'Method':<30} {'Mean (ms)':>12} {'Std (ms)':>10} {'Calls':>8}")
    print("-" * 70)
    for method, res in latency_results.items():
        print(f"{method:<30} {res['mean_ms']:>12.2f} {res['std_ms']:>10.2f} "
              f"{res['classifier_calls_per_sample']:>8.1f}")

    # ========================================
    # 实验2: 校准敏感性分析
    # ========================================
    print("\n" + "=" * 70)
    print("Experiment 2: Calibration Sensitivity Analysis")
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

    # ========================================
    # 保存结果
    # ========================================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    output_file = os.path.join(output_dir, 'supplemental_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")

    # 生成LaTeX表格
    print("\n" + "=" * 70)
    print("LaTeX Tables for Paper")
    print("=" * 70)

    print("\n% Table: Wall-Clock Latency Comparison")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Wall-Clock Latency Comparison (CPU, 200 samples)}")
    print("\\label{tab:latency}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Method & Latency (ms) & Calls & Speedup \\\\")
    print("\\midrule")
    baseline = latency_results['Full-text']['mean_ms']
    for method, res in latency_results.items():
        speedup = baseline / res['mean_ms'] if res['mean_ms'] > 0 else 0
        print(f"{method} & {res['mean_ms']:.1f} $\\pm$ {res['std_ms']:.1f} & "
              f"{res['classifier_calls_per_sample']:.1f} & {speedup:.2f}$\\times$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n% Table: Calibration Sensitivity")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Calibration Sensitivity Analysis}")
    print("\\label{tab:calibration_sensitivity}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Target ECE & Actual ECE & F1 & FPR & FNR \\\\")
    print("\\midrule")
    for target_ece, res in sensitivity_results['sensitivity'].items():
        print(f"{target_ece:.2f} & {res['actual_ece']:.4f} & {res['f1']:.4f} & "
              f"{res['fpr']:.4f} & {res['fnr']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supplemental Experiments (CPU)')
    parser.add_argument('--model_path', type=str,
                        default='../_backup/streaming_sprt_paper/models/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results/supplemental',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()

    run_experiments(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )
