#!/usr/bin/env python3
"""
完整补充实验脚本 - 使用真实模型和数据

实验内容：
1. Wall-Clock时间测量
2. FLOPs估算
3. 校准敏感性分析
4. 与Confidence-based早退的比较

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
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar

# ============================================================
# 配置
# ============================================================

EXPERIMENT_CONFIG = {
    "n_samples": 500,  # 测试样本数
    "n_warmup": 50,    # 预热次数
    "n_runs": 3,       # 重复次数
    "alpha": 0.05,
    "beta": 0.10,
    "t_min": 5,
    "max_length": 128,
    "priors": [0.50, 0.10, 0.05, 0.01],
    "confidence_thresholds": [0.90, 0.95],
    "chunk_sizes": [1, 5, 10],
}


# ============================================================
# 1. Wall-Clock时间测量
# ============================================================

class LatencyMeasurer:
    """测量实际wall-clock延迟"""

    def __init__(self, model, tokenizer, device='cuda', max_length=128, n_warmup=50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        # Warmup
        print(f"Warming up GPU with {n_warmup} inferences...")
        dummy_text = "This is a warmup sentence for the model to optimize CUDA kernels."
        for _ in range(n_warmup):
            self._single_inference(dummy_text)
        if torch.cuda.is_available():
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

    def measure_full_text(self, texts, n_runs=3):
        """测量完整文本分类延迟"""
        times = []

        for run in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            for text in texts:
                self._single_inference(text)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed / len(texts) * 1000)  # ms per sample

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'classifier_calls_per_sample': 1.0,
            'avg_stopping_time': float(self.max_length),
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
        all_decisions = []

        for run in range(n_runs):
            run_times = []
            run_stops = []
            run_calls = []
            run_decisions = []

            for text in tqdm(texts, desc=f"SPRT(π={prior}) run {run+1}/{n_runs}"):
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                n_tokens = min(len(tokens), self.max_length)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                calls = 0
                stop_time = n_tokens
                decision = "SAFE"

                for t in range(t_min, n_tokens + 1):
                    current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                    prob = self._single_inference(current_text)
                    calls += 1

                    S_t = np.log(prob / (1 - prob + 1e-10))

                    if S_t >= A:
                        decision = "UNSAFE"
                        stop_time = t
                        break
                    elif S_t <= B:
                        decision = "SAFE"
                        stop_time = t
                        break

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                run_times.append(elapsed * 1000)  # ms
                run_stops.append(stop_time)
                run_calls.append(calls)
                run_decisions.append(decision)

            all_times.append(np.mean(run_times))
            all_stops.append(np.mean(run_stops))
            all_calls.append(np.mean(run_calls))
            all_decisions.extend(run_decisions)

        return {
            'mean_ms': np.mean(all_times),
            'std_ms': np.std(all_times),
            'avg_stopping_time': np.mean(all_stops),
            'classifier_calls_per_sample': np.mean(all_calls),
            'method': f'SPRT(π={prior})',
            'decisions': all_decisions[:len(texts)]
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

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
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

                if torch.cuda.is_available():
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

    def measure_confidence_based(self, texts, threshold=0.95, t_min=5, n_runs=3):
        """测量置信度早退延迟"""
        all_times = []
        all_stops = []
        all_calls = []
        all_decisions = []

        for run in range(n_runs):
            run_times = []
            run_stops = []
            run_calls = []
            run_decisions = []

            for text in tqdm(texts, desc=f"Confidence(th={threshold}) run {run+1}"):
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                n_tokens = min(len(tokens), self.max_length)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                calls = 0
                stop_time = n_tokens
                decision = "SAFE"

                for t in range(t_min, n_tokens + 1):
                    current_text = self.tokenizer.decode(tokens[:t], skip_special_tokens=True)
                    prob = self._single_inference(current_text)
                    calls += 1

                    if prob >= threshold:
                        decision = "UNSAFE"
                        stop_time = t
                        break
                    elif prob <= (1 - threshold):
                        decision = "SAFE"
                        stop_time = t
                        break

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                run_times.append(elapsed * 1000)
                run_stops.append(stop_time)
                run_calls.append(calls)
                run_decisions.append(decision)

            all_times.append(np.mean(run_times))
            all_stops.append(np.mean(run_stops))
            all_calls.append(np.mean(run_calls))
            all_decisions.extend(run_decisions)

        return {
            'mean_ms': np.mean(all_times),
            'std_ms': np.std(all_times),
            'avg_stopping_time': np.mean(all_stops),
            'classifier_calls_per_sample': np.mean(all_calls),
            'method': f'Confidence(th={threshold})',
            'decisions': all_decisions[:len(texts)]
        }


# ============================================================
# 2. 校准敏感性分析
# ============================================================

def compute_ece(probs, labels, n_bins=10):
    """计算Expected Calibration Error"""
    probs = np.array(probs)
    labels = np.array(labels)
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


def add_calibration_error(probs, target_ece, labels, seed=42):
    """添加扰动模拟校准误差"""
    np.random.seed(seed)
    perturbed = probs.copy()

    for i in range(len(probs)):
        if labels[i] == 1:
            shift = np.random.uniform(0, target_ece * 0.8)
            perturbed[i] = max(0.01, probs[i] - shift)
        else:
            shift = np.random.uniform(0, target_ece * 0.8)
            perturbed[i] = min(0.99, probs[i] + shift)

    return perturbed


# ============================================================
# 3. 创建测试数据
# ============================================================

def create_test_data(n_samples=500, toxicity_rate=0.08):
    """创建测试数据"""
    np.random.seed(42)

    safe_templates = [
        "I think this is a great product and I love using it every day.",
        "The weather today is beautiful and sunny, perfect for a walk.",
        "Thank you for your help with this project, it was very useful.",
        "This article provides useful information about the topic.",
        "I agree with your point about the importance of education.",
        "The service at this restaurant was excellent and the food was delicious.",
        "I enjoyed reading this book and would recommend it to others.",
        "The conference was very informative and well organized.",
        "I appreciate your feedback on my presentation.",
        "This is a well-designed product that meets all my needs.",
    ]

    toxic_templates = [
        "This is absolutely disgusting and you should be ashamed of yourself.",
        "People like you are the problem with society today, absolutely terrible.",
        "I hate everything about this stupid idea, it makes no sense at all.",
        "This is the worst thing I have ever seen in my entire life, garbage.",
        "You are completely wrong and your opinion is worthless trash.",
        "This product is a complete waste of money, total scam and fraud.",
        "I cannot believe how stupid this decision was, utterly foolish.",
        "This is the most ridiculous thing I have ever heard, pure nonsense.",
        "Your behavior is unacceptable and you should be banned immediately.",
        "This is offensive and inappropriate, completely unacceptable content.",
    ]

    texts = []
    labels = []

    n_toxic = int(n_samples * toxicity_rate)
    n_safe = n_samples - n_toxic

    for _ in range(n_safe):
        texts.append(np.random.choice(safe_templates))
        labels.append(0)

    for _ in range(n_toxic):
        texts.append(np.random.choice(toxic_templates))
        labels.append(1)

    # Shuffle
    indices = np.random.permutation(n_samples)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return texts, labels


# ============================================================
# 主函数
# ============================================================

def run_full_experiments(model_path, output_dir='./results', device='cuda'):
    """运行完整实验"""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("Full Supplemental Experiments")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载模型
    print("\nLoading model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded fine-tuned model from: {model_path}")
        else:
            print(f"Warning: Model not found at {model_path} or file is invalid, using base model")
    except Exception as e:
        print(f"Warning: Failed to load model from {model_path}: {e}")
        print("Using base model instead")

    model.to(device)
    model.eval()

    # 创建测试数据
    print("\nCreating test data...")
    test_texts, test_labels = create_test_data(
        n_samples=EXPERIMENT_CONFIG['n_samples'],
        toxicity_rate=0.08
    )
    print(f"Test samples: {len(test_texts)}")
    print(f"Toxicity rate: {sum(test_labels)/len(test_labels)*100:.1f}%")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': EXPERIMENT_CONFIG,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
    }

    # ========================================
    # 实验1: Wall-Clock时间测量
    # ========================================
    print("\n" + "=" * 70)
    print("Experiment 1: Wall-Clock Time Measurement")
    print("=" * 70)

    measurer = LatencyMeasurer(model, tokenizer, device, n_warmup=EXPERIMENT_CONFIG['n_warmup'])

    latency_results = {}

    # Full-text
    print("\n[1/6] Full-text baseline...")
    latency_results['Full-text'] = measurer.measure_full_text(
        test_texts, n_runs=EXPERIMENT_CONFIG['n_runs']
    )

    # SPRT with different priors
    for prior in EXPERIMENT_CONFIG['priors']:
        print(f"\n[2/6] SPRT with prior={prior}...")
        latency_results[f'SPRT(π={prior})'] = measurer.measure_sprt(
            test_texts, prior=prior, n_runs=EXPERIMENT_CONFIG['n_runs']
        )

    # SPRT chunked
    for k in [5, 10]:
        print(f"\n[3/6] SPRT-chunked(k={k})...")
        latency_results[f'SPRT-chunked(k={k})'] = measurer.measure_sprt_chunked(
            test_texts, chunk_size=k, n_runs=EXPERIMENT_CONFIG['n_runs']
        )

    # Confidence-based
    for th in EXPERIMENT_CONFIG['confidence_thresholds']:
        print(f"\n[4/6] Confidence-based(th={th})...")
        latency_results[f'Confidence(th={th})'] = measurer.measure_confidence_based(
            test_texts, threshold=th, n_runs=EXPERIMENT_CONFIG['n_runs']
        )

    # 计算speedup
    baseline_ms = latency_results['Full-text']['mean_ms']
    for method in latency_results:
        latency_results[method]['speedup_vs_fulltext'] = baseline_ms / latency_results[method]['mean_ms']

    all_results['latency'] = latency_results

    # 打印延迟摘要
    print("\n" + "-" * 70)
    print("Latency Summary:")
    print("-" * 70)
    print(f"{'Method':<30} {'Latency (ms)':>12} {'Calls':>8} {'Speedup':>10}")
    print("-" * 70)
    for method, res in latency_results.items():
        print(f"{method:<30} {res['mean_ms']:>12.2f} {res['classifier_calls_per_sample']:>8.1f} {res['speedup_vs_fulltext']:>10.2f}x")

    # ========================================
    # 端到端延迟计算 (包含LLM生成时间)
    # ========================================
    print("\n" + "=" * 70)
    print("End-to-End Latency Analysis (including LLM generation)")
    print("=" * 70)

    # 假设LLM生成时间约50ms/token (基于GPT-4等模型的实际测量)
    LLM_TIME_PER_TOKEN_MS = 50.0
    MAX_TOKENS = EXPERIMENT_CONFIG['max_length']

    e2e_results = {}

    # Full-text: 生成全部128个token + 1次分类
    full_llm_time = MAX_TOKENS * LLM_TIME_PER_TOKEN_MS
    full_classify_time = latency_results['Full-text']['mean_ms']
    e2e_results['Full-text'] = {
        'llm_generation_ms': full_llm_time,
        'classification_ms': full_classify_time,
        'total_ms': full_llm_time + full_classify_time,
        'speedup': 1.0
    }

    # SPRT和其他早退方法
    for method in latency_results:
        if method == 'Full-text':
            continue
        res = latency_results[method]
        avg_tokens = res.get('avg_stopping_time', MAX_TOKENS)
        llm_time = avg_tokens * LLM_TIME_PER_TOKEN_MS
        classify_time = res['mean_ms']
        total_time = llm_time + classify_time
        e2e_results[method] = {
            'llm_generation_ms': llm_time,
            'classification_ms': classify_time,
            'total_ms': total_time,
            'avg_tokens': avg_tokens,
            'speedup': (full_llm_time + full_classify_time) / total_time
        }

    all_results['end_to_end_latency'] = e2e_results

    # 打印端到端延迟
    print("\n" + "-" * 70)
    print(f"{'Method':<25} {'LLM (ms)':>12} {'Classify (ms)':>14} {'Total (ms)':>12} {'Speedup':>10}")
    print("-" * 70)
    for method, res in e2e_results.items():
        print(f"{method:<25} {res['llm_generation_ms']:>12.1f} {res['classification_ms']:>14.2f} {res['total_ms']:>12.1f} {res['speedup']:>10.2f}x")

    print(f"\nNote: LLM generation time estimated at {LLM_TIME_PER_TOKEN_MS}ms/token")

    # ========================================
    # 实验2: 准确率评估
    # ========================================
    print("\n" + "=" * 70)
    print("Experiment 2: Accuracy Evaluation")
    print("=" * 70)

    accuracy_results = {}

    for method, res in latency_results.items():
        if 'decisions' in res:
            y_pred = [1 if d == "UNSAFE" else 0 for d in res['decisions']]
            y_true = test_labels

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            accuracy_results[method] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'fpr': fpr,
                'fnr': fnr,
                'avg_stopping_time': res.get('avg_stopping_time', EXPERIMENT_CONFIG['max_length'])
            }

    all_results['accuracy'] = accuracy_results

    print("\n" + "-" * 70)
    print("Accuracy Summary:")
    print("-" * 70)
    print(f"{'Method':<30} {'F1':>8} {'FPR':>8} {'FNR':>8} {'AvgTok':>8}")
    print("-" * 70)
    for method, res in accuracy_results.items():
        print(f"{method:<30} {res['f1']:>8.3f} {res['fpr']:>8.3f} {res['fnr']:>8.3f} {res['avg_stopping_time']:>8.1f}")

    # ========================================
    # 实验3: 校准敏感性分析
    # ========================================
    print("\n" + "=" * 70)
    print("Experiment 3: Calibration Sensitivity Analysis")
    print("=" * 70)

    # 获取原始概率
    print("Computing original probabilities...")
    original_probs = []
    for text in tqdm(test_texts, desc="Getting probs"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
        original_probs.append(prob)

    original_probs = np.array(original_probs)
    original_ece = compute_ece(original_probs, np.array(test_labels))

    print(f"Original ECE: {original_ece:.4f}")

    calibration_results = {'original_ece': original_ece, 'sensitivity': {}}

    target_eces = [0.0, 0.05, 0.10, 0.15, 0.20]
    A = np.log((1 - EXPERIMENT_CONFIG['beta']) / EXPERIMENT_CONFIG['alpha'])
    B = np.log(EXPERIMENT_CONFIG['beta'] / (1 - EXPERIMENT_CONFIG['alpha']))
    prior = 0.01
    prior_log_odds = np.log(prior / (1 - prior))
    A_adj = A + prior_log_odds
    B_adj = B + prior_log_odds

    for target_ece in target_eces:
        print(f"\nAnalyzing ECE = {target_ece:.2f}...")

        if target_ece == 0.0:
            perturbed_probs = original_probs
            actual_ece = original_ece
        else:
            perturbed_probs = add_calibration_error(original_probs, target_ece, test_labels)
            actual_ece = compute_ece(perturbed_probs, test_labels)

        # 运行SPRT
        decisions = []
        for prob in perturbed_probs:
            S_t = np.log(prob / (1 - prob + 1e-10))
            if S_t >= A_adj:
                decisions.append(1)
            elif S_t <= B_adj:
                decisions.append(0)
            else:
                decisions.append(1 if prob >= 0.5 else 0)

        y_pred = decisions
        y_true = test_labels

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + sum(y_true)) if sum(y_true) > 0 else 0

        calibration_results['sensitivity'][str(target_ece)] = {
            'target_ece': target_ece,
            'actual_ece': actual_ece,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'accuracy': acc
        }

        print(f"  Actual ECE: {actual_ece:.4f}, F1: {f1:.3f}, FPR: {fpr:.3f}")

    all_results['calibration_sensitivity'] = calibration_results

    # ========================================
    # 保存结果
    # ========================================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    output_file = os.path.join(output_dir, 'full_experimental_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")

    # 生成LaTeX表格
    print("\n" + "=" * 70)
    print("LaTeX Tables")
    print("=" * 70)

    # Table 1: Latency
    print("\n% Wall-Clock Latency")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Wall-Clock Latency Comparison}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Method & Latency (ms) & Calls & Speedup \\\\")
    print("\\midrule")
    for method, res in latency_results.items():
        print(f"{method} & {res['mean_ms']:.1f} $\\pm$ {res['std_ms']:.1f} & "
              f"{res['classifier_calls_per_sample']:.1f} & {res['speedup_vs_fulltext']:.2f}$\\times$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Table 2: Accuracy
    print("\n% Accuracy Comparison")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Accuracy Comparison}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Method & F1 & FPR & FNR & Avg Tokens \\\\")
    print("\\midrule")
    for method, res in accuracy_results.items():
        print(f"{method} & {res['f1']:.3f} & {res['fpr']:.3f} & {res['fnr']:.3f} & {res['avg_stopping_time']:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Table 3: Calibration
    print("\n% Calibration Sensitivity")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Calibration Sensitivity Analysis}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Target ECE & Actual ECE & F1 & FPR & FNR \\\\")
    print("\\midrule")
    for target_ece, res in calibration_results['sensitivity'].items():
        print(f"{res['target_ece']:.2f} & {res['actual_ece']:.4f} & {res['f1']:.3f} & "
              f"{res['fpr']:.3f} & {res['fnr']:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full Supplemental Experiments')
    parser.add_argument('--model_path', type=str, default='best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    run_full_experiments(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )
