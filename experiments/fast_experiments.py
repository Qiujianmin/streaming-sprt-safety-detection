#!/usr/bin/env python3
"""
快速实验脚本 - 用于审稿意见回应
使用更小的样本量，保存中间结果

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
from dataclasses import dataclass

# 设置Hugging Face镜像（中国网络环境）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 配置 - 更小的样本量
FAST_CONFIG = {
    "n_samples_per_condition": 50,  # 减少到50
    "n_warmup": 10,
    "alpha_values": [0.05, 0.10],
    "beta_values": [0.10, 0.20],
    "priors": [0.5, 0.1, 0.01],
    "t_min": 5,
    "max_length": 128,
}


@dataclass
class SampleResult:
    text: str
    true_label: int
    decision: str
    stopping_time: int
    toxicity_onset: Optional[int]
    is_valid_detection: bool


def create_test_data(n_samples: int = 50, seed: int = 42):
    """创建测试数据"""
    np.random.seed(seed)

    texts, labels, onsets = [], [], []

    # 安全样本
    safe_templates = [
        "I think this product is excellent and would recommend it.",
        "The weather today is beautiful, perfect for a walk.",
        "Thank you for your help with this project.",
        "This article provides helpful information.",
        "I completely agree with your perspective.",
    ]

    for _ in range(n_samples):
        base = np.random.choice(safe_templates)
        ext = " Furthermore, this demonstrates the importance of quality."
        texts.append(base + ext)
        labels.append(0)
        onsets.append(None)

    # 早期毒性
    early_toxic = [
        "This is absolutely disgusting and you should be ashamed",
        "People like you are the problem with society today",
        "I hate everything about this stupid idea",
    ]

    for _ in range(n_samples):
        toxic = np.random.choice(early_toxic)
        safe = " and I hope you can understand my perspective."
        texts.append(toxic + safe)
        labels.append(1)
        onsets.append(1)

    # 晚期毒性
    safe_prefixes = [
        "I have been thinking about this topic for a while and wanted to share.",
        "After careful consideration of all the different viewpoints,",
        "This is a complex issue that requires careful examination.",
    ]

    late_toxic_suffixes = [
        " and I have to say this is absolutely disgusting behavior.",
        " but frankly, people like you are the problem with society.",
        " which is why this is the worst idea I have ever heard.",
    ]

    for _ in range(n_samples):
        prefix = np.random.choice(safe_prefixes)
        suffix = np.random.choice(late_toxic_suffixes)
        text = prefix + suffix
        texts.append(text)
        labels.append(1)
        # 计算onset
        prefix_tokens = prefix.split()
        onsets.append(max(5, len(prefix_tokens)))

    # 打乱
    indices = np.random.permutation(len(texts))
    return ([texts[i] for i in indices],
            [labels[i] for i in indices],
            [onsets[i] for i in indices])


class SPRTDetector:
    """简化的SPRT检测器"""

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

        # 计算阈值
        self.A = np.log((1 - beta) / alpha) + np.log(prior / (1 - prior))
        self.B = np.log(beta / (1 - alpha)) + np.log(prior / (1 - prior))
        self.epsilon = 1e-10

    def predict(self, text, toxicity_onset=None):
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

        true_label = 1 if toxicity_onset is not None else 0
        predicted = 1 if decision == "UNSAFE" else 0

        is_valid = False
        if true_label == 1 and predicted == 1:
            if toxicity_onset is not None:
                is_valid = stopping_time >= toxicity_onset
            else:
                is_valid = True

        return SampleResult(text, true_label, decision, stopping_time,
                           toxicity_onset, is_valid)


def save_intermediate(output_dir, exp_name, data):
    """保存中间结果"""
    path = os.path.join(output_dir, f'{exp_name}_intermediate.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  中间结果已保存: {path}")


def run_experiment_1(model, tokenizer, device, output_dir):
    """实验1: Late Toxicity检测"""
    print("\n" + "=" * 60)
    print("实验1: Late Toxicity检测 (快速版)")
    print("=" * 60)

    texts, labels, onsets = create_test_data(n_samples=50)
    print(f"数据: {len(texts)} 样本, 安全={sum(1 for l in labels if l==0)}, 有毒={sum(1 for l in labels if l==1)}")

    results = {}

    for prior in [0.5, 0.1, 0.01]:
        print(f"\n测试 prior={prior}...")
        detector = SPRTDetector(model, tokenizer, alpha=0.05, beta=0.10,
                               prior=prior, device=device)

        sample_results = []
        for text, onset in tqdm(zip(texts, onsets), total=len(texts)):
            result = detector.predict(text, onset)
            sample_results.append(result)

        y_true = [r.true_label for r in sample_results]
        y_pred = [1 if r.decision == "UNSAFE" else 0 for r in sample_results]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        late_results = [r for r in sample_results if r.toxicity_onset and r.toxicity_onset > 5]
        safe_results = [r for r in sample_results if r.true_label == 0]

        valid_tp = sum(1 for r in late_results if r.is_valid_detection)
        valid_rate = valid_tp / len(late_results) if late_results else 0

        fpr_safe = sum(1 for r in safe_results if r.decision == "UNSAFE") / len(safe_results) if safe_results else 0

        f1 = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)[2]

        results[f'prior_{prior}'] = {
            'f1': float(f1),
            'fpr': float(fp / (fp + tn) if (fp + tn) > 0 else 0),
            'fnr': float(fn / (fn + tp) if (fn + tp) > 0 else 0),
            'valid_detection_rate': float(valid_rate),
            'fpr_on_safe': float(fpr_safe),
            'avg_stop': float(np.mean([r.stopping_time for r in sample_results])),
        }

        print(f"  F1={f1:.3f}, FPR={results[f'prior_{prior}']['fpr']:.3f}, "
              f"ValidDet={valid_rate:.3f}, SafeFPR={fpr_safe:.3f}")

        # 保存中间结果
        save_intermediate(output_dir, 'exp1', {'results': results})

    return {'experiment': 'late_toxicity', 'results': results}


def run_experiment_2(model, tokenizer, device, output_dir):
    """实验2: Wald界限验证"""
    print("\n" + "=" * 60)
    print("实验2: Wald界限验证 (快速版)")
    print("=" * 60)

    texts, labels, _ = create_test_data(n_samples=30)

    # 创建8%毒性率的数据集
    safe = [t for t, l in zip(texts, labels) if l == 0]
    toxic = [t for t, l in zip(texts, labels) if l == 1]

    np.random.seed(42)
    n_total = 200
    n_toxic = int(n_total * 0.08)
    n_safe = n_total - n_toxic

    balanced = list(np.random.choice(safe, n_safe, replace=True)) + \
               list(np.random.choice(toxic, n_toxic, replace=True))
    balanced_labels = [0] * n_safe + [1] * n_toxic

    print(f"数据: {n_total} 样本, 毒性率={sum(balanced_labels)/n_total*100:.1f}%")

    results = {}

    configs = [
        (0.05, 0.10, 0.5),
        (0.05, 0.10, 0.01),
        (0.10, 0.20, 0.5),
        (0.10, 0.20, 0.01),
    ]

    for alpha, beta, prior in configs:
        name = f"a{alpha}_b{beta}_p{prior}"
        print(f"\n测试 {name}...")

        detector = SPRTDetector(model, tokenizer, alpha=alpha, beta=beta,
                               prior=prior, device=device)

        sample_results = []
        for text in tqdm(balanced, desc=name):
            result = detector.predict(text, None)
            sample_results.append(result)

        y_pred = [1 if r.decision == "UNSAFE" else 0 for r in sample_results]
        tn, fp, fn, tp = confusion_matrix(balanced_labels, y_pred).ravel()

        emp_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        emp_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        results[name] = {
            'alpha': alpha,
            'beta': beta,
            'prior': prior,
            'empirical_fpr': float(emp_fpr),
            'empirical_fnr': float(emp_fnr),
            'wald_holds': emp_fpr <= alpha,
            'avg_stop': float(np.mean([r.stopping_time for r in sample_results])),
        }

        print(f"  目标FPR≤{alpha}, 实际FPR={emp_fpr:.3f}, "
              f"Wald{'成立' if emp_fpr <= alpha else '不成立'}")

        save_intermediate(output_dir, 'exp2', {'results': results})

    return {'experiment': 'wald_bounds', 'results': results}


def run_experiment_3(model, tokenizer, device, output_dir):
    """实验3: Pareto Frontier"""
    print("\n" + "=" * 60)
    print("实验3: Pareto Frontier (快速版)")
    print("=" * 60)

    texts, labels, _ = create_test_data(n_samples=40)

    # 基线
    print("\n获取基线...")
    baseline_pred = []
    for text in tqdm(texts, desc="Baseline"):
        inputs = tokenizer(text, return_tensors='pt',
                          truncation=True, max_length=128).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(**inputs).logits, dim=-1)[0, 1].item()
        baseline_pred.append(1 if prob >= 0.5 else 0)

    baseline_f1 = precision_recall_fscore_support(
        labels, baseline_pred, average='binary', zero_division=0)[2]
    print(f"基线 F1 = {baseline_f1:.3f}")

    operating_points = []
    configs = [
        (0.05, 0.10, 0.5), (0.05, 0.10, 0.1), (0.05, 0.10, 0.01),
        (0.10, 0.20, 0.5), (0.10, 0.20, 0.1), (0.10, 0.20, 0.01),
        (0.15, 0.20, 0.5), (0.15, 0.20, 0.1), (0.15, 0.20, 0.01),
    ]

    for alpha, beta, prior in configs:
        detector = SPRTDetector(model, tokenizer, alpha=alpha, beta=beta,
                               prior=prior, device=device)

        sample_results = []
        for text in texts:
            result = detector.predict(text, None)
            sample_results.append(result)

        y_pred = [1 if r.decision == "UNSAFE" else 0 for r in sample_results]
        f1 = precision_recall_fscore_support(
            labels, y_pred, average='binary', zero_division=0)[2]

        avg_stop = np.mean([r.stopping_time for r in sample_results])
        latency_red = (128 - avg_stop) / 128 * 100
        f1_drop = (baseline_f1 - f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0

        tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        operating_points.append({
            'alpha': alpha, 'beta': beta, 'prior': prior,
            'f1': float(f1), 'fpr': float(fpr),
            'avg_stop': float(avg_stop),
            'latency_reduction': float(latency_red),
            'f1_drop_percent': float(f1_drop),
            'acceptable': f1_drop < 5
        })

        print(f"  a={alpha}, b={beta}, p={prior}: F1={f1:.3f}, "
              f"Stop={avg_stop:.1f}, LatRed={latency_red:.1f}%, "
              f"F1Drop={f1_drop:.1f}%")

        save_intermediate(output_dir, 'exp3', {
            'baseline_f1': float(baseline_f1),
            'points': operating_points
        })

    acceptable = [p for p in operating_points if p['acceptable']]
    acceptable.sort(key=lambda x: x['latency_reduction'], reverse=True)

    print(f"\n找到 {len(acceptable)} 个可接受操作点 (F1损失<5%)")

    return {
        'experiment': 'pareto',
        'baseline_f1': float(baseline_f1),
        'points': operating_points,
        'acceptable': acceptable[:5]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./fast_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型
    print("\n加载模型...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    model.eval()

    # 预热
    print("预热GPU...")
    dummy = tokenizer("Warmup", return_tensors='pt').to(device)
    for _ in range(10):
        with torch.no_grad():
            model(**dummy)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': FAST_CONFIG,
    }

    # 运行实验
    try:
        all_results['exp1'] = run_experiment_1(model, tokenizer, str(device), args.output_dir)
    except Exception as e:
        print(f"实验1失败: {e}")
        all_results['exp1'] = {'error': str(e)}

    try:
        all_results['exp2'] = run_experiment_2(model, tokenizer, str(device), args.output_dir)
    except Exception as e:
        print(f"实验2失败: {e}")
        all_results['exp2'] = {'error': str(e)}

    try:
        all_results['exp3'] = run_experiment_3(model, tokenizer, str(device), args.output_dir)
    except Exception as e:
        print(f"实验3失败: {e}")
        all_results['exp3'] = {'error': str(e)}

    # 保存最终结果
    output_file = os.path.join(args.output_dir, 'final_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n结果已保存: {output_file}")
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    main()
