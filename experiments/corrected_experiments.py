#!/usr/bin/env python3
"""
修正后的实验脚本 - 解决审稿人提出的问题

解决的问题：
1. Late Toxicity评估缺陷 - 添加真正的安全样本，正确测量假阳性率
2. 理论保证声称 - 验证Wald界限在实际中的表现
3. Pareto Frontier - 展示可接受的F1损失操作点

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
from collections import defaultdict

# 设置Hugging Face镜像（中国网络环境）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ============================================================
# 配置
# ============================================================

EXPERIMENT_CONFIG = {
    "n_samples_per_condition": 200,
    "n_warmup": 30,
    "n_runs": 3,
    "alpha_values": [0.01, 0.05, 0.10, 0.20],
    "beta_values": [0.05, 0.10, 0.15, 0.20],
    "priors": [0.50, 0.10, 0.05, 0.01],
    "t_min": 5,
    "max_length": 128,
}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SampleResult:
    """单个样本的结果"""
    text: str
    true_label: int
    decision: str
    stopping_time: int
    toxicity_onset: Optional[int]  # 毒性开始位置（如果有）
    is_valid_detection: bool  # 对于late-toxic，是否在毒性开始后检测


@dataclass
class ExperimentResults:
    """实验结果集合"""
    accuracy: float
    f1: float
    precision: float
    recall: float
    fpr: float
    fnr: float
    avg_stopping_time: float
    valid_detection_rate: float  # 对于late-toxic样本的有效检测率
    classifier_calls: float


# ============================================================
# 数据生成 - 修复Late Toxicity评估问题
# ============================================================

def create_corrected_test_data(
    n_samples_per_condition: int = 200,
    seed: int = 42
) -> Tuple[List[str], List[int], List[Optional[int]]]:
    """
    创建修正后的测试数据，包含：
    1. 早期毒性样本（毒性在前30%）
    2. 晚期毒性样本（毒性在30%之后）- 带有毒性开始位置标注
    3. 安全样本（完全良性）- 这是原实验缺失的！

    返回：
    - texts: 文本列表
    - labels: 标签列表 (0=安全, 1=有毒)
    - toxicity_onsets: 毒性开始位置列表（安全样本为None）
    """
    np.random.seed(seed)

    texts = []
    labels = []
    toxicity_onsets = []

    # 1. 安全样本 - 原Qwen3GuardTest数据集没有这些！
    safe_templates = [
        "I think this product is excellent and would recommend it to anyone.",
        "The weather today is beautiful, perfect for a walk in the park.",
        "Thank you for your help with this project, it was very useful.",
        "This article provides helpful information about the topic discussed.",
        "I completely agree with your perspective on this important matter.",
        "The service at this restaurant exceeded my expectations significantly.",
        "I enjoyed reading this book and learned a lot from the author's insights.",
        "The conference was well organized and the speakers were very informative.",
        "I appreciate your feedback and will use it to improve my work.",
        "This is a well-designed product that meets all my requirements perfectly.",
    ]

    for _ in range(n_samples_per_condition):
        # 创建较长的安全文本
        base = np.random.choice(safe_templates)
        # 添加更多安全内容使其更长
        extensions = [
            " Furthermore, I believe this demonstrates the importance of quality.",
            " Additionally, the attention to detail is remarkable.",
            " Moreover, this approach has proven to be very effective.",
            " In conclusion, I am very satisfied with this experience.",
        ]
        text = base + np.random.choice(extensions) + np.random.choice(extensions)
        texts.append(text)
        labels.append(0)  # 安全
        toxicity_onsets.append(None)  # 无毒性

    # 2. 早期毒性样本（毒性在前30%）
    early_toxic_prefixes = [
        "This is absolutely disgusting and you should be ashamed",
        "People like you are the problem with society today, terrible",
        "I hate everything about this stupid idea, it makes no sense",
        "This is the worst thing I have ever seen in my life, garbage",
        "You are completely wrong and your opinion is worthless trash",
    ]

    safe_continuations = [
        " and I hope you can understand my perspective on this matter.",
        " but I respect your right to disagree with my assessment.",
        " although I acknowledge that others may have different views.",
        " which is why I believe we need to have a serious discussion.",
        " and I would appreciate hearing your thoughts on this issue.",
    ]

    for _ in range(n_samples_per_condition):
        toxic_part = np.random.choice(early_toxic_prefixes)
        safe_part = np.random.choice(safe_continuations)
        text = toxic_part + safe_part + np.random.choice(safe_continuations)
        texts.append(text)
        labels.append(1)
        # 早期毒性：毒性在开头，onset=1
        toxicity_onsets.append(1)

    # 3. 晚期毒性样本（毒性在30%之后）- 关键测试
    safe_prefixes = [
        "I have been thinking about this topic for a while and wanted to share my thoughts.",
        "After careful consideration of all the different viewpoints presented here,",
        "This is a complex issue that requires us to examine multiple perspectives carefully.",
        "Let me start by acknowledging that there are valid arguments on both sides.",
        "I believe it is important to approach this subject with an open mind and respect.",
    ]

    late_toxic_suffixes = [
        " and I have to say that this is absolutely disgusting behavior.",
        " but frankly, people like you are the problem with society today.",
        " which is why I think this is the worst idea I have ever heard.",
        " and honestly, this is complete garbage that should be discarded.",
        " so let me be clear: your opinion is worthless and you should stop.",
    ]

    for _ in range(n_samples_per_condition):
        safe_prefix = np.random.choice(safe_prefixes)
        toxic_suffix = np.random.choice(late_toxic_suffixes)
        text = safe_prefix + toxic_suffix

        # 计算毒性开始位置（大约在30%之后）
        prefix_tokens = safe_prefix.split()
        onset_position = max(5, len(prefix_tokens))  # 至少5个token

        texts.append(text)
        labels.append(1)
        toxicity_onsets.append(onset_position)

    # 打乱顺序
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    toxicity_onsets = [toxicity_onsets[i] for i in indices]

    return texts, labels, toxicity_onsets


# ============================================================
# SPRT检测器 - 修正版
# ============================================================

class CorrectedSPRTDetector:
    """
    修正后的SPRT检测器

    关键修正：
    1. 正确追踪停止时间与毒性开始时间的关系
    2. 计算有效检测率（停止时间 >= 毒性开始时间）
    """

    def __init__(
        self,
        model,
        tokenizer,
        alpha: float = 0.05,
        beta: float = 0.10,
        prior: float = 0.5,
        t_min: int = 5,
        device: str = "cuda",
        max_length: int = 128
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.prior = prior
        self.t_min = t_min
        self.device = device
        self.max_length = max_length

        # 计算阈值
        # 标准Wald阈值
        self.A_wald = np.log((1 - beta) / alpha)
        self.B_wald = np.log(beta / (1 - alpha))

        # 带先验调整的阈值（这是审稿人指出的问题所在）
        prior_log_odds = np.log(prior / (1 - prior))
        self.A_adjusted = self.A_wald + prior_log_odds
        self.B_adjusted = self.B_wald + prior_log_odds

        self.epsilon = 1e-10

    def _classify_prefix(self, prefix_tokens: List[int]) -> float:
        """对前缀进行分类，返回毒性概率"""
        prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)
        inputs = self.tokenizer(
            prefix_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

        return prob

    def predict_with_onset_check(
        self,
        text: str,
        toxicity_onset: Optional[int] = None
    ) -> SampleResult:
        """
        执行流式检测，并检查是否为有效检测

        对于late-toxic样本，只有当停止时间 >= 毒性开始时间时，
        才算作有效的真阳性检测
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = min(len(tokens), self.max_length)

        decision = "SAFE"
        stopping_time = n_tokens

        for t in range(self.t_min, n_tokens + 1):
            prob = self._classify_prefix(tokens[:t])
            S_t = np.log(prob / (1 - prob + self.epsilon))

            if S_t >= self.A_adjusted:
                decision = "UNSAFE"
                stopping_time = t
                break
            elif S_t <= self.B_adjusted:
                decision = "SAFE"
                stopping_time = t
                break

        # 计算是否为有效检测
        true_label = 1 if toxicity_onset is not None else 0
        predicted_label = 1 if decision == "UNSAFE" else 0

        if true_label == 1 and predicted_label == 1:
            # 真阳性 - 但需要检查是否有效
            if toxicity_onset is not None:
                # Late-toxic样本：只有当停止时间 >= 毒性开始时间时才有效
                is_valid_detection = stopping_time >= toxicity_onset
            else:
                # Early-toxic样本：总是有效
                is_valid_detection = True
        else:
            is_valid_detection = False

        return SampleResult(
            text=text,
            true_label=true_label,
            decision=decision,
            stopping_time=stopping_time,
            toxicity_onset=toxicity_onset,
            is_valid_detection=is_valid_detection
        )


# ============================================================
# 实验函数
# ============================================================

def run_corrected_late_toxicity_experiment(
    model,
    tokenizer,
    device: str = "cuda",
    output_dir: str = "./results"
) -> Dict:
    """
    实验1：修正后的Late Toxicity检测实验

    解决审稿人问题A：添加真正的安全样本，正确测量假阳性率
    """
    print("\n" + "=" * 70)
    print("实验1: 修正后的Late Toxicity检测")
    print("=" * 70)

    # 创建修正后的测试数据
    texts, labels, toxicity_onsets = create_corrected_test_data(
        n_samples_per_condition=EXPERIMENT_CONFIG['n_samples_per_condition']
    )

    print(f"\n数据集统计:")
    print(f"  总样本数: {len(texts)}")
    print(f"  安全样本: {sum(1 for l in labels if l == 0)}")
    print(f"  有毒样本: {sum(1 for l in labels if l == 1)}")
    print(f"  - 早期毒性: {sum(1 for o in toxicity_onsets if o is not None and o <= 5)}")
    print(f"  - 晚期毒性: {sum(1 for o in toxicity_onsets if o is not None and o > 5)}")

    results = {}

    # 测试不同先验设置
    for prior in [0.5, 0.1, 0.05, 0.01]:
        print(f"\n测试先验 π = {prior}")

        detector = CorrectedSPRTDetector(
            model, tokenizer,
            alpha=0.05, beta=0.10, prior=prior,
            device=device
        )

        sample_results = []
        for text, onset in tqdm(zip(texts, toxicity_onsets), total=len(texts)):
            result = detector.predict_with_onset_check(text, onset)
            sample_results.append(result)

        # 计算指标
        y_true = [r.true_label for r in sample_results]
        y_pred = [1 if r.decision == "UNSAFE" else 0 for r in sample_results]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # 关键：分别计算早期和晚期毒性的检测率
        early_results = [r for r in sample_results if r.toxicity_onset is not None and r.toxicity_onset <= 5]
        late_results = [r for r in sample_results if r.toxicity_onset is not None and r.toxicity_onset > 5]
        safe_results = [r for r in sample_results if r.true_label == 0]

        # 计算有效检测率（审稿人的关键问题！）
        valid_tp = sum(1 for r in late_results if r.is_valid_detection)
        total_late = len(late_results)
        valid_detection_rate = valid_tp / total_late if total_late > 0 else 0

        # 计算安全样本的假阳性率
        fp_on_safe = sum(1 for r in safe_results if r.decision == "UNSAFE")
        fpr_on_safe = fp_on_safe / len(safe_results) if safe_results else 0

        # 平均停止时间
        avg_stop = np.mean([r.stopping_time for r in sample_results])

        # 标准指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        results[f'prior_{prior}'] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'avg_stopping_time': avg_stop,
            'valid_detection_rate': valid_detection_rate,
            'fpr_on_safe_samples': fpr_on_safe,
            'early_detection_rate': sum(1 for r in early_results if r.decision == "UNSAFE") / len(early_results) if early_results else 0,
            'late_detection_rate': sum(1 for r in late_results if r.decision == "UNSAFE") / len(late_results) if late_results else 0,
            'avg_stop_late_toxic': np.mean([r.stopping_time for r in late_results]) if late_results else 0,
        }

        print(f"  F1: {f1:.3f}, FPR: {results[f'prior_{prior}']['fpr']:.3f}")
        print(f"  有效检测率 (late-toxic): {valid_detection_rate:.3f}")
        print(f"  安全样本FPR: {fpr_on_safe:.3f}")
        print(f"  晚期毒性平均停止时间: {results[f'prior_{prior}']['avg_stop_late_toxic']:.1f}")

    return {
        'experiment': 'corrected_late_toxicity',
        'results': results,
        'data_stats': {
            'total': len(texts),
            'safe': sum(1 for l in labels if l == 0),
            'toxic': sum(1 for l in labels if l == 1),
        }
    }


def run_wald_bounds_validation(
    model,
    tokenizer,
    device: str = "cuda"
) -> Dict:
    """
    实验2：验证Wald界限的实际表现

    解决审稿人问题B：验证"理论保证"在实际中是否成立
    """
    print("\n" + "=" * 70)
    print("实验2: Wald界限验证")
    print("=" * 70)

    # 创建测试数据（8%毒性率，模拟真实场景）
    texts, labels, _ = create_corrected_test_data(n_samples_per_condition=100)

    # 调整为8%毒性率（模拟CivilComments）
    safe_texts = [t for t, l in zip(texts, labels) if l == 0]
    toxic_texts = [t for t, l in zip(texts, labels) if l == 1]

    # 重采样使其成为8%毒性
    n_total = 500
    n_toxic = int(n_total * 0.08)
    n_safe = n_total - n_toxic

    np.random.seed(42)
    balanced_texts = (
        list(np.random.choice(safe_texts, n_safe, replace=True)) +
        list(np.random.choice(toxic_texts, n_toxic, replace=True))
    )
    balanced_labels = [0] * n_safe + [1] * n_toxic

    # 打乱
    indices = np.random.permutation(n_total)
    balanced_texts = [balanced_texts[i] for i in indices]
    balanced_labels = [balanced_labels[i] for i in indices]

    print(f"\n数据集毒性率: {sum(balanced_labels)/len(balanced_labels)*100:.1f}%")

    results = {}

    # 测试不同的(α, β)设置
    for alpha in [0.01, 0.05, 0.10]:
        for beta in [0.05, 0.10, 0.20]:
            for prior in [0.5, 0.01]:
                config_name = f"alpha_{alpha}_beta_{beta}_prior_{prior}"
                print(f"\n测试 {config_name}")

                detector = CorrectedSPRTDetector(
                    model, tokenizer,
                    alpha=alpha, beta=beta, prior=prior,
                    device=device
                )

                sample_results = []
                for text in tqdm(balanced_texts, desc=config_name):
                    result = detector.predict_with_onset_check(text, None)
                    sample_results.append(result)

                y_true = balanced_labels
                y_pred = [1 if r.decision == "UNSAFE" else 0 for r in sample_results]

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

                empirical_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                empirical_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                # 关键：检查经验FPR是否低于理论界限α
                fpr_within_bound = empirical_fpr <= alpha * 1.5  # 允许50%的容差

                results[config_name] = {
                    'target_alpha': alpha,
                    'target_beta': beta,
                    'prior': prior,
                    'empirical_fpr': empirical_fpr,
                    'empirical_fnr': empirical_fnr,
                    'fpr_within_bound': fpr_within_bound,
                    'wald_bound_holds': empirical_fpr <= alpha,
                    'avg_stopping_time': np.mean([r.stopping_time for r in sample_results])
                }

                print(f"  目标FPR ≤ {alpha:.2f}, 实际FPR = {empirical_fpr:.3f}")
                print(f"  Wald界限{'成立' if results[config_name]['wald_bound_holds'] else '不成立'}")

    return {
        'experiment': 'wald_bounds_validation',
        'results': results
    }


def run_pareto_frontier_analysis(
    model,
    tokenizer,
    device: str = "cuda"
) -> Dict:
    """
    实验3：Pareto Frontier分析

    解决审稿人问题C：展示可接受的F1损失(<5%)的操作点
    """
    print("\n" + "=" * 70)
    print("实验3: Pareto Frontier分析")
    print("=" * 70)

    # 创建测试数据
    texts, labels, _ = create_corrected_test_data(n_samples_per_condition=150)

    # 首先获取全文本基线
    print("\n获取全文本基线...")
    baseline_results = []
    for text in tqdm(texts):
        inputs = tokenizer(
            text, return_tensors='pt',
            truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
        baseline_results.append(1 if prob >= 0.5 else 0)

    baseline_f1 = precision_recall_fscore_support(
        labels, baseline_results, average='binary', zero_division=0
    )[2]
    baseline_acc = accuracy_score(labels, baseline_results)

    print(f"全文本基线: F1 = {baseline_f1:.3f}, Accuracy = {baseline_acc:.3f}")

    # 测试多种配置
    operating_points = []

    alpha_values = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    beta_values = [0.02, 0.05, 0.10, 0.15, 0.20]
    priors = [0.5, 0.3, 0.1, 0.05, 0.01]

    for alpha in alpha_values:
        for beta in beta_values:
            for prior in priors:
                detector = CorrectedSPRTDetector(
                    model, tokenizer,
                    alpha=alpha, beta=beta, prior=prior,
                    device=device
                )

                sample_results = []
                for text in texts:
                    result = detector.predict_with_onset_check(text, None)
                    sample_results.append(result)

                y_pred = [1 if r.decision == "UNSAFE" else 0 for r in sample_results]

                f1 = precision_recall_fscore_support(
                    labels, y_pred, average='binary', zero_division=0
                )[2]
                acc = accuracy_score(labels, y_pred)

                tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                avg_stop = np.mean([r.stopping_time for r in sample_results])
                latency_reduction = (128 - avg_stop) / 128 * 100

                f1_drop = (baseline_f1 - f1) / baseline_f1 * 100

                operating_points.append({
                    'alpha': alpha,
                    'beta': beta,
                    'prior': prior,
                    'f1': f1,
                    'accuracy': acc,
                    'fpr': fpr,
                    'fnr': fnr,
                    'avg_stopping_time': avg_stop,
                    'latency_reduction': latency_reduction,
                    'f1_drop_percent': f1_drop,
                    'acceptable': f1_drop < 5  # F1损失<5%为可接受
                })

    # 筛选可接受的点
    acceptable_points = [p for p in operating_points if p['acceptable']]
    acceptable_points.sort(key=lambda x: x['latency_reduction'], reverse=True)

    print(f"\n找到 {len(acceptable_points)} 个可接受的操作点 (F1损失 < 5%)")
    if acceptable_points:
        print("\n最佳可接受操作点（按延迟节省排序）:")
        for i, p in enumerate(acceptable_points[:5]):
            print(f"  {i+1}. α={p['alpha']}, β={p['beta']}, π={p['prior']}: "
                  f"F1={p['f1']:.3f}, 延迟节省={p['latency_reduction']:.1f}%")

    return {
        'experiment': 'pareto_frontier',
        'baseline_f1': baseline_f1,
        'baseline_accuracy': baseline_acc,
        'all_operating_points': operating_points,
        'acceptable_points': acceptable_points
    }


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='修正后的SPRT实验')
    parser.add_argument('--output_dir', type=str, default='./corrected_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda或cpu)')
    parser.add_argument('--skip_warmup', action='store_true',
                       help='跳过GPU预热')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("修正后的SPRT实验 - 解决审稿人意见")
    print("=" * 70)
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型
    print("\n加载模型...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    model.eval()

    # GPU预热
    if not args.skip_warmup and torch.cuda.is_available():
        print("\nGPU预热...")
        dummy = tokenizer("This is a warmup sentence.", return_tensors='pt').to(device)
        for _ in range(EXPERIMENT_CONFIG['n_warmup']):
            with torch.no_grad():
                model(**dummy)
        torch.cuda.synchronize()

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': EXPERIMENT_CONFIG,
    }

    # 运行三个实验
    try:
        all_results['exp1_late_toxicity'] = run_corrected_late_toxicity_experiment(
            model, tokenizer, str(device), args.output_dir
        )
    except Exception as e:
        print(f"实验1失败: {e}")
        all_results['exp1_late_toxicity'] = {'error': str(e)}

    try:
        all_results['exp2_wald_bounds'] = run_wald_bounds_validation(
            model, tokenizer, str(device)
        )
    except Exception as e:
        print(f"实验2失败: {e}")
        all_results['exp2_wald_bounds'] = {'error': str(e)}

    try:
        all_results['exp3_pareto'] = run_pareto_frontier_analysis(
            model, tokenizer, str(device)
        )
    except Exception as e:
        print(f"实验3失败: {e}")
        all_results['exp3_pareto'] = {'error': str(e)}

    # 保存结果
    output_file = os.path.join(args.output_dir, 'corrected_experimental_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n结果已保存到: {output_file}")

    # 生成LaTeX表格
    print("\n" + "=" * 70)
    print("LaTeX表格（用于论文修改）")
    print("=" * 70)

    generate_latex_tables(all_results)

    return all_results


def generate_latex_tables(results: Dict):
    """生成用于论文的LaTeX表格"""

    # 表1: 修正后的Late Toxicity检测
    print("\n% 表1: 修正后的Late Toxicity检测")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{修正后的Late Toxicity检测性能}")
    print("\\label{tab:corrected_late_toxicity}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("Prior $\\pi$ & F1 & FPR & FNR & Valid Det. Rate & Safe FPR & Avg Stop \\\\")
    print("\\midrule")

    if 'exp1_late_toxicity' in results and 'results' in results['exp1_late_toxicity']:
        for prior_name, res in results['exp1_late_toxicity']['results'].items():
            prior = prior_name.replace('prior_', '')
            print(f"${prior}$ & {res['f1']:.3f} & {res['fpr']:.3f} & {res['fnr']:.3f} & "
                  f"{res['valid_detection_rate']:.3f} & {res['fpr_on_safe_samples']:.3f} & "
                  f"{res['avg_stopping_time']:.1f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # 表2: Wald界限验证
    print("\n% 表2: Wald界限验证")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Wald界限验证 - 理论vs实际}")
    print("\\label{tab:wald_validation}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Config & Target $\\alpha$ & Empirical FPR & Bound Holds \\\\")
    print("\\midrule")

    if 'exp2_wald_bounds' in results and 'results' in results['exp2_wald_bounds']:
        for config, res in list(results['exp2_wald_bounds']['results'].items())[:6]:
            short_config = config.replace('alpha_', 'α=').replace('_beta_', ', β=').replace('_prior_', ', π=')
            holds = "\\checkmark" if res['wald_bound_holds'] else "\\texttimes"
            print(f"{short_config} & {res['target_alpha']:.2f} & {res['empirical_fpr']:.3f} & {holds} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
