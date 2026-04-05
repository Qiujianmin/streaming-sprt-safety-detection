"""
Utility functions for SPRT experiments
"""

import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime


def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results: Dict[str, Any], filepath: str):
    """Save experiment results to JSON file."""
    results['timestamp'] = datetime.now().isoformat()
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted labels (0/1)
        labels: True labels (0/1)

    Returns:
        Dictionary of metrics
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    tp = ((predictions == 1) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def format_results_table(results: Dict[str, Any]) -> str:
    """
    Format results as markdown table.

    Args:
        results: Results dictionary

    Returns:
        Markdown formatted table string
    """
    lines = ["| Method | Accuracy | F1 | FPR | FNR | AvgTokens |"]
    lines.append("|--------|----------|-----|-----|-----|----------|")

    for method, metrics in results.get('results', {}).items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1', 0)
            fpr = metrics.get('fpr', 0)
            fnr = metrics.get('fnr', 0)
            avg_stop = metrics.get('avg_stopping_time', 0)
            lines.append(f"| {method} | {acc:.3f} | {f1:.3f} | {fpr:.3f} | {fnr:.3f} | {avg_stop:.1f} |")

    return "\n".join(lines)


def aggregate_results(result_files: List[str]) -> Dict[str, Any]:
    """
    Aggregate results from multiple experiment files.

    Args:
        result_files: List of paths to result JSON files

    Returns:
        Aggregated results dictionary
    """
    all_results = {}

    for filepath in result_files:
        results = load_results(filepath)
        dataset = results.get('dataset', 'unknown')
        all_results[dataset] = results

    return all_results
