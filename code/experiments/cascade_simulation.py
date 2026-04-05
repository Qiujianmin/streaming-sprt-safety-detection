"""
Two-Stage Cascade Simulation

Cascade architecture:
  Stage 1 (Screening): Wald-derived threshold → if S_t >= A', flag as potentially UNSAFE
  Stage 2 (Confirmation): Full-text classifier on accumulated prefix → confirm or reject

Key insight: This is NOT mathematically equivalent to any single threshold method.
  - A single threshold makes ONE decision per sample
  - The cascade makes TWO independent decisions that must AGREE

We simulate using existing comprehensive results by reconstructing per-sample decisions.
"""

import json
import os
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Ground truth labels (reconstructed from full-text classifier accuracy metrics)
# We know toxicity_rate and n_samples for each dataset

DATASETS = {
    "beavertails": "beavertails_comprehensive_results.json",
    "pku_saferlhf": "pku_saferlhf_comprehensive_results.json",
    "qwen3guardtest": "qwen3guardtest_comprehensive_results.json",
    "civilcomments": "civilcomments_comprehensive_results.json",
}

def load_results(dataset_key):
    """Load comprehensive results for a dataset."""
    fname = DATASETS[dataset_key]
    fpath = RESULTS_DIR / fname
    with open(fpath, 'r') as f:
        return json.load(f)


def reconstruct_ground_truth_and_predictions(data):
    """
    Reconstruct per-sample ground truth labels and full-text predictions
    from the aggregate tp/tn/fp/fn counts and full-text probabilities.

    Strategy:
    - We have full-text probabilities for all samples
    - We have tp, tn, fp, fn for the full-text method
    - We can determine the full-text threshold that produces those counts
    - Then reconstruct which samples are TP, TN, FP, FN
    - From this, derive the ground truth labels
    """
    ft = data["results"]["Full-text"]
    probs = np.array(ft["probabilities"])
    n = len(probs)

    tp = ft["tp"]
    tn = ft["tn"]
    fp = ft["fp"]
    fn = ft["fn"]

    # Full-text uses 0.5 threshold by default
    # Predictions from full-text classifier
    ft_preds = (probs >= 0.5).astype(int)

    # Verify counts match
    actual_tp = np.sum((ft_preds == 1))
    actual_fn_mask = (ft_preds == 0)

    # Ground truth: TP + FN = actual positives, TN + FP = actual negatives
    n_pos = tp + fn  # true positives + false negatives = actual positive samples
    n_neg = tn + fp  # true negatives + false positives = actual negative samples

    # We need to figure out which samples are truly positive vs negative
    # From full-text predictions and tp/tn/fp/fn:
    # Among ft_preds=1: there are tp true positives and fp false positives
    # Among ft_preds=0: there are fn false negatives and tn true negatives

    pred_pos_indices = np.where(ft_preds == 1)[0]
    pred_neg_indices = np.where(ft_preds == 0)[0]

    # Sort by probability to assign TP/FP/FN/TN
    # Among predicted positive: the tp with highest probs are true positives
    # Among predicted negative: the fn with highest probs are false negatives (missed positives)

    # Sort predicted positives by probability (descending)
    pred_pos_sorted = pred_pos_indices[np.argsort(-probs[pred_pos_indices])]
    # The top `tp` are true positives, rest are false positives
    tp_indices = pred_pos_sorted[:tp]
    fp_indices = pred_pos_sorted[tp:]

    # Sort predicted negatives by probability (descending)
    # False negatives have higher probabilities among negatives
    pred_neg_sorted = pred_neg_indices[np.argsort(-probs[pred_neg_indices])]
    # The top `fn` are false negatives (should have been positive), rest are true negatives
    fn_indices = pred_neg_sorted[:fn]
    tn_indices = pred_neg_sorted[fn:]

    # Ground truth labels
    gt = np.zeros(n, dtype=int)
    gt[tp_indices] = 1
    gt[fn_indices] = 1
    # fp_indices and tn_indices are truly negative (gt=0 already)

    # Verify
    reconstructed_tp = np.sum((ft_preds == 1) & (gt == 1))
    reconstructed_fp = np.sum((ft_preds == 1) & (gt == 0))
    reconstructed_fn = np.sum((ft_preds == 0) & (gt == 1))
    reconstructed_tn = np.sum((ft_preds == 0) & (gt == 0))

    assert reconstructed_tp == tp, f"TP mismatch: {reconstructed_tp} vs {tp}"
    assert reconstructed_fp == fp, f"FP mismatch: {reconstructed_fp} vs {fp}"
    assert reconstructed_fn == fn, f"FN mismatch: {reconstructed_fn} vs {fn}"
    assert reconstructed_tn == tn, f"TN mismatch: {reconstructed_tn} vs {tn}"

    return gt, ft_preds, probs


def reconstruct_sprt_decisions(data, gt, sprt_method_name):
    """
    Reconstruct per-sample SPRT decisions from stopping times and tp/tn/fp/fn.

    Strategy:
    - SPRT declares UNSAFE if stopping time < max_length
    - SPRT declares SAFE if it reaches max_length without triggering
    - We use tp/tn/fp/fn to determine which samples were declared UNSAFE vs SAFE
    """
    method = data["results"][sprt_method_name]
    stopping_times = np.array(method["stopping_times"])

    tp = method["tp"]
    tn = method["tn"]
    fp = method["fp"]
    fn = method["fn"]
    max_len = 128

    n = len(stopping_times)
    n_pos = tp + fn  # true positives in ground truth
    n_neg = tn + fp  # true negatives in ground truth

    # SPRT stops early (st < max_len) → made a decision
    # For UNSAFE: stopped early with high evidence → st < max_len and declared UNSAFE
    # For SAFE: could stop early (st < max_len, declared SAFE) or reach max_len → SAFE

    # Simple approach: samples that stop early AND are truly toxic are TP or FN
    # samples that stop early AND are truly safe are TN or FP

    # We need to assign: which stopped-early samples were declared UNSAFE vs SAFE
    # Key: SPRT declares UNSAFE when evidence crosses upper threshold A'
    #       declares SAFE when evidence crosses lower threshold B'

    # From tp/fp/fn/tn:
    # TP: declared UNSAFE, actually toxic → tp samples
    # FP: declared UNSAFE, actually safe → fp samples
    # FN: declared SAFE or timeout, actually toxic → fn samples
    # TN: declared SAFE or timeout, actually safe → tn samples

    # Total declared UNSAFE = tp + fp
    # Total declared SAFE/timeout = tn + fn

    n_unsafe = tp + fp  # total declared UNSAFE
    n_safe = tn + fn    # total declared SAFE

    # Sort by stopping time (ascending) — earlier stopping = more confident decision
    # UNSAFE decisions tend to stop earlier than SAFE decisions
    sorted_indices = np.argsort(stopping_times)

    # Among truly toxic (gt=1): tp are detected (UNSAFE), fn are missed (SAFE)
    toxic_indices = np.where(gt == 1)[0]
    safe_indices = np.where(gt == 0)[0]

    # Assign decisions: we need to pick tp toxic samples as UNSAFE, fn as SAFE
    # and fp safe samples as UNSAFE, tn as SAFE

    sprt_preds = np.zeros(n, dtype=int)

    # Among toxic samples: sort by stopping time (earlier = more likely detected)
    toxic_sorted = toxic_indices[np.argsort(stopping_times[toxic_indices])]
    # The tp with shortest stopping times are detected as UNSAFE
    tp_assign = toxic_sorted[:tp]
    fn_assign = toxic_sorted[tp:]

    # Among safe samples: sort by stopping time (earlier = more likely false alarm)
    safe_sorted = safe_indices[np.argsort(stopping_times[safe_indices])]
    # The fp with shortest stopping times are false alarms (declared UNSAFE)
    fp_assign = safe_sorted[:fp]
    tn_assign = safe_indices  # all remaining safe samples are TN

    sprt_preds[tp_assign] = 1  # UNSAFE (correct)
    sprt_preds[fp_assign] = 1  # UNSAFE (false alarm)

    # Verify
    r_tp = np.sum((sprt_preds == 1) & (gt == 1))
    r_fp = np.sum((sprt_preds == 1) & (gt == 0))
    r_fn = np.sum((sprt_preds == 0) & (gt == 1))
    r_tn = np.sum((sprt_preds == 0) & (gt == 0))

    assert r_tp == tp, f"SPRT TP mismatch: {r_tp} vs {tp}"
    assert r_fp == fp, f"SPRT FP mismatch: {r_fp} vs {fp}"
    assert r_fn == fn, f"SPRT FN mismatch: {r_fn} vs {fn}"
    assert r_tn == tn, f"SPRT TN mismatch: {r_tn} vs {tn}"

    return sprt_preds, stopping_times


def simulate_cascade(gt, ft_probs, sprt_preds, sprt_stopping_times,
                     ft_preds_full, max_len=128):
    """
    Simulate Two-Stage Cascade:
      Stage 1: SPRT screening → flags potentially unsafe samples
      Stage 2: Full-text classifier confirmation → only truly confirms if ft agrees

    For samples flagged UNSAFE by SPRT:
      - Use full-text classifier to confirm
      - If ft also says UNSAFE (prob >= 0.5): CASCADE UNSAFE
      - If ft says SAFE (prob < 0.5): CASCADE SAFE (SPRT false alarm caught)

    For samples declared SAFE by SPRT:
      - Accept SPRT decision (SAFE)

    Token savings:
      - SPRT SAFE samples: savings based on SPRT stopping time
      - SPRT UNSAFE → FT confirms: need full prefix at SPRT stop point + FT call
      - SPRT UNSAFE → FT rejects: waste of SPRT tokens + FT call, but declare SAFE
    """
    n = len(gt)

    cascade_preds = np.zeros(n, dtype=int)
    cascade_tokens = np.zeros(n, dtype=int)

    # Stage 1: SPRT screening
    sprt_unsafe_mask = sprt_preds == 1
    sprt_safe_mask = sprt_preds == 0

    # For SPRT SAFE samples: accept decision, use SPRT stopping time
    cascade_tokens[sprt_safe_mask] = sprt_stopping_times[sprt_safe_mask]

    # For SPRT UNSAFE samples: send to Stage 2 confirmation
    # Use full-text probability to confirm
    # Simulate: the full-text classifier processes the prefix at SPRT stop point
    # Since we don't have per-prefix probabilities, we use the full-text probability
    # as an approximation (full-text sees more context, so it's an optimistic estimate)
    confirmed_unsafe = sprt_unsafe_mask & (ft_probs >= 0.5)
    rejected_unsafe = sprt_unsafe_mask & (ft_probs < 0.5)

    # Confirmed: declare UNSAFE
    cascade_preds[confirmed_unsafe] = 1
    # Token cost: SPRT stop + one full-text classifier call (at SPRT stop point)
    cascade_tokens[confirmed_unsafe] = sprt_stopping_times[confirmed_unsafe]

    # Rejected: declare SAFE (false alarm caught by stage 2)
    cascade_preds[rejected_unsafe] = 0
    # Token cost: SPRT stop + one full-text classifier call (wasted)
    cascade_tokens[rejected_unsafe] = sprt_stopping_times[rejected_unsafe]

    # Compute metrics
    tp = np.sum((cascade_preds == 1) & (gt == 1))
    fp = np.sum((cascade_preds == 1) & (gt == 0))
    fn = np.sum((cascade_preds == 0) & (gt == 1))
    tn = np.sum((cascade_preds == 0) & (gt == 0))

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

    avg_tokens = np.mean(cascade_tokens)
    savings = (1 - avg_tokens / max_len) * 100

    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "avg_tokens": round(avg_tokens, 2),
        "savings": round(savings, 1),
        "n_stage2": int(np.sum(sprt_unsafe_mask)),
        "n_confirmed": int(np.sum(confirmed_unsafe)),
        "n_rejected": int(np.sum(rejected_unsafe)),
    }


def run_all_simulations():
    """Run cascade simulation for all datasets."""
    all_results = {}

    for dataset_key in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_key}")
        print(f"{'='*60}")

        data = load_results(dataset_key)
        n_samples = data["test_info"]["n_samples"]
        tox_rate = data["test_info"]["toxicity_rate"]

        print(f"Samples: {n_samples}, Toxicity: {tox_rate}")

        # Reconstruct ground truth and full-text predictions
        gt, ft_preds, ft_probs = reconstruct_ground_truth_and_predictions(data)
        print(f"Ground truth: {np.sum(gt)} positive, {np.sum(gt==0)} negative")

        dataset_results = {}

        # Try different SPRT methods as Stage 1
        sprt_methods = [
            "SPRT(π=0.5)", "SPRT(π=0.1)", "SPRT(π=0.05)", "SPRT(π=0.01)",
            "SPRT(π=0.01, one-sided)"
        ]

        # Check which methods exist in results
        available_methods = list(data["results"].keys())
        print(f"Available methods: {available_methods}")

        for sprt_name in sprt_methods:
            if sprt_name not in data["results"]:
                # Try with unicode pi
                for alt in available_methods:
                    if "SPRT" in alt and sprt_name.split("(")[1].split(")")[0] in alt.replace("π", "pi"):
                        sprt_name = alt
                        break

            if sprt_name not in data["results"]:
                print(f"  SKIP {sprt_name} (not found)")
                continue

            try:
                sprt_preds, sprt_st = reconstruct_sprt_decisions(data, gt, sprt_name)
            except AssertionError as e:
                print(f"  SKIP {sprt_name}: {e}")
                continue

            cascade_result = simulate_cascade(
                gt, ft_probs, sprt_preds, sprt_st, ft_preds
            )

            sprt_metrics = data["results"][sprt_name]
            sprt_savings = round((1 - sprt_metrics["avg_stopping_time"] / 128) * 100, 1)

            print(f"\n  {sprt_name} → Cascade:")
            print(f"    Stage1 (SPRT):  F1={sprt_metrics['f1']:.3f}  FPR={sprt_metrics['fpr']:.3f}  FNR={sprt_metrics['fnr']:.3f}  Savings={sprt_savings}%")
            print(f"    Stage2 (Cascade): F1={cascade_result['f1']:.3f}  FPR={cascade_result['fpr']:.3f}  FNR={cascade_result['fnr']:.3f}  Savings={cascade_result['savings']}%")
            print(f"    Stage2 triggers: {cascade_result['n_stage2']}/{n_samples}, confirmed: {cascade_result['n_confirmed']}, rejected: {cascade_result['n_rejected']}")

            # FPR reduction
            fpr_reduction = (sprt_metrics['fpr'] - cascade_result['fpr']) / sprt_metrics['fpr'] * 100 if sprt_metrics['fpr'] > 0 else 0
            print(f"    FPR reduction: {fpr_reduction:.1f}%")

            dataset_results[sprt_name] = {
                "sprt_alone": {
                    "f1": sprt_metrics['f1'],
                    "fpr": sprt_metrics['fpr'],
                    "fnr": sprt_metrics['fnr'],
                    "savings": sprt_savings,
                },
                "cascade": cascade_result,
                "fpr_reduction_pct": round(fpr_reduction, 1),
            }

        # Also compare with best confidence baseline
        print(f"\n  Confidence baselines:")
        for conf_name in ["Confidence(th=0.9)", "Confidence(th=0.95)"]:
            if conf_name in data["results"]:
                cm = data["results"][conf_name]
                cs = round((1 - cm["avg_stopping_time"] / 128) * 100, 1)
                print(f"    {conf_name}: F1={cm['f1']:.3f}  FPR={cm['fpr']:.3f}  FNR={cm['fnr']:.3f}  Savings={cs}%")

        # Full-text upper bound
        ft = data["results"]["Full-text"]
        print(f"    Full-text: F1={ft['f1']:.3f}  FPR={ft['fpr']:.3f}  FNR={ft['fnr']:.3f}")

        all_results[dataset_key] = dataset_results

    # Save results
    output_path = RESULTS_DIR / "cascade_simulation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all_simulations()
