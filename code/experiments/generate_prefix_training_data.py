#!/usr/bin/env python3
"""
Generate prefix training data for fine-tuning Llama Guard on short prefixes.

For each dataset's TRAINING split:
1. Train Stage 1 (RoBERTa) classifier
2. Run streaming detector to collect stopping points
3. Extract prefixes at multiple token positions
4. Save (prefix_text, label) pairs for QLoRA training

Output: JSONL files with prefix samples for each dataset + combined.
"""

import json
import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Dataset Loading (same as cascade_guard_experiments.py)
# ============================================================
def load_dataset(dataset_name):
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    if dataset_name == 'beavertails':
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
        return {'text': train_texts, 'label': train_labels}, \
               {'text': val_texts, 'label': val_labels}, \
               {'text': test_texts[:3021], 'label': test_labels[:3021]}

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
        return {'text': train_texts, 'label': train_labels}, \
               {'text': val_texts, 'label': val_labels}, \
               {'text': test_texts[:3000], 'label': test_labels[:3000]}

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
        return {'text': train_texts, 'label': train_labels}, \
               {'text': val_texts, 'label': val_labels}, \
               {'text': test_texts[:651], 'label': test_labels[:651]}

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================
# Train Stage 1 RoBERTa
# ============================================================
def train_roberta(train_data, val_data, device='cuda'):
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, RobertaForSequenceClassification

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

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())

        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"  Epoch {epoch+1}/3: val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, tokenizer


# ============================================================
# Generate Prefix Samples
# ============================================================
def generate_prefix_samples(model, tokenizer, texts, labels, device='cuda',
                            prefix_positions=[5, 10, 15, 20, 30, 50, 72],
                            max_prefix_length=128):
    """For each text, extract prefixes at specified token positions."""
    model.eval()
    all_prefix_samples = []

    for idx, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts),
                                              desc="Generating prefixes")):
        # Tokenize full text
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_prefix_length,
                                   truncation=True)

        for pos in prefix_positions:
            if pos >= len(tokens):
                continue

            prefix_tokens = tokens[:pos]
            prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)

            if len(prefix_text.strip()) < 5:  # Skip very short prefixes
                continue

            # Get RoBERTa's probability on this prefix
            enc = tokenizer(prefix_text, max_length=max_prefix_length,
                           padding='max_length', truncation=True, return_tensors='pt')
            with torch.no_grad():
                logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
                prob = torch.softmax(logits, dim=-1)[0, 1].item()

            all_prefix_samples.append({
                'prefix_text': prefix_text,
                'prefix_tokens': pos,
                'label': label,
                'roberta_prob': round(prob, 4),
                'source_index': idx,
                'source_length': len(tokens)
            })

    return all_prefix_samples


# ============================================================
# Main
# ============================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = '/tmp/prefix_training_data'
    os.makedirs(output_dir, exist_ok=True)

    datasets = ['beavertails', 'pku_saferlhf', 'qwen3guardtest']
    all_combined = []

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {ds_name}")
        print(f"{'='*60}")

        # Load data
        train_data, val_data, test_data = load_dataset(ds_name)
        print(f"Train: {len(train_data['text'])}, Val: {len(val_data['text'])}, Test: {len(test_data['text'])}")
        print(f"Toxicity rate: {sum(train_data['label'])/len(train_data['label'])*100:.1f}%")

        # Train Stage 1 RoBERTa
        print("Training Stage 1 RoBERTa...")
        model, tokenizer = train_roberta(train_data, val_data, device)

        # Generate prefix samples from TRAINING data only
        print("Generating prefix samples from training data...")
        prefix_samples = generate_prefix_samples(
            model, tokenizer,
            train_data['text'], train_data['label'],
            device=device
        )

        # Also generate from test data for evaluation
        print("Generating prefix samples from test data...")
        test_prefix_samples = generate_prefix_samples(
            model, tokenizer,
            test_data['text'], test_data['label'],
            device=device
        )

        # Save per-dataset
        train_path = os.path.join(output_dir, f'{ds_name}_prefix_train.jsonl')
        with open(train_path, 'w') as f:
            for s in prefix_samples:
                s['dataset'] = ds_name
                f.write(json.dumps(s) + '\n')
        print(f"  Saved {len(prefix_samples)} prefix samples to {train_path}")

        test_path = os.path.join(output_dir, f'{ds_name}_prefix_test.jsonl')
        with open(test_path, 'w') as f:
            for s in test_prefix_samples:
                s['dataset'] = ds_name
                f.write(json.dumps(s) + '\n')
        print(f"  Saved {len(test_prefix_samples)} test prefix samples to {test_path}")

        all_combined.extend(prefix_samples)

        # Free GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save combined training data
    combined_path = os.path.join(output_dir, 'combined_prefix_train.jsonl')
    with open(combined_path, 'w') as f:
        for s in all_combined:
            f.write(json.dumps(s) + '\n')
    print(f"\nSaved {len(all_combined)} total combined prefix samples to {combined_path}")

    # Print statistics
    print(f"\n{'='*60}")
    print("Prefix Training Data Statistics")
    print(f"{'='*60}")
    for ds_name in datasets:
        ds_samples = [s for s in all_combined if s['dataset'] == ds_name]
        safe = sum(1 for s in ds_samples if s['label'] == 0)
        unsafe = sum(1 for s in ds_samples if s['label'] == 1)
        print(f"  {ds_name}: {len(ds_samples)} samples ({safe} safe, {unsafe} unsafe)")

    # Token position distribution
    pos_counts = {}
    for s in all_combined:
        pos = s['prefix_tokens']
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    print(f"\n  Token position distribution:")
    for pos in sorted(pos_counts.keys()):
        print(f"    {pos} tokens: {pos_counts[pos]} samples")


if __name__ == '__main__':
    main()
