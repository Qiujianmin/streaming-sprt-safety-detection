# Datasets

This folder contains information about the datasets used in this project. Due to size constraints, the actual data files are not included in this repository.

## Download Instructions

### 1. CivilComments (CivilComments)

```python
from datasets import load_dataset
dataset = load_dataset("civil_comments")
```

- **Source**: https://huggingface.co/datasets/civil_comments
- **Test Size**: 5,000 samples
- **Toxicity Rate**: 8.0%

### 2. BeaverTails

```python
from datasets import load_dataset
dataset = load_dataset("PKU-Alignment/BeaverTails")
```

- **Source**: https://huggingface.co/datasets/PKU-Alignment/BeaverTails
- **Test Size**: 3,021 samples
- **Toxicity Rate**: 57.4%

### 3. PKU-SafeRLHF

```python
from datasets import load_dataset
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
```

- **Source**: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
- **Test Size**: 3,000 samples
- **Toxicity Rate**: 58.3%

### 4. Qwen3GuardTest

```python
from datasets import load_dataset
dataset = load_dataset("Qwen/Qwen3GuardTest")
```

- **Source**: https://huggingface.co/datasets/Qwen/Qwen3GuardTest
- **Test Size**: 651 samples
- **Toxicity Rate**: 100%
- **Special Feature**: Position annotations for late-toxicity analysis

## Data Processing

After downloading, process the data using:

```bash
python code/experiments/comprehensive_experiments.py --prepare_data
```

## Data Format

All datasets are converted to a unified format:

```json
{
    "text": "The text content",
    "label": 0 or 1,  // 0 = safe, 1 = toxic
    "toxicity_position": null or int  // For Qwen3GuardTest only
}
```
