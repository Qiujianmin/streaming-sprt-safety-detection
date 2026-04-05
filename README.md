# SPRT Safety Detection

**Streaming Safety Detection for Large Language Models: A Sequential Hypothesis Testing Approach**

> Qiu Jianmin, Han Jinguang
>
> Southeast University
>
> ACM TIST 2026 (Under Review)

---

## Overview

This project implements **Contextual SPRT**, a sequential hypothesis testing framework for real-time LLM safety detection. The method monitors cumulative log-likelihood ratios and makes early decisions when evidence reaches statistically significant thresholds.

### Key Features

- **Real-time Detection**: Detects toxic content during generation
- **Theoretical Guarantees**: Bounded error rates (Type I and Type II) via Wald's SPRT
- **Prior Adjustment**: Handles imbalanced toxicity priors
- **77-96% Token Savings**: Significant latency reduction across all datasets

---

## Project Structure

```
SPRT_Safety_Detection/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── paper/                       # Academic paper
│   ├── safety_detection_paper.tex
│   ├── REBUTTAL.md             # First round rebuttal
│   └── REBUTTAL_ROUND2.md      # Second round rebuttal
│
├── code/                        # Source code
│   ├── sprt/                   # Core SPRT implementation
│   │   ├── __init__.py
│   │   ├── detector.py         # SPRTDetector class
│   │   ├── calibration.py      # Temperature scaling
│   │   └── utils.py            # Helper functions
│   └── experiments/            # Experiment scripts
│       └── comprehensive_experiments.py
│
├── results/                     # Experimental results
│   ├── civilcomments_comprehensive_results.json
│   ├── beavertails_comprehensive_results.json
│   ├── pku_saferlhf_comprehensive_results.json
│   └── qwen3guardtest_comprehensive_results.json
│
└── datasets/                    # Dataset information
    └── README.md               # Dataset download instructions
```

---

## Quick Start

### Installation

```bash
cd SPRT_Safety_Detection
pip install -r requirements.txt
```

### Basic Usage

```python
import sys
sys.path.append('code')

from sprt import SPRTDetector
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load classifier
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")

# Initialize SPRT detector
detector = SPRTDetector(
    classifier=model,
    alpha=0.05,      # False positive rate bound
    beta=0.10,       # False negative rate bound
    prior=0.01,      # Prior toxicity probability
    device="cuda"
)

# Stream detection
text = "Your generated text here..."
decision, stopping_time = detector.predict_streaming(text, tokenizer)
print(f"Decision: {decision}, Detected at token: {stopping_time}")
```

---

## Datasets

| Dataset | Test Size | Toxicity Rate | Content Type |
|---------|-----------|---------------|--------------|
| CivilComments | 5,000 | 8.0% | User comments |
| BeaverTails | 3,021 | 57.4% | LLM responses |
| PKU-SafeRLHF | 3,000 | 58.3% | Preference data |
| Qwen3GuardTest | 651 | 100% | Adversarial (with position annotations) |

See `datasets/README.md` for download instructions.

---

## Main Results

| Dataset | Method | Token Savings | F1 |
|---------|--------|---------------|-----|
| CivilComments | SPRT(π=0.01) | 94.7% | 0.326 |
| BeaverTails | SPRT(π=0.5) | 77.3% | 0.689 |
| PKU-SafeRLHF | SPRT(π=0.01) | 87.8% | 0.817 |
| Qwen3GuardTest | SPRT(π=0.05) | 96.1% | 1.000 |

---

## Citation

```bibtex
@article{qiu2026streaming,
  title={Streaming Safety Detection for Large Language Models: A Sequential Hypothesis Testing Approach},
  author={Qiu, Jianmin and Han, Jinguang},
  journal={ACM Transactions on Intelligent Systems and Technology},
  year={2026}
}
```

---

## License

MIT License

---

## Contact

Qiu Jianmin: 230239771@seu.edu.cn

Southeast University, Nanjing, China
