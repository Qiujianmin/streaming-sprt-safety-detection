# Cascade-Guard: Streaming Safety Detection for LLM Outputs

**To Cascade or Not to Cascade? An Empirical Study of Two-Stage Safety Detection for Streaming LLM Outputs**

> Jianmin Qiu, Jinguang Han
>
> Southeast University, Nanjing, China
>
> ACM TIST 2026 (Under Review)

---

## Overview

Cascade-Guard is a heterogeneous two-stage architecture for real-time safety detection during LLM text generation:

- **Stage 1 (RoBERTa, 125M params)**: Fast screening (~7ms/call) — monitors each generated prefix using Wald's formula-derived thresholds
- **Stage 2 (Llama Guard 3 8B)**: Precise confirmation (~86ms/call) — validates only flagged prefixes to eliminate false alarms

Key findings:
- Single-stage streaming achieves 71--96% token savings with competitive F1 (0.356--0.780)
- The heterogeneous cascade reduces FPR by 50--96%, but at the cost of recall when Stage 2 overrules correct Stage 1 decisions on short prefixes (**prefix bottleneck**)
- Prefix-aware fine-tuning of Llama Guard improves cascade F1 by 25--209% and reduces prefix failures by 52--79%
- End-to-end latency verified at **6.3x speedup** (measured on RTX 4090)

---

## Project Structure

```
SPRT_Safety_Detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── code/
│   ├── sprt/                              # Core SPRT implementation
│   │   ├── detector.py                    # SPRTDetector class
│   │   ├── calibration.py                 # Temperature scaling
│   │   └── utils.py                       # Helper functions
│   └── experiments/                       # Experiment scripts
│       ├── cascade_guard_experiments.py   # Main cascade experiments
│       ├── cascade_guard_ft_experiments.py # Prefix-finetuned experiments
│       ├── generate_prefix_training_data.py
│       ├── finetune_llama_guard_prefix.py # QLoRA fine-tuning
│       ├── generate_pareto_plot.py        # Pareto frontier figure
│       ├── run_full_pipeline.sh           # Full pipeline runner
│       └── setup_autodl.sh               # AutoDL server setup
│
├── paper/
│   ├── safety_detection_paper.tex         # Paper source
│   ├── safety_detection_paper.pdf         # Compiled paper
│   ├── reviewer3_rebuttal.md             # Reviewer 3 rebuttal
│   ├── experiment_results.json           # Experiment data
│   ├── experiment_results/               # Per-experiment results
│   │   ├── calibration_pku_results.json
│   │   ├── fixed_w10_pku_results.json
│   │   └── prefix_failure_examples.json
│   └── figures/
│       └── pareto_frontier.pdf           # Pareto frontier plot
│
├── results/                               # Comprehensive experiment results
│   ├── beavertails_comprehensive_results.json
│   ├── civilcomments_comprehensive_results.json
│   ├── pku_saferlhf_comprehensive_results.json
│   ├── qwen3guardtest_comprehensive_results.json
│   ├── cascade_simulation_results.json
│   └── supplemental_full_results.json
│
├── llama_guard_prefix_adapter/            # QLoRA adapter for prefix-aware LG
│   ├── README.md                          # Loading instructions
│   ├── tokenizer_config.json
│   └── trainer_state.json
│
└── datasets/
    └── README.md                          # Dataset download instructions
```

---

## Datasets

| Dataset | Test Size | Toxicity Rate | Content Type |
|---------|-----------|---------------|--------------|
| CivilComments | 5,000 | 8.0% | User comments |
| BeaverTails | 907 | 57.3% | LLM responses |
| PKU-SafeRLHF | 3,000 | 52.0% | Preference data |
| Qwen3GuardTest | 489 | 100% | Adversarial |

See `datasets/README.md` for download instructions.

---

## End-to-End Latency (Measured on RTX 4090)

| Component | Latency |
|-----------|---------|
| LLM generation (8B FP16) | 27.2 ms/tok |
| RoBERTa Stage 1 | 7.0 ms/call |
| Llama Guard Stage 2 | 86.2 ms/call |
| **Normal TBT** | **34.2 ms** |
| **Flagged TBT** | **120.3 ms** |
| Cascade(π=0.01) total | ~570 ms |
| Full-text baseline | ~3,570 ms |
| **Speedup** | **6.3x** |

---

## Citation

```bibtex
@article{qiu2026cascade,
  title={To Cascade or Not to Cascade? An Empirical Study of Two-Stage Safety Detection for Streaming LLM Outputs},
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

Jianmin Qiu: 230239771@seu.edu.cn

Southeast University, Nanjing, China
