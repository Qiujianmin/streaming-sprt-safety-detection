#!/bin/bash
# Complete pipeline: prefix data generation → QLoRA fine-tuning → cascade comparison
# Usage: bash run_full_pipeline.sh
# Expected runtime: ~15-20 hours on RTX 4090

set -e

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py310

# Install dependencies
pip install peft bitsandbytes accelerate datasets scikit-learn -q 2>/dev/null || true

# Set HF mirror for China
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache

echo "=============================================="
echo "STEP 1: Generate prefix training data"
echo "=============================================="
python /tmp/generate_prefix_training_data.py 2>&1 | tee /tmp/prefix_data_gen.log

echo ""
echo "=============================================="
echo "STEP 2: QLoRA fine-tune Llama Guard"
echo "=============================================="
python /tmp/finetune_llama_guard_prefix.py \
    --output_dir /tmp/llama_guard_prefix_finetuned \
    2>&1 | tee /tmp/qlora_finetune.log

echo ""
echo "=============================================="
echo "STEP 3: Run cascade comparison (ZS vs FT)"
echo "=============================================="
python /tmp/cascade_guard_ft_experiments.py \
    --adapter_path /tmp/llama_guard_prefix_finetuned/final_adapter \
    --llama_guard_path /root/autodl-tmp/llama_guard/LLM-Research/Llama-Guard-3-8B \
    --output_dir /tmp/cascade_ft_results \
    --datasets beavertails pku_saferlhf qwen3guardtest \
    2>&1 | tee /tmp/cascade_comparison.log

echo ""
echo "=============================================="
echo "STEP 4: Copy results"
echo "=============================================="
mkdir -p /root/autodl-tmp/results/cascade_ft_results
cp -r /tmp/cascade_ft_results/* /root/autodl-tmp/results/cascade_ft_results/ 2>/dev/null || true
cp -r /tmp/llama_guard_prefix_finetuned /root/autodl-tmp/results/ 2>/dev/null || true
cp -r /tmp/prefix_training_data /root/autodl-tmp/results/ 2>/dev/null || true

echo ""
echo "=============================================="
echo "DONE. Check /root/autodl-tmp/results/"
echo "=============================================="
ls -la /root/autodl-tmp/results/cascade_ft_results/
