#!/bin/bash
# Run on AutoDL GPU server to generate prefix data + fine-tune Llama Guard
# Usage: bash run_prefix_finetune.sh
# Expected runtime: ~10-15 hours on RTX 4090

set -e

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py310

# Install dependencies if needed
pip install peft bitsandbytes accelerate -q 2>/dev/null || true

cd /root/autodl-tmp/code

echo "=============================================="
echo "STEP 1: Generate prefix training data"
echo "=============================================="
python experiments/generate_prefix_training_data.py 2>&1 | tee /tmp/prefix_data_generation.log

echo ""
echo "=============================================="
echo "STEP 2: QLoRA fine-tune Llama Guard"
echo "=============================================="
python experiments/finetune_llama_guard_prefix.py \
    --output_dir /tmp/llama_guard_prefix_finetuned \
    2>&1 | tee /tmp/qdora_finetune.log

echo ""
echo "=============================================="
echo "STEP 3: Copy results for download"
echo "=============================================="
cp -r /tmp/prefix_training_data /root/autodl-tmp/results/
cp -r /tmp/llama_guard_prefix_finetuned /root/autodl-tmp/results/

echo ""
echo "=============================================="
echo "DONE. Check /root/autodl-tmp/results/ for outputs"
echo "=============================================="
ls -la /root/autodl-tmp/results/prefix_training_data/
ls -la /root/autodl-tmp/results/llama_guard_prefix_finetuned/
