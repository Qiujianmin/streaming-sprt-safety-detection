#!/bin/bash
# Setup script for Cascade-Guard experiments on AutoDL GPU server
# Run this after SSH-ing into the server

set -e

echo "============================================"
echo "Cascade-Guard Experiment Setup on AutoDL"
echo "============================================"

# 1. Check GPU
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# 2. Install dependencies
echo ""
echo "[2/6] Installing Python dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
pip install torch torchvision torchaudio 2>/dev/null || true
pip install transformers>=4.40.0 datasets>=2.14.0 scikit-learn>=1.3.0 \
    scipy>=1.11.0 numpy>=1.24.0 tqdm>=4.65.0 accelerate>=0.20.0

# 3. Clone/upload code
echo ""
echo "[3/6] Setting up project directory..."
mkdir -p /root/cascade_guard && cd /root/cascade_guard

# 4. Download Llama Guard 3 8B
echo ""
echo "[4/6] Downloading Llama Guard 3 8B (this may take 10-20 minutes)..."
echo "Note: Requires HuggingFace login and Meta license acceptance."
echo "Run: huggingface-cli login"
echo "Then: python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/LlamaGuard-3-8B')\""

# Alternative: use a local path if model is pre-downloaded
echo ""
echo "If model is already downloaded, set --llama_guard_path to the local path."

# 5. Download datasets
echo ""
echo "[5/6] Pre-downloading datasets..."
python3 -c "
from datasets import load_dataset
print('  Downloading BeaverTails...')
load_dataset('PKU-Alignment/BeaverTails')
print('  Downloading PKU-SafeRLHF...')
load_dataset('PKU-Alignment/PKU-SafeRLHF')
print('  Downloading Qwen3GuardTest...')
load_dataset('Qwen/Qwen3GuardTest')
print('  Downloading CivilComments...')
load_dataset('google/civil_comments')
print('  All datasets downloaded.')
" || echo "  [WARN] Some datasets failed to download. They will be downloaded during experiments."

# 6. Run experiments
echo ""
echo "[6/6] Ready to run experiments!"
echo ""
echo "Commands to run:"
echo ""
echo "# Single dataset (fastest test):"
echo "cd /root/cascade_guard/code/experiments"
echo "python cascade_guard_experiments.py --dataset beavertails --device cuda --output_dir ./results"
echo ""
echo "# All datasets:"
echo "python cascade_guard_experiments.py --dataset all --device cuda --output_dir ./results"
echo ""
echo "# With custom Llama Guard path:"
echo "python cascade_guard_experiments.py --dataset all --device cuda --llama_guard_path /root/models/LlamaGuard-3-8B --output_dir ./results"
echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
