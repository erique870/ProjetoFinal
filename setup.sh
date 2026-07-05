#!/bin/bash

python3 -m venv env

source env/bin/activate

conda deactivate 2>/dev/null || true

pip install -r requirements.txt

pip install -e . --no-deps

pip uninstall torch torchvision torchaudio -y

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -U "bitsandbytes>=0.46.1"

pip install --upgrade trl transformers accelerate

HF_TOKEN="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"

hf auth login --token "$HF_TOKEN" --add-to-git-credential