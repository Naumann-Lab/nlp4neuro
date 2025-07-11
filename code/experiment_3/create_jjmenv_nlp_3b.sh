#!/usr/bin/env bash
# create_jjmenv_nlp_3b.sh ― build & test the GPU-enabled env for experiment 3b
# Usage:  bash create_jjmenv_nlp_3b.sh

set -euo pipefail

ENV_NAME="jjmenv_nlp_3b"
YML_FILE="jjmenv_nlp_3b.yml"

echo "▶ Creating conda environment \"$ENV_NAME\" from $YML_FILE …"
conda env create -f "$YML_FILE"

echo "▶ Activating environment and running a quick GPU sanity check …"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python - <<'PY'
import torch, transformers, bitsandbytes as bnb
print("PyTorch      :", torch.__version__)
print("CUDA OK?     :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device       :", torch.cuda.get_device_name(0))
print("Transformers :", transformers.__version__)
print("bitsandbytes :", bnb.__version__)
print("\n✅ Environment jjmenv_nlp_3b ready for experiment 3b.")
PY
