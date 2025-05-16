#!/usr/bin/env bash
# create_jjmenv_nlp4b.sh — build & test GPU-enabled env
# Usage:  bash create_jjmenv_nlp4b.sh

set -euo pipefail

ENV_NAME="jjmenv_nlp4b"
YML_FILE="jjmenv_nlp4b.yml"

echo "▶ Creating conda environment \"$ENV_NAME\" from $YML_FILE …"
conda env create -f "$YML_FILE"

echo "▶ Activating environment and running a quick GPU sanity check …"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"