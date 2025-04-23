#!/usr/bin/env bash
#SBATCH -J YOLO_sweep
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --array=1-24
#SBATCH -o logs/%x_%a.out
#SBATCH -e logs/%x_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bauermax@oregonstate.edu

set -euo pipefail

# 1) make sure logs dir exists
mkdir -p logs

# 2) create & populate venvs if they don't already exist

# ---- augmentenv ----
module load python/3.10
if [ ! -d augmentenv ]; then
  python3 -m venv augmentenv
  source augmentenv/bin/activate
  pip install -r req/augment-requirements.txt
  deactivate
fi

# ---- labelenv ----
module load python/3.10
if [ ! -d labelenv ]; then
  python3 -m venv labelenv
  source labelenv/bin/activate
  pip install -r req/label-requirements.txt
  deactivate
fi

# ---- trainenv ----
module load python/3.8
if [ ! -d trainenv ]; then
  python3 -m venv trainenv
  source trainenv/bin/activate
  pip install -r req/train-requirements.txt
  deactivate
fi

# 3) pull in our sweep parameters
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}q;d" combos.csv)
IFS=',' read -r MODEL_TYPE IMAGE_SIZE BRIGHT_POS BRIGHT_NEG BLUR <<< "$LINE"
export MODEL_TYPE IMAGE_SIZE BRIGHT_POS BRIGHT_NEG BLUR

# 4) kick off the driver
bash run_pipeline.sh