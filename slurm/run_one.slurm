#!/bin/bash
#SBATCH --job-name=foci_array
#SBATCH --output=results/logs/array_%A_%a.out   # %A=array job id, %a=task id
#SBATCH --error=results/logs/array_%A_%a.err
#SBATCH --partition=capella                      # H100 partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1                         # 1 full H100 per slide
#SBATCH --cpus-per-task=16                        # feed the GPU (tile I/O + TF threads)
#SBATCH --mem=64G
#SBATCH --time=00:40:00                           # per-slide walltime (generous)

# --- foci_counter: ONE slide per SLURM array task ---------------------------
# Each array task processes the slide at index $SLURM_ARRAY_TASK_ID in the glob.
# Submit with the right array range via slurm/submit_array.sh (recommended),
# which fills in --array automatically. Manual example (19 slides -> 0..18):
#     sbatch --array=0-18 slurm/run_one.slurm "dataset/*_HE_*.czi"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

GLOB="${1:-dataset/*_HE_*.czi}"
FILES=( $(ls $GLOB) )          # slide names have no spaces; portable to bash 3.2
CZI="${FILES[$SLURM_ARRAY_TASK_ID]}"
echo "=== task $SLURM_ARRAY_TASK_ID/$(( ${#FILES[@]} - 1 ))  slide: $CZI ==="
echo "=== node: $(hostname)  job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="

# Activate env; activate.d sets LD_LIBRARY_PATH for the pip CUDA libs.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate focicnt

# Belt-and-suspenders: ensure TF can find the pip CUDA libs even if activate.d
# is absent on this env.
export LD_LIBRARY_PATH=$(python -c "import os,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(os.path.join(b,d,'lib') for d in os.listdir(b) if os.path.isdir(os.path.join(b,d,'lib'))))"):${LD_LIBRARY_PATH:-}

# Confirm the GPU is actually seen (into the log) before the heavy work.
nvidia-smi -L || echo "nvidia-smi not available"
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

export PYTHONPATH="scripts:${PYTHONPATH:-}"
python scripts/process_one.py "$CZI"
