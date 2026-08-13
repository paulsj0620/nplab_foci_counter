#!/bin/bash
#SBATCH --job-name=foci_batch
#SBATCH --output=results/logs/slurm_%j.out    # stdout+stderr (%j = job id)
#SBATCH --error=results/logs/slurm_%j.err
#SBATCH --partition=capella                    # H100 partition (nodes c3-c156)
#SBATCH --nodes=1                              # this cluster requires node count
#SBATCH --gres=gpu:h100:1                      # request 1 full H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00                        # walltime; raise if many slides

# --- foci_counter batch on a SLURM GPU node ---------------------------------
# Submit from the project root:
#     sbatch slurm/run_batch.slurm
# Process specific slides instead of all HE:
#     sbatch slurm/run_batch.slurm "dataset/*_HE_*.czi"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

# 1) CUDA toolkit. If you installed TF with `pip install "tensorflow[and-cuda]"`
#    the CUDA libs are bundled and you can leave this commented out. If you used
#    plain `tensorflow`, load a matching module (available here: 12.1.1, 12.6.0,
#    12.8.0, 13.0.0). TF 2.21 pairs well with CUDA 12.x + cuDNN 9:
# module load CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0   # <-- uncomment if needed

# 2) Activate the conda env (see slurm/SETUP_HPC.md to create it once).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate focicnt

# 2b) Make TF find the pip-installed CUDA libs (nvidia-*-cu12 packages).
#     Without this TF prints "Cannot dlopen some GPU libraries" and falls back
#     to CPU even on a GPU node.
export LD_LIBRARY_PATH=$(python -c "import os,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(os.path.join(b,d,'lib') for d in os.listdir(b) if os.path.isdir(os.path.join(b,d,'lib'))))"):${LD_LIBRARY_PATH:-}

# 3) Report the GPU actually seen, into the log, before starting.
echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi || echo "nvidia-smi not available"
python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"

# 4) Run the batch (detection + output set + combined summary + logs).
export PYTHONPATH="scripts:${PYTHONPATH:-}"
python scripts/run_batch.py "$@"
