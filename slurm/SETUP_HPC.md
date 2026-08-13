# Running foci_counter on the HPC (SLURM + GPU)

One-time setup, then submit the batch as a SLURM job. On CUDA GPUs StarDist
runs on the GPU automatically (no Metal issues like on Mac).

## 0. Check the GPU (do this first)

This cluster's GPU partitions (from `sinfo -o "%P %G %N" | grep -i gpu`):
- **`capella`** — full H100s, 4 per node (`gpu:h100:4`), nodes c3-c156. Use for batches.
- **`capella-interactive`** — H100 MIG slices, 12 GB (`gpu:h100_1g.12gb`). Use for quick tests.

Grab an interactive slice and inspect (note: this cluster requires `--nodes`):

```bash
srun --partition=capella-interactive --nodes=1 --gres=gpu:h100_1g.12gb:1 --pty bash
nvidia-smi                      # confirms the H100
module avail cuda 2>&1 | head   # CUDA/12.1.1, 12.6.0, 12.8.0, 13.0.0 available
```

## 1. Create the conda env (once)

```bash
cd <project>                    # where you git-cloned foci_counter
conda create -n focicnt python=3.12 -y
conda activate focicnt

# GPU TensorFlow (Linux/CUDA). This pulls matching CUDA libs via pip:
pip install "tensorflow[and-cuda]"
# Everything else:
pip install stardist csbdeep czifile imagecodecs scikit-image scikit-learn \
    scipy numpy openpyxl pillow
```

If your cluster prefers module-provided CUDA, `module load cuda/<ver> cudnn`
before `pip install tensorflow` (plain `tensorflow`) and uncomment the matching
line in `run_batch.slurm`.

Verify the GPU is visible:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# -> should list a GPU (non-empty), unlike on the Mac.
```

## 2. Copy the dataset

`dataset/` (~18 GB) is not in git. Transfer it to the project on the cluster:

```bash
rsync -avP dataset/ <user>@<hpc>:<project>/dataset/
```

## 3. Submit — parallel job array (recommended)

Process **one slide per array task**, each on its own H100, so all slides run
in parallel. `submit_array.sh` fills in the `--array` range automatically:

```bash
cd <project>
bash slurm/submit_array.sh                       # all *_HE_* slides
bash slurm/submit_array.sh "dataset/*_HE_*.czi"   # explicit glob
bash slurm/submit_array.sh "dataset/*.czi" 8      # cap 8 tasks at once
```

Per-task resources (edit in `slurm/run_one.slurm`): `--gres=gpu:h100:1`,
`--cpus-per-task=16`, `--mem=64G`, `--time=00:40:00`. With enough free GPUs the
whole set finishes in roughly the time of a single slide.

When all tasks are done, build the combined workbook (fast — caches exist):

```bash
python scripts/run_batch.py "dataset/*_HE_*.czi"   # -> results/summary_all.xlsx
```

### Alternative: one job, slides in sequence

```bash
sbatch slurm/run_batch.slurm                       # all *_HE_* slides, 1 GPU, serial
```

Edit for your site in either script: `--partition`, `--time`, `--mem`,
`--cpus-per-task`.

## 4. Monitor / results

```bash
squeue -u $USER                       # job status
tail -f results/logs/slurm_<jobid>.out
```

Outputs are identical to local runs:
- `results/<slide>/` — overview + gallery + xlsx per slide
- `results/summary_all.xlsx` — combined table
- `results/logs/` — batch log + SLURM out/err

## Notes

- **Internet**: the first StarDist run downloads the `2D_versatile_he` weights.
  Your cluster has internet, so this just works; the weights are then cached.
- **Parallelism**: this script uses one GPU for the whole batch. To parallelize
  across slides, submit a job array (one slide per task) — ask and we can add a
  `run_one.slurm` array variant once the GPU/partition details are known.
