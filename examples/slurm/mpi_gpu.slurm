#!/bin/bash
#SBATCH --job-name=pyrpmd-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --distribution=block:block
#SBATCH --time=00:10:00
#SBATCH --output=pyrpmd-gpu-%j.log
set -euo pipefail

# Activate the site's compatible CUDA/PyTorch/MPI environment before submission.
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYRPMD_BACKEND=mpi
export PYRPMD_TRAJECTORY_GROUPS=8

# Each node: 2 visible GPUs * 2 trajectory processes/GPU = 4 MPI ranks.
# All local ranks must see both allocated GPUs. Device selection then uses local
# MPI rank. Each rank loads its own model, so budget two model copies per card.
# Adapt --mpi and GPU allocation directives to the cluster's scheduler/runtime.
srun --cpu-bind=cores --gpu-bind=none python -m pyrpmd run examples/drpmd/gpu.py \
  --output "runs/mpi-gpu-${SLURM_JOB_ID}"
