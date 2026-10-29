#!/bin/bash
#SBATCH --job-name=pyrpmd-cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --distribution=block:block
#SBATCH --time=00:10:00
#SBATCH --output=pyrpmd-cpu-%j.log
set -euo pipefail

# Submit from the repository root with the correct MPI/Python environment active.
# All nodes must share this directory and the configured PES/model files.
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYRPMD_TRAJECTORY_GROUPS=4
export PYRPMD_BEAD_WORKERS=2

# 8 MPI ranks = 4 trajectory groups * 2 bead ranks; Q=1, no coordinator rank.
# Use the site's matching --mpi=pmix / --mpi=pmi2 option if required by Slurm.
srun --cpu-bind=cores python -m pyrpmd run examples/drpmd/mpi.py \
  --output "runs/mpi-cpu-${SLURM_JOB_ID}"
