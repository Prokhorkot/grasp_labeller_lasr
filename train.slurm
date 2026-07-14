#!/usr/bin/env bash
#SBATCH --job-name=grasp-classifier
#SBATCH --account=p_tactile_sensing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"

VENV_PYTHON="${SLURM_SUBMIT_DIR}/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Missing ${VENV_PYTHON}. Create the Linux virtual environment first." >&2
    exit 1
fi

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun "${VENV_PYTHON}" -m grasp_labeller_lasr.training.training "$@"
