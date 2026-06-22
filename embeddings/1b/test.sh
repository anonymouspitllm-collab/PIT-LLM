#!/bin/bash -l
#SBATCH --job-name=1b-embed-test
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/$USER/repo/embeddings/1b/logs/%j.out
#SBATCH --error=/home/$USER/repo/embeddings/1b/logs/%j.err

module purge
module load python cuda 2>/dev/null || true

VENV=/scratch/$USER/venvs/chronogpt
export PATH="$VENV/bin:$PATH"

SCRIPT_DIR=/home/$USER/repo/embeddings/1b
cd "$SCRIPT_DIR"
export PYTHONPATH="/home/$USER/repo:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p /home/$USER/repo/embeddings/1b/logs
mkdir -p /scratch/$USER/embeddings/1b

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 main.py --test

echo "Finished : $(date)"
