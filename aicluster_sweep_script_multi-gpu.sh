#!/bin/sh
# SBATCH directives
#SBATCH -J anomalyov-vision-encder
#SBATCH -o ./out/%j.out  # Output file
##SBATCH -o ./out/%j.out
#SBATCH -t 3-00:00:00  # Run time (D-HH:MM:SS)

#### Select GPU
#SBATCH -p 3090              # Partition
##SBATCH -p 3090              # Partition
##SBATCH -p A6000
#SBATCH --nodes=1            # Number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # Number of CPUs
#SBATCH --gres=gpu:1         # Number of GPUs

cd $SLURM_SUBMIT_DIR

srun -I /bin/hostname
srun -I /bin/pwd
srun -I /bin/date

## Load modules
module purge
module load cuda/13.0.2

## Python Virtual Environment
echo "Start"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export project_name="prj-mmfm-ad-da" # W&B Project Name
export agent="fh55mfor" # W&B Sweep Agent ID

## uv
echo "source .venv/bin/activate"
source .venv/bin/activate

# Run W&B Sweep Agent
srun wandb agent postech-log-mmfm/$project_name/$agent

date

squeue --job $SLURM_JOBID

echo "##### END #####"