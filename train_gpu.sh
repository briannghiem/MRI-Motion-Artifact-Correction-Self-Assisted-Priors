#!/bin/bash -l
#SBATCH --account=uludag_gpu
#SBATCH --job-name=train_nn
#SBATCH --nodes=1
#SBATCH -p gpu #partition
#SBATCH --gres=gpu:v100:1               # max 3 GPUs
#SBATCH --cpus-per-task=2         # max 41 CPUs
#SBATCH --mem=40G              # max > 200GB memory per node
#SBATCH --time=3-0:00:00 #wall time is 3 days
#SBATCH --mail-user=brian.nghiem@rmp.uhn.ca
#SBATCH --mail-type=ALL #mail notifications

# Build output directory
export HOME=/cluster/projects/uludag/Brian/RMC_repos/MRI-Motion-Artifact-Correction-Self-Assisted-Priors
cd ${HOME}

conda activate /cluster/home/brian.nghiem/.conda/envs/tf_env

python ${HOME}/main.py > ${HOME}/output_${SLURM_JOBID}.txt
