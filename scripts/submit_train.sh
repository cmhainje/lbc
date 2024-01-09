#!/bin/bash

#SBATCH -J lbc_train
#SBATCH -t 00:30:00
#SBATCH --mem 10GB
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu
#SBATCH --mail-type=all
#SBATCH --mail-user=ch4407@nyu.edu
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

module purge

singularity exec --nv \
    --overlay /home/ch4407/py/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif* \
    /bin/bash -c "source /ext3/env.sh; venv lbc; cd /home/ch4407/lbc/scripts; python train.py 50_000 -c 4 -l 8"

