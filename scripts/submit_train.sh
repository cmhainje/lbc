#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu
#SBATCH --job-name=lbc_train

module purge

singularity exec --nv \
    --overlay /home/ch4407/py/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif* \
    /bin/bash -c "source /ext3/env.sh; venv lbc; cd /home/ch4407/lbc/scripts; python train.py 50000 -c 4 -l 128"

