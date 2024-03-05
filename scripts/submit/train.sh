#!/bin/bash

#SBATCH -J train
#SBATCH -t 01:00:00
#SBATCH --mem 10GB
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu
#SBATCH --mail-type=all
#SBATCH --mail-user=ch4407@nyu.edu
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.out
#SBATCH --array=1,2,4,8,16,32,64

module purge
singularity exec --nv \
    --overlay /home/ch4407/py/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif* \
    /bin/bash -c \
    "source /ext3/env.sh; venv lbc; cd /home/ch4407/lbc/scripts; \
    python train.py 100_000 -c 4 -l $SLURM_ARRAY_TASK_ID --camera r; \
    python train.py 100_000 -c 4 -l $SLURM_ARRAY_TASK_ID --camera b;"

#     'source /ext3/env.sh; venv lbc; cd /home/ch4407/lbc/scripts; \
#     for x in 1 2 4 8 16 32 64; do \
#         python train.py 50_000 -c 4 -l $x --camera r; \
#         python train.py 50_000 -c 4 -l $x --camera b; \
#     done'

# note: single quotes are important for the command above!
# single quotes do *not* do variable interpolation, so the $x doesn't get
# interpolated when bash interprets the string. because of this, the $x will is
# persisted until the string is run as a command, so the for loop will work!

