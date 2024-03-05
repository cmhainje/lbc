#!/bin/bash

#SBATCH -J lbc_prep
#SBATCH -t 03:00:00
#SBATCH --mem 8GB
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=all
#SBATCH --mail-user=ch4407@nyu.edu
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

module purge

RAW_DIR=/scratch/ch4407/lbc/data/arcs
PROC_DIR=/scratch/ch4407/lbc/data-processed/arcs

singularity exec --nv \
    --overlay /home/ch4407/py/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/ubuntu-20.04.4.sif* \
    /bin/bash -c \
    "source /ext3/env.sh; venv lbc; cd /home/ch4407/lbc/scripts; \
    python preprocess.py $RAW_DIR $PROC_DIR --camera r && \
    python preprocess.py $RAW_DIR $PROC_DIR --camera b"

