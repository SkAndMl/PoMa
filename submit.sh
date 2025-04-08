#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1:mem=48gb
#PBS -l walltime=48:00:00
#PBS -P personal-sathyakr
#PBS -N train_pos_matrix

# Commands start here
cd ${PBS_O_WORKDIR}
module unload gcc/12.1.0-nscc
module load gcc/8.1.0
module load cuda/11.8.0
module load python/3.10.9
source venv/bin/activate

python3.10 train.py