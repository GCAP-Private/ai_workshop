#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --output=hello.out
#SBATCH --error=hello.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --partition="maggiori"

unset MODULEPATH
unset MODULESHOME
CONTAINER_PATH=/oak/stanford/groups/maggiori/GCAP/data/scratch/xwfeng/apptainers/hello.sif
apptainer run --nv ${CONTAINER_PATH}