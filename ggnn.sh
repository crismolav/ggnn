#!/bin/bash
#SBATCH -J task1
#SBATCH -p high
#SBATCH --workdir=/homedtic/cmorales
#SBATCH -o /homedtic/cmorales/log2/%N.%J.task1.out # STDOUT
#SBATCH -e /homedtic/cmorales/log2/%N.%J.task1.err # STDOUT
# Number of GPUs per node
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

export PATH="$HOME/project/anaconda3/bin:$PATH"
source activate tfgpu
cd /homedtic/cmorales/cmol/ggnn
python tf2/chem_tensorflow_dense.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11}