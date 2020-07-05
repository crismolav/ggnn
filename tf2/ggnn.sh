#!/bin/bash
#SBATCH -J task1
#SBATCH -p high
#SBATCH --workdir=/homedtic/cmorales
#SBATCH -o /homedtic/cmorales/log/%N.%J.task1.out # STDOUT
#SBATCH -e /homedtic/cmorales/log/%N.%J.task1.err # STDOUT
# Number of GPUs per node
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
# Select Intel nodes (with Infiniband)
#SBATCH --constraint=intel

export PATH="$HOME/project/anaconda3/bin:$PATH"
source activate tfgpu
cd /homedtic/cmorales/cmol/ggnn
python tf2/chem_tensorflow_dense.py --pr identity --restrict_data $1 --log_dir borrar $2 $3 $4