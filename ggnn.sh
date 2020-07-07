#!/bin/bash
#SBATCH -J task1
#SBATCH -p high
#SBATCH --workdir=/homedtic/cmorales
#SBATCH -o /homedtic/cmorales/log/%N.%J.task1.out # STDOUT
#SBATCH -e /homedtic/cmorales/log/%N.%J.task1.err # STDOUT
# Number of GPUs per node
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

export PATH="$HOME/project/anaconda3/bin:$PATH"
source activate tfgpu1
cd /homedtic/cmorales/cmol/ggnn
python chem_tensorflow_dense.py --pr identity --restrict_data $1 --log_dir borrar $2 $3 $4