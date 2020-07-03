#!/bin/bash
#SBATCH -J task1
#SBATCH -p high
#SBATCH --workdir=/homedtic/cmorales
#SBATCH -o /homedtic/cmorales/log/%N.%J.task1.out # STDOUT
#SBATCH -e /homedtic/cmorales/log/%N.%J.task1.err # STDOUT
# Number of GPUs per node
#SBATCH --gres=gpu:1

module load CUDA/10.0.130
cd /homedtic/cmorales
source tf1/bin/activate
cd cmol/ggnn
python chem_tensorflow_dense.py --pr identity --restrict_data $1 --log_dir borrar $2