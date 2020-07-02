#!/bin/bash
#SBATCH -J task1
#SBATCH -p high
#SBATCH --workdir=/homedtic/cmorales
#SBATCH -o /homedtic/cmorales/log/%N.%J.task1.out # STDOUT
#SBATCH -e /homedtic/cmorales/log/%N.%J.task1.err # STDOUT
# Number of GPUs per node
#SBATCH --gres=gpu:1
#SBATCH --mem=5G

# export PATH="$HOME/project/anaconda3/bin:$PATH"
# source activate goog3
# cd /homedtic/cmorales/cmol/ggnn


cd /homedtic/cmorales
source tf1/bin/activate
ml load TensorFlow/1.14.0-foss-2017a-Python-3.6.4
ml load TensorFlow-gpu/1.14.0-foss-2017a-Python-3.6.4-CUDA-10.0.130
module load CUDA/8.0.61
cd cmol/ggnn
python chem_tensorflow_dense.py --pr identity --restrict_data $1 --log_dir borrar $2