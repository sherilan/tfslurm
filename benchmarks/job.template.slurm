#!/bin/sh
#SBATCH --partition=EPICALL
#SBATCH --time=00:15:00
#SBATCH --mail-type=ALL
#SBATCH --job-name="tfslurm_benchmarks"

# --- THESE ARE PASSED BY TEMPLATE
#SBATCH --mail-user={email_to}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={tasks}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --output={jobdir}/stdout
#SBATCH --gres={gpus}

# CD into submit dir so that relative filepaths works out
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
# Load Easybuild modules and activate virtualenv
echo 'Setting up environment'
# Clear all modules
module purge 
# Get dependencies
module load GCC/6.4.0-2.28  CUDA/9.0.176  OpenMPI/2.1.1
# Get python 3.6.3
module load Python/3.6.3
# Get the newest, (working) available version of CuDNN
module load cuDNN/7
# Activate virtualenv
source venv/bin/activate
# Call benchmark script
srun python run_benchmark.py --jobdir {jobdir} --batch_size {batch_size} --num-ps {num_ps} {extra_args}