#!/bin/bash
#
#SBATCH --time=4-00:00:00 
#SBATCH --nodes=1 
#SBATCH --nodelist=n06
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=50
#SBATCH --job-name="lottery_simulation_gpu"
#SBATCH --mail-user=cowzin@snu.ac.kr
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:100
#SBATCH --output=out/py-%x.%j.out
#SBATCH --error=out/py-%x.%j.err

module python3
srun python3 /home/cowzin/code/simulation_model_gpu.py