#!/bin/bash
#
#SBATCH --time=4-00:00:00 
#SBATCH --nodes=1 
#SBATCH --nodelist=n05
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=7
#SBATCH --job-name="lottery_simulation3_12"
#SBATCH --mail-user=cowzin@snu.ac.kr
#SBATCH --mail-type=ALL
#SBATCH --output=out/py-%x.%j.out
#SBATCH --error=out/py-%x.%j.err

module python3

srun  python3 /home/cowzin/code/simulation_model3_12.py

