#!/bin/bash
#
#SBATCH --time=06:00:00 
#SBATCH --nodes=1 
#SBATCH --nodelist=n06
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --job-name="test_simulation"
#SBATCH --mail-user=cowzin@snu.ac.kr
#SBATCH --mail-type=ALL
#SBATCH --output=out/py-%x.%j.out
#SBATCH --error=out/py-%x.%j.err

module python3
srun python /home/cowzin/code/test.py