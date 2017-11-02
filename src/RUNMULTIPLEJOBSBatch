#!/bin/bash

#SBATCH --job-name=G09Test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=60GB
#SBATCH --time=12:00:00

module purge
module load gaussian/intel/g09e01

cd /share/apps/examples/slurm/multiple-jobs

for((i=0; i<12; i++)); do
    srun --ntasks=1 --cpus-per-task=4 --exclusive --mem=11Gb run-gaussian 2H4N-penta-5c.com > $i.log 2>&1 &
done

wait