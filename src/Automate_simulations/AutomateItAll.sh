#!/bin/bash

#SBATCH --job-name=Intuits-Simulation
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=72:00:00
#SBATCH --mail-user=rr2635@nyu.edu
#SBATCH --array=0-20
module purge
module load python3/intel/3.6.3
#module load parallel/20171022

RANDOM=42
limits=( 10, 20, 26, 39, 48)
ns=( 20, 40, 100, 200 )
ns=( 20)
mns=( 2)
mxs=(4, 6, 10, 16)
ps=( 2, 10, 15, 50, 60, 100 )
Ts=( 50, 100, 200 )


ns=( 20, 40, 100, 200 )
ns=( 20)

mns=( 2)

mxs=(4, 6, 10, 16)
mxs=(8)

ps=( 2, 10, 15, 30, 50, 60, 100 )
ps=( 50)

Ts=( 100, 500, 1000 )
Ts=( 10 )
#workingDirectory="simulation/hello"
SCRIPT_PATH="./statistical_echo_chamber.sh"

#repeats= 5
for n in "${ns[@]}"
do
	for mn in "${mns[@]}"
	do
		for mx in "${mxs[@]}"
		do
			for p in "${ps[@]}"
			do
				for T in "${Ts[@]}"
				do

workingDirectory="simulations/trial"$run"_n_"$n"_p_"$p"_mn_"$mn"_mx_"$mx"_T_"$T"_$(date +%Y%m%d_%H%M%S)"
#bash $SCRIPT_PATH "simulations" "$n" "$p" "$mn" "$mx" "$T" $SLURM_ARRAY_TASK_ID "$workingDirectory" # the trial number is used for the random_seed
bash $SCRIPT_PATH "simulations" "$n" "$p" "$mn" "$mx" "$T" 1 "$workingDirectory" # the trial number is used for the random_seed

				done
			done
		done
	done
done

