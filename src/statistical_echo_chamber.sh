#!/bin/bash

#SBATCH --job-name=Intuits-Simulation
#SBATCH --nodes=2
#SBATCH --cpus-per-task=25
#SBATCH --mem=50GB
#SBATCH --time=72:00:00

#module purge
#module load python3/intel/3.6.3
#module load parallel/20171022

RANDOM=42

n=40  # number of people
p=0.6  # density of graph
mn=2  # minimum size of clique
mx=8  # maximum size of clique
numberOfTimesteps=80  # number of total conversations to have

workingDirectory="$1/$(date +%Y%m%d_%H%M%S)"

# for each person make a learner and store it in initial folder
mkdir -p "$workingDirectory/1"
for ((i=0;i<$n;i+=1))
do
    # for running on local machine
    python3 generate_statistical_learner.py --seed=$RANDOM --writeFile="$workingDirectory/1/$i"
    # srun --ntasks=1 --cpus-per-task=1 --exclusive --mem=1Gb python3 generate_vecspace_learner.py --seed=$RANDOM --writeFile="$workingDirectory/1/$i"
done

echo "Made initial representations of learners"

# initializes graph, and generates the cliques
mkdir -p "$workingDirectory/graph"
python3 generate_graph.py --n=$n --p=$p --adjFile="$workingDirectory/graph/graph.adj" --cliqueFile="$workingDirectory/graph/cliques.txt"

# outputs random list of covers of the graph to do conversations
function generate_covers {
    python3 generate_random_cover.py --n=$n --mn=$mn --mx=$mx --rFile="$workingDirectory/graph/cliques.txt" --wFile="temp" --seed=$1
}

for ((i=1;i<$numberOfTimesteps;i+=1))
do
    rand=$RANDOM
    time output=$(generate_covers $rand)
    readarray lines <<<"$output"
    cp -r "$workingDirectory/$i" "$workingDirectory/$((i+1))"
    for ((j=1;j<${#lines[@]};j+=1))
    do
        # for running on local machine
        python3 largeStatisticalEchoChamber.py --seed=$RANDOM --readDirectory="$workingDirectory/$i" --writeDirectory="$workingDirectory/$((i+1))" --learnerNumbers="${lines[$j]}" &
        # srun --ntasks=1 --cpus-per-task=1 --exclusive --mem=3Gb python3 largeEchoChamber.py --seed=$RANDOM --readDirectory="$workingDirectory/$i" --writeDirectory="$workingDirectory/$((i+1))" --learnerNumbers="${lines[$j]}" --num_iters=2 --step_size=0.01 &
    done
    wait
done
