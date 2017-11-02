#!/bin/bash

#SBATCH --job-name=Intuits-Simulation
#SBATCH --nodes=2
#SBATCH --cpus-per-task=25
#SBATCH --mem=50GB
#SBATCH --time=72:00:00

RANDOM=42

n=10  # number of people
p=0.2 # density of graph
mn=2  # minimum size of clique
mx=6  # maximum size of clique

workingDirectory="$1"

# initializes graph, and generates the cliques
python generate_graph.py --n=$n --p=$p --adjFile="$workingDirectory/output/graph.adj" --cliqueFile="$workingDirectory/output/cliques.txt"

# outputs random list of covers of the graph to do conversations
function generate_covers {
    python generate_random_cover.py --n=$n --mn=$mn --mx=$mx --rFile="$workingDirectory/output/cliques.txt" --wFile="temp" --seed=$1
}

numberOfTimesteps=10

for ((i=1;i<$numberOfTimesteps;i+=1))
do
    echo $i
    rand=$RANDOM
    time output=$(generate_covers $rand)
    readarray lines <<<"$output"
    cp -r "$workingDirectory/$i" "$workingDirectory/$((i+1))"
    for ((j=1;j<${#lines[@]};j+=1))
    do
        echo $j
        #srun --ntasks=1 --cpus-per-task=1 --exclusive --mem=3Gb
        python largeEchoChamber.py --seed=$RANDOM --readDirectory="$workingDirectory/$i" --writeDirectory="$workingDirectory/$((i+1))" --learnerNumbers="${lines[$j]}" &
    done
    wait
done
