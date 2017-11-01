#!/bin/bash

RANDOM=42

n=10  # number of people
p=0.2 # density of graph
mn=2  # minimum size of clique
mx=2  # maximum size of clique

# initializes graph, and generates the cliques
python generate_graph.py --n=$n --p=$p --adjFile="../output/graph.adj" --cliqueFile="../output/cliques.txt"

# outputs random list of covers of the graph to do conversations
function generate_covers {
    python generate_random_cover.py --n=$n --mn=$mn --mx=$mx --rFile="../output/cliques.txt" --wFile="temp" --seed=$1
}

workingDirectory="."
numberOfTimesteps=10

for ((i=1;i<$numberOfTimesteps;i+=1))
do
    echo $i
    rand=$RANDOM
    time output=$(generate_covers $rand)
    readarray lines <<<"$output"
done
