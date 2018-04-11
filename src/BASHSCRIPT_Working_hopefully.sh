#!/bin/bash

#SBATCH --job-name=Intuits-Simulation
#SBATCH --nodes=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=01:00:00

module purge
module load python3/intel/3.6.3
module load parallel/20171022

envs=$(for e in $(env | grep -v '{' | grep -v '}' | grep -v '()' | grep -v _= | cut -d= -f1); do echo "--env $e"; done)
nodes=$(for node in $(scontrol show hostname $SLURM_NODELIST); do echo "--sshlogin ${node}-ib0"; done)

wrk_dir=/beegfs/$USER

RANDOM=42

n=10  # number of people
p=0.2 # density of graph
mn=2  # minimum size of clique
mx=6  # maximum size of clique

workingDirectory="$wrk_dir/$(date +%Y%m%d_%H%M%S)"

# for each person make a learner and store it in initial folder
mkdir -p "$workingDirectory/1"
for ((i=0;i<$n;i+=1)); do
    # for running on local machine
    # python generate_vecspace_learner.py --seed=$RANDOM --writeFile="$workingDirectory/1/$i"
    echo "python3 generate_vecspace_learner.py --seed=$RANDOM --writeFile=$workingDirectory/1/$i > $i.log 2>&1"
done |  parallel --no-notice \
    --workdir $(pwd) \
    --jobs 10  \
    --tmpdir $SLURM_JOB_TMP \
    $envs \
    $nodes

echo "Made initial representations of learners"

# initializes graph, and generates the cliques
mkdir -p "$workingDirectory/graph"
python3 generate_graph.py --n=$n --p=$p --adjFile="$workingDirectory/graph/graph.adj" --cliqueFile="$workingDirectory/graph/cliques.txt"

# outputs random list of covers of the graph to do conversations
function generate_covers {
    python3 generate_random_cover.py --n=$n --mn=$mn --mx=$mx --rFile="$workingDirectory/graph/cliques.txt" --wFile="temp" --seed=$1
}

numberOfTimesteps=10

envs=$(for e in $(env | grep -v '{' | grep -v '}' | grep -v '()' | grep -v _= | cut -d= -f1); do echo "--env $e"; done)

for ((i=1;i<$numberOfTimesteps;i+=1))
do
    rand=$RANDOM
    time output=$(generate_covers $rand | awk '{print $1}')
    readarray lines <<<"$output"
    cp -r "$workingDirectory/$i" "$workingDirectory/$((i+1))"
    for ((j=1;j<${#lines[@]};j+=1)); do
	index=$(echo ${lines[$j]})
        # for running on local machine
        # python largeEchoChamber.py --seed=$RANDOM --readDirectory="$workingDirectory/$i" --writeDirectory="$workingDirectory/$((i+1))" --learnerNumbers="${lines[$j]}" --num_iters=10 --step_size=0.01 &
        echo "python3 largeEchoChamber.py --seed=$RANDOM --readDirectory=$workingDirectory/$i --writeDirectory=$workingDirectory/$((i+1)) --learnerNumbers=$index --num_iters=2 --step_size=0.01 > $i-$j.log 2>&1"
    done | parallel --no-notice \
        --workdir $(pwd) \
        --jobs 10 \
	--tmpdir $SLURM_JOB_TMP \
        $envs \
        $nodes
done
