# args : number of people, p, where to write
# initializes graph, and generates the cliques
python generate_Graph.py --n=10 --p=.4 --adjFile="../output/graph.adj" --cliqueFile="../output/cliques.txt"

#args : number of people, mingroup size, max group size, where to read the all cliques, where to write the covers to(for abhinav to work on)
# generate a disjioint rand cover, and output the output to "COVER.txt"
python 2Bash_genRandCover.py 10 0 4 "../output/cliques.txt" "Cover.Txt"
#CURRENTLY, does not write to cover.txt
