# args : number of people, p, where to write
python 1Bash_genGraph.py 10 .4 "Graph" # initializes graph, and generates the cliques

#args : number of people, mingroup size, max group size, where to read the all cliques, where to write the covers to(for abhinav to work on)
python 2Bash_genRandCover.py 10 0 4 "GraphBronk.txt" "Cover.Txt" # generate a disjioint rand cover, and output the output to "COVER.txt"
#CURRENTLY, does not write to cover.txt