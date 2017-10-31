import networkx as nx
import argparse

parser = argparse.ArgumentParser(
    description=
    'Random graph generator and clique finder for using in suimulation')
parser.add_argument(
    '--seed', metavar='r', type=int, default=42, help='seed for randomness')
parser.add_argument(
    '--n', metavar='n', type=int, default=42, help='size of the network')
parser.add_argument(
    '--p', metavar='p', type=int, default=42, help='density of the network')
parser.add_argument(
    '--adjFile',
    metavar='f',
    required=True,
    type=str,
    help='File to print the adjacency list')
parser.add_argument(
    '--cliqueFile',
    metavar='f',
    required=True,
    type=str,
    help='File to print the maximal cliques')
args = parser.parse_args()

graph = nx.erdos_renyi_graph(args.n, args.p, args.seed)

nx.write_adjlist(graph, args.adjFile)

allCliques = nx.find_cliques(graph)

fId = open(args.cliqueFile, "w")
fId.write("\n".join(["\t".join(map(str, i)) for i in allCliques]))
fId.close()
