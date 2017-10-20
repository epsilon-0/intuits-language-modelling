import networkx as nx
import matplotlib.pyplot as plt
import random

n  = 10 # people
p = .4
seed = 1
G = nx.generators.fast_gnp_random_graph(n,p,seed, directed = False)
#pos = nx.spring_layout(G)
#print G.edges()

def getRandEdges(g, n):
    visited = set()
    results = []
    edges = random.sample(g.edges(), n)
    print edges
    for edge in edges:
        if edge[0] in visited or edge[1] in visited:
            continue
        results.append(edge)
        if len(results) == n:
            break
        visited.update(edge)
    return results

getRandEdges(G, 5)
print G.number_of_nodes()
print G.number_of_edges()

nx.draw(G)
plt.show()