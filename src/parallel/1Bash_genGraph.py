import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import csv
import pickle

# program takes 3 arguments
# number of graphs, probability of edge, and where to print
# creates 2 txt files 1- the Adj list of the graph, 2- the list of the "All Cliques"
n = int(sys.argv[1])# 10  # people
p = float(sys.argv[2])
FileToPrint = sys.argv[3]
seed = 1

def write(nameTxt, data):

	np.savetxt(nameTxt + ".txt", data)
	#print np.loadtxt(nameTxt)
	#np.savetxt(nameTxt, data, delimiter = ',')
def read(nameTxt):
	return np.loadtxt(nameTxt)
# graph Erdos
#graphE = nx.generators.fast_gnp_random_graph(n,p,seed, directed = False)
def N(v, g):
    return [i for i, n_v in enumerate(g[v])
            if (n_v and i != v)]  # enumerate = list from an iterator

counter = 0
def bronk2(
        A, R, P, X
):  # for some reason, this does not give ONLY the largest cliques... gives at least 1 not-largest-lclique
    global counter
    #with pivots
    # both P and X are iterators, any = or. If not( P or X) = not P and not X (both empty)
    #if not a.any((P, X)):

    if not (any(P) or any(X)):
        #    print "R<", R
        yield R  # R is the maximal clique, if X is empty and P is empty
        #yield R # return iterator of R
        return  # need this return (otherwise, XuP[0] might not point to anything)
    XuP = list(set().union(X, P))
    #print "xup", XuP
    pivot = XuP[0]
    pivs = [v1 for v1 in P if (v1 not in N(pivot, A))]
    for v in pivs:
        R_v = R + [v]
        #print "P", P

        # P intersects with neighbors of v
        P_v = [v1 for v1 in P
               if (v1 in N(v, A))]  # important! do not include self edges
        #print "PV", P_v
        X_v = [v1 for v1 in X if (v1 in N(v, A))]
        bronk2(A, R_v, P_v, X_v)
        for r in bronk2(A, R_v, P_v, X_v):
            yield r
        P.remove(v)
        X.append(v)




graphE = nx.erdos_renyi_graph(n, p)
peopleTalking = np.zeros(n)
for i in range(n):
    graphE.node[i]['marked'] = 0  #irrelevant
graphAdj = np.array(nx.to_scipy_sparse_matrix(graphE).todense())
print(graphAdj)
write(FileToPrint,graphAdj)
s = FileToPrint + "bronk" + ".txt"
#thefile = open(s, 'w')

allCliques = list(bronk2(graphAdj, [], list(range(n)), []))

fId = open(s, "w")
fId.write("\n".join(["\t".join(map(str, i)) for i in allCliques]))
fId.close()


#for item in allCliques:
#  thefile.write("%s\n" % item)





