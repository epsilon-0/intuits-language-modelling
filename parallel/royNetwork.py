import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import heapq

n  = 10 # people
p = .4
seed = 1
G = nx.generators.wheel_graph(10)#nx.generators.fast_gnp_random_graph(n,p,seed, directed = False)
#pos = nx.spring_layout(G)
#print G.edges()
for i in xrange(n):
    G.node[i]['marked']= 0
A = np.matrix([[1,1,1,1,1,0],[1,1,1,1,1,0],[1,1,1,1,1,0],[1,1,1,1,1,0],[1,1,1,1,1,1], [0,0,0,0,1,1]])
M = nx.from_numpy_matrix(A)


def getRandEdges(g, n): # in graph g, get n random edges
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

#getRandEdges(G, 5)
#print G.number_of_nodes()
#print G.number_of_edges()

def isInClique(g, n, Ns):
    # is n in the clique of Ns
    for i in xrange(len(Ns)):
        if(not g.has_edge(Ns[i], n)):
            return False
    return True
'''
def maxClique(g, Ns):
    #Ns = nodes in the existing clique
    n = []
    for i in xrange(len(Ns)):
        a = g.neighbors(Ns[i])
        for j in xrange(len(a)):
            if(g.node[a[i]]['marked'] == 1): #neighbor already added
                continue
            if(isInClique(Ns, ))
            n.add(a[j])
    return
    '''
def neighbors(g, v):
    # returns list of indices neighbors of node v
    return -1
def maxCliqueBronKerbosch(g,A, R,P,X):
    # R is the nodes in the clique
    # P is the nodes you consider adding to the clique
    # X is the nodes you do not want to add to the clique

    if(len(P)==0 and len(X)==0):
        return R    
    for vertex in P[:]:  # iterates of vertices in P
       # print "==" , vertex
        Rnew = R + [vertex]
       # Rnew.append(vertex)
        print Rnew
       # print "neighbors, hopefully "
       # print A[vertex]
       # print A[vertex][0][0]
       # print "======"
        #print vertex
        Pnew = [val for val in P if (A[vertex][val][0]!=0 and val != vertex)] # intersection of P and neighbors of vertex
        #print "P<" , P
        #print "Pnew", Pnew
        Xnew = [val for val in X if (A[vertex][val][0]!=0 and val != vertex)] # intersection of X and neighbors of vertex
        maxCliqueBronKerbosch(g,A, Rnew,Pnew,Xnew) # run a new iteration over a different recursion
        # for the P
        P.remove(vertex) # remove removes the first matching value
        X.append(vertex) # append adds the vertex to the list

def N(v, g):
    # returns index i, if n_v (neighbor of v) is in the list of g[v] (includes itself)
    return [i for i, n_v in enumerate(g[v]) if (n_v and i!= v)] # enumerate = list from an iterator
counter = 0
def bronk2(A, R, P, X): # for some reason, this does not give ONLY the largest cliques... gives at least 1 not-largest-lclique
    global counter
    #with pivots
    # both P and X are iterators, any = or. If not( P or X) = not P and not X (both empty)
    #if not a.any((P, X)): 

    if not ( any(P) or any(X)):
        print "R<", R
        yield R # R is the maximal clique, if X is empty and P is empty 
        #yield R # return iterator of R
        return # need this return (otherwise, XuP[0] might not point to anything)
    XuP = list(set().union(X,P))
    print "xup", XuP
    pivot = XuP[0]
    pivs = [v1 for v1 in P if (v1 not in N(pivot, A))]
    for v in pivs:
        R_v = R + [v]
        #print "P", P

        # P intersects with neighbors of v
        P_v = [v1 for v1 in P if (v1 in N(v, A))] # important! do not include self edges
        #print "PV", P_v
        X_v = [v1 for v1 in X if (v1 in N(v, A))]
        bronk2(A,R_v,P_v,X_v)
        for r in bronk2(A, R_v, P_v, X_v):
            yield r

        P.remove(v)
        X.append(v)
        # print "P is ", P
       # print "new P is", P
'''
def bronk2(A, R, P, X): # for some reason, this does not give ONLY the largest cliques... gives at least 1 not-largest-lclique
    global counter
    # both P and X are iterators, any = or. If not( P or X) = not P and not X (both empty)
    #if not a.any((P, X)): 

    if not ( any(P) or any(X)):
        print "R<", R
        yield R # R is the maximal clique, if X is empty and P is empty 
        #yield R # return iterator of R
    pivot = P[np.random.randint(0,len(P))] # choose a pivot vertex in P U X
    
    for v in P[:]:
        R_v = R[:] + [v]
        #print "P", P

        # P intersects with neighbors of v
        P_v = [v1 for v1 in P if (v1 in N(v, A))] # important! do not include self edges
        #print "PV", P_v
        X_v = [v1 for v1 in X if (v1 in N(v, A))]
        bronk2(A,R_v,P_v,X_v)
        for r in bronk2(A, R_v, P_v, X_v):
            yield r

        P.remove(v)
        X.append(v)
        # print "P is ", P
       # print "new P is", P
'''
        
A =nx.to_numpy_recarray(M)
print A
print A[0]
print A[0][0]
print N(1,A)
print "===="
print "range is ", range(6)
for bf in xrange(len(A[0])):
    print "Neighbors" , bf, N(bf, A)
print list(bronk2(A, [],range(6),[]))

graph = [[0,1,0,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[0,0,0,1,0,0]]
print list(bronk2(graph, [], range(len(graph[0])), []))
gg = np.matrix(graph)
ggg = nx.from_numpy_matrix(gg)
nx.draw(ggg, with_labels = True)
plt.show()

 # R is nodes it must have, P is edges it may have, X is edges it can't have
'''
print M.number_of_nodes()
print M.number_of_edges()
print M.nodes()
print 'poop'
print [i for i in range(0,M.number_of_nodes())]
print '--'
print nx.node_connected_component(M,0)
print maxCliqueBronKerbosch(M,nx.to_numpy_recarray(M),[],range(M.number_of_nodes()), [])
'''

#nx.draw(M,with_labels = True)
#plt.show()


def killClique(g, Ns):
    for i in xrange(len(Ns)):
        g.node[Ns[i]]['marked']= 0
	#neighbors 
def conversation(g):
	getRandEdges(g, 1)


#nx.draw(G)
#plt.show()







