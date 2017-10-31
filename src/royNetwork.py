from util import learnerProcess
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import heapq


def getRandEdges(g, n):  # in graph g, get n random edges
    visited = set()
    results = []
    edges = random.sample(g.edges(), n)
    print(edges)
    for edge in edges:
        if edge[0] in visited or edge[1] in visited:
            continue
        results.append(edge)
        if len(results) == n:
            break
        visited.update(edge)
    return results


def isInClique(g, n, Ns):
    # is n in the clique of Ns
    for i in range(len(Ns)):
        if (not g.has_edge(Ns[i], n)):
            return False
    return True


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


cliquesTalking = 0
cliqueConversations = []

def killClique(g, i, heapCliques, peopleTalking, allCliques):
    cliquesTalking -= 1
    cliqueConversations[i] = 0
    for b in range(allCliques[i]):
        peopleTalking[i] = 0
    heapq.heappush(heapCliques, i)

    for a in allCliques[i]:
        g.node[allCliques[i][a]]['marked'] = 0


def birthClique(g, i, peopleTalking, allCliques):
    cliquesTalking += 1
    cliqueConversations[i] = 1
    for a in allCliques[i]:
        peopleTalking[i] = 1


    # g.node[allCliques[i][a]]['marked']= 1
def everyoneQuiet(g, peopleTalking, clique):
    for i in range(len(clique)):
        if (peopleTalking[clique[i]] > 0):
            return False
    return True


def conversation(g, heapCliques, cliquesTalking, peopleTalking, allCliques):
    # heap cliques is a heap of indices (from allCliques) of cliques not talking
    #cliques talking is a list of all the cliques that have people talking
    cliq = []
    ind = -1
    if (cliquesTalking >= len(allCliques)):
        return -1
    while (len(heapCliques) > 0):
        tempInd = heapq.heappop(heapCliques)
        if (everyoneQuiet(g, peopleTalking, allCliques[tempInd])):
            ind = tempInd
    birthClique(g, ind, peopleTalking, allCliques)
    return i


'''
    r = np.random.randomint(0, len(allCliques)-cliquesTalking)
    count = 0
    for i in range(len(allCliques)):
        if(allCliques[i]==0):
            count+=1
        if(count == r):
            birthClique(g, c)
            return
'''


def disjointCovers(covers, n, sizeMin, sizeMax):
    # n total nodes
    # returns list of covers between the range sizeMin to SizeMax, which are disjoint

    ret = []
    used = np.zeros(n)
    for i in range(len(covers)):
        ind = np.random.randint(0, len(covers))
        size = len(covers[ind])
        if (size < sizeMin):
            continue
        disjoint = True
        for a in range(len(covers[ind])):
            if (used[covers[ind][a]] == 1):
                disjoint = False
        if (disjoint == True):
            if (sizeMax < len(covers[ind])):
                asd = []
                count = 0
                while (count < sizeMax):
                    ind2 = np.random.randint(0, len(covers[ind]))
                    if (used[covers[ind][ind2]] == 0):
                        asd.append(covers[ind][ind2])
                        used[covers[ind][ind2]] = 1
                        count += 1
                ret.append(asd)
            else:
                for b in range(len(covers[ind])):
                    used[covers[ind][b]] = 1
                ret.append(covers[ind])
    return ret

    return 1


n = 10  # people
p = .4
seed = 1
# graph Erdos
#graphE = nx.generators.fast_gnp_random_graph(n,p,seed, directed = False)
graphE = nx.erdos_renyi_graph(n, p)
peopleTalking = np.zeros(n)
for i in range(n):
    graphE.node[i]['marked'] = 0  #irrelevant
graphAdj = np.array(nx.to_scipy_sparse_matrix(graphE).todense())
print(graphAdj)

allCliques = list(bronk2(graphAdj, [], range(n), []))
#print allCliques
conversationsToHave = disjointCovers(allCliques, len(graphAdj[0]), 0, 2)

print("size of disjoint conversations to have", len(conversationsToHave))
'''
for i in range(0, 10):
    print disjointCovers(allCliques,len(graphAdj[0]), 0, 2)
'''

talksPerConversation = 200
numConversations = 10
conversationMin = 2
groupConversationMax = 3

initializeLearners(n)
allCliques = list(bronk2(graphAdj, [], range(n), []))
for t in range(0, numConversations):
    print("conversation ", t)
    conversationsToHave = disjointCovers(
        allCliques, len(graphAdj[0]), conversationMin, groupConversationMax
    )  # this is slightly randomized process to generate
    #a set of disjoint covers
    print(conversationsToHave)
    for i in range(0, len(conversationsToHave)):
        learnerProcess(conversationsToHave[i],
                       talksPerConversation)  #abhinav code
nx.draw(graphE, with_labels=True)
plt.show()

###Working here now:
### have conversation (run code for convergence, and update reps), and then kill clique after

#i =conversation(graphAdj,heapQuietCliques, cliquesTalking, peopleTalking, allCliques)
#killClique(graphAdj,i ,heapQuietCliques, peopleTalking,allCliques)

###
#nx.draw(G)
#plt.show()

#IGNORE:
''' OTHER
cliqueConversations = np.zeros(len(allCliques))
heapQuietCliques = []# heap cliques is a heap of indices (from allCliques) of cliques not talking
for i in range(0, len(allCliques)):
    heapq.heappush(heapQuietCliques ,i)

cliquesTalking = 0
'''

#graphAdj = nx.to_numpy_recarray(graphE, dtype=[('weight',float),('marked',int)]) ) # adjacency matrix of graph
'''
getRandEdges(g, 1)
A =nx.to_numpy_recarray(M)
print A
print A[0]
print A[0][0]
print N(1,A)
print "===="
print "range is ", range(6)
for bf in range(len(A[0])):
    print "Neighbors" , bf, N(bf, A)
print list(bronk2(A, [],range(6),[]))

graph = [[0,1,0,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[0,0,0,1,0,0]]
print list(bronk2(graph, [], range(len(graph[0])), []))
gg = np.matrix(graph)
ggg = nx.from_numpy_matrix(gg)
nx.draw(ggg, with_labels = True)
plt.show()

'''
'''

G = nx.generators.wheel_graph(10)#nx.generators.fast_gnp_random_graph(n,p,seed, directed = False)
#pos = nx.spring_layout(G)
#print G.edges()
for i in range(n):
    G.node[i]['marked']= 0
A = np.matrix([[1,1,1,1,1,0],[1,1,1,1,1,0],[1,1,1,1,1,0],[1,1,1,1,1,0],[1,1,1,1,1,1], [0,0,0,0,1,1]])
M = nx.from_numpy_matrix(A)



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

'''
