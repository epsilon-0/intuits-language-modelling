from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import sys
import pickle as pkl

edges = np.load(sys.argv[1])
d = pkl.load(open(sys.argv[2], "rb"))
print(d)

vertices = len(edges)
g = nx.Graph()
for i in range(vertices):
    g.add_node(i)

for i in range(vertices):
    for j in range(i + 1, vertices):
        if edges[i][j] >= 0:
            g.add_edge(i, j, weight=edges[i][j])

print("Number of vertices: ", vertices, "\nConnected status: ",
      nx.is_connected(g), "\nNumber of edges: ", len(g.edges))

components = sorted(nx.connected_components(g), key=len, reverse=True)
print("Number of connected components = ", len(components))

similarities_ = dict(nx.all_pairs_dijkstra_path_length(g))

similarities = [[0 for i in range(vertices)] for j in range(vertices)]
for i in range(vertices):
    for j in range(vertices):
        similarities[i][j] = 1
        if j in similarities_[i]:
            similarities[i][j] = similarities_[i][j]

mds = manifold.MDS(
    n_components=2, #3,
    max_iter=3000,
    eps=1e-9,
    dissimilarity="precomputed",
    n_jobs=1)
pos = mds.fit(similarities).embedding_

clf = PCA(n_components=2)
pos = clf.fit_transform(pos)
fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])

s = 100
plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
plt.legend(scatterpoints=1, loc='best', shadow=False)
plt.show()
