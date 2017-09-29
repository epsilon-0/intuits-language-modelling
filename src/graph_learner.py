import networkx as nx

import megaman.embedding.spectral_embedding as sp
import megaman.geometry.geometry as gm

import math

import numpy as np

import random

import sklearn.manifold

import scipy.spatial


def euclidean_distance(a, b):
    return math.sqrt(sum([(a[i] - b[i])**2 for i in range(len(a))]))


def crawl(graph,
          conversation,
          word,
          stopProbability=0.2,
          steps=0,
          maxSteps=10,
          jumpProbability=0.1):
    nbh = graph.neighbors(word)
    weights = np.array(
        [math.exp(-1 * graph.edge[word][n]['weight']) for n in nbh])
    weights /= sum(weights)
    if (len(nbh) == 0):
        return word
    nextWord = np.random.choice(nbh, p=weights)
    if (random.random() < jumpProbability):
        nextWord = random.choice(graph.nodes())
    if ((random.random() < stopProbability and nextWord not in conversation) or
            steps > maxSteps):
        return nextWord
    else:
        return crawl(graph, conversation, nextWord, stopProbability, steps + 1,
                     maxSteps)


class GraphLearner:
    def __init__(self, graph):
        """graph must be an undirected weighted graph of networkx"""
        self.graph = graph
        return

    def getNextWord(self, conversation, word, stopProbability=0.2,
                    maxSteps=10):
        return crawl(self.graph, conversation, word, stopProbability, 0,
                     maxSteps)

    def updateRepresentation(self,
                             conversation,
                             timestamp,
                             maxhistory,
                             forgetPercent=1):
        added_edges = 0
        conv = list(conversation)
        for i in range(len(conv)):
            for j in range(i + 1, len(conv)):
                w1 = conv[i]
                w2 = conv[j]
                self.graph.node[w1]['count'] += 1
                self.graph.node[w2]['count'] += 1
                if (w2 not in self.graph.edge[w1]):
                    self.graph.add_edge(
                        w1, w2, {'weight': 0.0,
                                 'count': 1,
                                 'time': timestamp})
                    added_edges += 1
                else:
                    self.graph.edge[w1][w2]['count'] += 1
                    self.graph.edge[w1][w2]['time'] = timestamp
                self.graph.graph['count'] += 2

        old_edges = []
        for e in self.graph.edges(data=True):
            if (e[2]['time'] < timestamp - maxhistory):
                old_edges.append((e[0], e[1]))
        old_edges = np.array(old_edges)
        r_edges = 0
        if (len(old_edges) > 0):
            removed_edges = old_edges[np.random.choice(
                len(old_edges),
                replace=False,
                size=int((len(old_edges) * forgetPercent) / 100))]
            for e in removed_edges:
                self.graph.graph[
                    'count'] -= self.graph.node[e[0]]['count'] + self.graph.node[e[1]]['count']
                self.graph.node[e[0]]['count'] -= self.graph.edge[e[0]][e[1]][
                    'count']
                self.graph.node[e[1]]['count'] -= self.graph.edge[e[0]][e[1]][
                    'count']
                self.graph.graph[
                    'count'] += self.graph.node[e[0]]['count'] + self.graph.node[e[1]]['count']
                self.graph.remove_edge(e[0], e[1])
            r_edges = len(removed_edges)
        return (added_edges, r_edges)

    def updateWeights(self):
        for i in self.graph.edges():
            self.graph.edge[i[0]][i[1]]['weight'] = 2.0 * self.graph.edge[i[
                0]][i[1]]['count'] / (self.graph.node[i[0]]['count'] +
                                      self.graph.node[i[1]]['count'])
        return

    def getDistanceMatrix(self):
        self.updateWeights()
        return nx.all_pairs_dijkstra_path_length(self.graph)

    def getSpectralDistanceMatrix(self, nodes, dimension):
        self.updateWeights()
        subgraphs = list(nx.connected_component_subgraphs(self.graph))
        distances = {}
        for subg in subgraphs:
            subnodes = subg.nodes()
            adjacency = nx.adjacency_matrix(subg, subnodes)
            lower_embedding = sklearn.manifold.spectral_embedding(
                adjacency, n_components=dimension)
            tdistances = {
                subnodes[i]: {
                    subnodes[j]: scipy.spatial.distance.euclidean(
                        lower_embedding[i], lower_embedding[j])
                    for j in range(len(subnodes))
                }
                for i in range(len(subnodes))
            }
            distances.update(tdistances)
        return distances

    def getDiffusionDistanceMatrix(self, nodes, dimension=8,
                                   diffusion_time=10):
        self.updateWeights()
        subgraphs = list(nx.connected_component_subgraphs(self.graph))
        print("Number of components = {:d}".format(len(subgraphs)))
        distances = {}
        for subg in subgraphs:
            subnodes = subg.nodes()
            geom = gm.Geometry()
            geom.set_affinity_matrix(nx.adjacency_matrix(subg, subnodes))
            geom.set_laplacian_matrix(
                nx.normalized_laplacian_matrix(subg, subnodes))
            lower_embedding = sp.spectral_embedding(
                geom,
                n_components=dimension,
                drop_first=True,
                diffusion_maps=True,
                diffusion_time=diffusion_time)
            tdistances = {
                subnodes[i]: {
                    subnodes[j]: euclidean_distance(lower_embedding[i],
                                                    lower_embedding[j])
                    for j in range(len(subnodes))
                }
                for i in range(len(subnodes))
            }
            distances.update(tdistances)
        return distances
