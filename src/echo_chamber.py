# imports
import argparse

import networkx as nx

import random
import math

import numpy as np

from graph_learner import *

# command line arguments
parser = argparse.ArgumentParser(
    description='Simulation for learners in echo chamber')
parser.add_argument(
    '--iterations',
    metavar='n',
    type=int,
    default=10,
    help='number of conversations to have')
parser.add_argument(
    '--inferencesteps',
    metavar='s',
    type=int,
    default=2,
    help='will generate a distance matrix every s steps')
parser.add_argument(
    '--outputfile',
    metavar='o',
    type=str,
    default='output',
    help=
    'prefix of file to write the output distances (will make multiple files with extensions)'
)
parser.add_argument(
    '--convlength',
    metavar='c',
    type=int,
    default=10,
    help='maximum length of conversation in the echo chamber')
parser.add_argument(
    '--inputfile1',
    required=True,
    metavar='i',
    type=str,
    help='input to learn intial random representations from for learner 1')
parser.add_argument(
    '--inputfile2',
    required=True,
    metavar='i',
    type=str,
    help='input to learn intial random representations from for learner 2')
parser.add_argument(
    '--maxhistory',
    metavar='g',
    type=int,
    default=100,
    help=
    'maximum time for a learner to remember similarity between words (will start forgetting after this time)'
)
parser.add_argument(
    '--forgetpercent',
    metavar='f',
    type=float,
    default=1,
    help='amount of edges to forget if you are using a graph representation')
parser.add_argument(
    '--seed', metavar='r', type=int, default=42, help='seed for randomness')
parser.add_argument(
    '--stopprob',
    metavar='p',
    type=float,
    default=0.05,
    help='probability of ending a conversation at any stage')
parser.add_argument(
    '--dimension',
    metavar='d',
    type=int,
    default=10,
    help='dimension for spectral embedding')
args = parser.parse_args()

random.seed(args.seed)

# read input file
learners = []
vocabSet = set()
vocab = []

fId1 = open(args.inputfile1, "r")
lines1 = fId1.readlines()
lines1 = [i.strip().split(' ') for i in lines1]
fId1.close()

fId2 = open(args.inputfile2, "r")
lines2 = fId2.readlines()
lines2 = [i.strip().split(' ') for i in lines2]
fId2.close()

graph1 = nx.Graph(count=0)
graph2 = nx.Graph(count=0)

for i in lines1:
    vocabSet.add(i[0])
    vocabSet.add(i[1])
    graph1.add_node(i[0], {'count': 0})
    graph1.add_node(i[1], {'count': 0})
    if (float(i[2]) < 1.0):
        graph1.add_edge(i[0], i[1], {
            'weight': 0,
            'count': random.randint(15, 30),
            'time': 0
        })
for i in lines2:
    vocabSet.add(i[0])
    vocabSet.add(i[1])
    graph2.add_node(i[0], {'count': 0})
    graph2.add_node(i[1], {'count': 0})
    if (float(i[2]) < 1.0):
        graph2.add_edge(i[0], i[1], {
            'weight': 0,
            'count': random.randint(15, 30),
            'time': 0
        })
for word in vocabSet:
    if word not in graph1.nodes():
        graph1.add_node(word, {'count': 0})
for word in vocabSet:
    if word not in graph2.nodes():
        graph2.add_node(word, {'count': 0})

for i in graph1.edges(data=True):
    graph1.node[i[0]]['count'] += i[2]['count']
    graph1.node[i[1]]['count'] += i[2]['count']
    graph1.graph['count'] += 2 * i[2]['count']
for i in graph2.edges(data=True):
    graph2.node[i[0]]['count'] += i[2]['count']
    graph2.node[i[1]]['count'] += i[2]['count']
    graph2.graph['count'] += 2 * i[2]['count']
vocab = list(vocabSet)

learners.append(GraphLearner(graph1))
learners.append(GraphLearner(graph2))

def printDictionary(dists, vocab):
    newDists = [[1000.0 for i in range(len(vocab))] for j in range(len(vocab))]
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if (vocab[j] in dists[vocab[i]]):
                newDists[i][j] = float(dists[vocab[i]][vocab[j]])
                newDists[j][i] = float(dists[vocab[i]][vocab[j]])
    return '\n'.join(['\t'.join(map(str, i)) for i in newDists])


convs = []

# echo chamber
for iters in range(1, args.iterations + 1):
    print("In Learning iteration {:d}.".format(iters))

    conversation = set()
    topic = random.choice(vocab)
    conversation.add(topic)

    conversationLength = 0

    prevWord = topic

    while (conversationLength < 2 or (conversationLength < args.convlength and
                                      random.random() > args.stopprob)):
        conversationLength += 1
        nextWord = learners[conversationLength % 2].getNextWord(
            conversation, prevWord)
        conversation.add(nextWord)
        prevWord = nextWord
        print("    Talked about {:d} words. Latest is '{:s}'".format(
            conversationLength, nextWord))

    convs.append(conversation)

    if (iters % args.inferencesteps == 0):

        for conv in convs:
            up1 = learners[0].updateRepresentation(
                conv, iters, args.maxhistory, args.forgetpercent)
            up2 = learners[1].updateRepresentation(
                conv, iters, args.maxhistory, args.forgetpercent)

        convs.clear()
        print("    Updated representation for learners.")

        print("    Calculating distance matrices at: {:d}.".format(iters))

        dist1 = learners[0].getDiffusionDistanceMatrix(
                vocab, args.dimension)
        dist2 = learners[1].getDiffusionDistanceMatrix(
                vocab, args.dimension)
        print("    Got distance matrices.")

        f1 = open(
            args.outputfile +
            ('.learner1.{0:0=2d}.tsv'.format(iters // args.inferencesteps)),
            "w")
        f2 = open(
            args.outputfile +
            ('.learner2.{0:0=2d}.tsv'.format(iters // args.inferencesteps)),
            "w")

        f1.write(printDictionary(dist1, vocab))
        f2.write(printDictionary(dist2, vocab))

        print("    Printed distance matrices at: {:d}.".format(iters))

        f1.close()
        f2.close()
        print("    Wrote distance matrices at {:d}.".format(iters))
