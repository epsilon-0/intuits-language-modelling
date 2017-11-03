import argparse
from util import writeVecspaceFile, generateRandomCorpus, squaredNorm
import numpy as np

# command line arguments
parser = argparse.ArgumentParser(
    description='Simulation for learners in echo chamber')
parser.add_argument(
    '--seed', metavar='r', type=int, default=42, help='seed for randomness')
parser.add_argument(
    '--corpusSize',
    metavar='n',
    type=int,
    default=20,
    help='size of text corpus')
parser.add_argument(
    '--density',
    metavar='d',
    type=float,
    default=0.2,
    help='density of random graph')
parser.add_argument(
    '--dimension',
    metavar='D',
    type=int,
    default=50,
    help='dimension of word2vec embedding')
parser.add_argument(
    '--num_iters',
    metavar='t',
    type=int,
    default=100,
    help='number of iterations to run optimization function')
parser.add_argument(
    '--writeFile',
    metavar='f',
    type=str,
    required=True,
    help='file to write the embedding to')
args = parser.parse_args()

np.random.seed(args.seed)

corpus = generateRandomCorpus(args.corpusSize, density=args.density)

embedding = squaredNorm(
    corpus, dimension=args.dimension, num_iters=args.num_iters)

writeVecspaceFile(args.writeFile, embedding[0], embedding[1])
