import argparse
from util import writeStatisticalLearnerFile, generateRandomCorpus
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
    help='density of corpus')
parser.add_argument(
    '--writeFile',
    metavar='f',
    type=str,
    required=True,
    help='file to write the embedding to')
parser.add_argument(
    '--history',
    metavar='h',
    type=float,
    default=5000,
    help='amount of history that each learner keeps')
args = parser.parse_args()

np.random.seed(args.seed)

corpus = generateRandomCorpus(args.corpusSize, density=args.density)
corpus *= args.history / sum(sum(corpus))

writeStatisticalLearnerFile(args.writeFile, corpus)
