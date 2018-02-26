from util import printDictionary, readVecspaceFile

import argparse

import autograd.numpy.random as rand

from vecspace_learner import VecspaceLearner

# command line arguments
parser = argparse.ArgumentParser(
    description='Simulation for learners in echo chamber')
parser.add_argument(
    '--seed', metavar='r', type=int, default=42, help='seed for randomness')
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
    help='input to learn initial random representations from for learner 2')
parser.add_argument(
    '--stopprob',
    metavar='p',
    type=float,
    default=0.05,
    help='probability of ending a conversation at any stage')

args = parser.parse_args()

rand.seed(args.seed)

inp1 = readVecspaceFile(args.inputfile1)
inp2 = readVecspaceFile(args.inputfile2)

def readDictionary(file_):
    d = {}
    with open(file_) as f:
        count = 0
        for line in f:
            d[count] = line[:-1]
            count+=1
    return d
#            (key, val) = line.split()
#            d[int(key)] = val

dictionary = readDictionary("words_alpha.txt")

def readScript(redditScriptFile):
    with open(file_) as f:
        count = 0
        for line in f:
            d[count] = line[:-1]
            count+=1
    return d











    










