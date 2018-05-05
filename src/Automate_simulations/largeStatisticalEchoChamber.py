import argparse
from util import readStatisticalLearnerFile, writeStatisticalLearnerFile, statisticalLearnerProcess
from statistical_learner import StatisticalLearner
import numpy as np

# command line arguments
parser = argparse.ArgumentParser(
    description='Simulation for learners in echo chamber')
parser.add_argument(
    '--seed', metavar='r', type=int, default=42, help='seed for randomness')
parser.add_argument(
    '--readDirectory',
    metavar='r',
    type=str,
    required=True,
    help='directory from where to read the learner data')
parser.add_argument(
    '--writeDirectory',
    metavar='w',
    type=str,
    required=True,
    help='directory to where to write the learner data')
parser.add_argument(
    '--learnerNumbers',
    metavar='f',
    type=str,
    required=True,
    help='numbers of the learners to participate in the conversation')
parser.add_argument(
    '--convLength',
    metavar='c',
    type=int,
    default=20,
    help='maximum length of conversation')
parser.add_argument(
    '--numConversations',
    metavar='g',
    type=int,
    default=100,
    help='number of conversations')
args = parser.parse_args()

np.random.seed(args.seed)

learnerNumbers = args.learnerNumbers.strip().split("\t")
print(learnerNumbers)
readNames = [args.readDirectory + "/" + i for i in learnerNumbers]
writeNames = [args.writeDirectory + "/" + i for i in learnerNumbers]

learners = []

for i in readNames:
    inp = readStatisticalLearnerFile(i)
    learners.append(StatisticalLearner(inp))

statisticalLearnerProcess(
    learners, Ntalks=args.numConversations, convlength=args.convLength)

for i in range(len(writeNames)):
    writeStatisticalLearnerFile(writeNames[i], learners[i].getDistanceMatrix())
