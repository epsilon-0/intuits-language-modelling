import argparse
from util import readVecspaceFile, writeVecspaceFile, learnerProcess
from vecspace_learner import VecspaceLearner
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
    default=30,
    help='number of conversations')
parser.add_argument(
    '--num_iters',
    metavar='nt',
    type=int,
    default=20,
    help='number of steps of optimization function')
parser.add_argument(
    '--step_size',
    metavar='st',
    type=float,
    default=0.01,
    help='step size for optimization adam function')
args = parser.parse_args()

np.random.seed(args.seed)

learnerNumbers = args.learnerNumbers.strip().split("\t")
print(learnerNumbers)
readNames = [args.readDirectory + "/" + i for i in learnerNumbers]
writeNames = [args.writeDirectory + "/" + i for i in learnerNumbers]

learners = []

for i in readNames:
    inp = readVecspaceFile(i)
    learners.append(VecspaceLearner(inp[0], inp[1]))

learnerProcess(
    learners,
    Ntalks=args.numConversations,
    convlength=args.convLength,
    num_iters=args.num_iters,
    step_size=args.step_size)

for i in range(len(writeNames)):
    writeVecspaceFile(writeNames[i], learners[i].vecspace, learners[i].C)
