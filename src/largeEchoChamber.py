import sys
from util import readVecspaceFile, writeVecspaceFile, learnerProcess
from vecspace_learner import VecspaceLearner

# first command line argument is number of files
# the first half of the remainder is input file names
# the remaining is the output file names

numFiles = int(sys.argv[1])

fileNames = sys.argv[1:numFiles + 1]

learners = []

for i in fileNames:
    inp = readVecspaceFile(i)
    learners.append(VecspaceLearner(inp[0], inp[1]))

learnerProcess(learners, Ntalks=30, convlength=20)

for i in range(numFiles):
    writeVecspaceFile(sys.argv[i + 1 + numFiles], learners[i].vecspace,
                      learners[i].C)
