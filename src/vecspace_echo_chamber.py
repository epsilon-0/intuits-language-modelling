from util import printDictionary

import argparse

import numpy.random as rand

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

learners = []

vocab = range(1, 200)

convs = []
# echo chamber
for iters in range(1, args.iterations + 1):
    print("In Learning iteration {:d}.".format(iters))

    conversation = set()
    topic = rand.choice(vocab)
    conversation.add(topic)

    conversationLength = 0

    prevWord = topic

    while (conversationLength < 2 or (conversationLength < args.convlength and
                                      rand.random() > args.stopprob)):
        conversationLength += 1
        nextWord = learners[conversationLength % 2].getNextWord(
            conversation, prevWord)
        conversation.add(nextWord)
        prevWord = nextWord
        print("    Talked about {:d} words. Latest is '{:s}'".format(
            conversationLength, nextWord))

    convs.append(conversation)

    if (iters % args.inferencesteps == 0):

        up1 = learners[0].updateRepresentation(convs)
        up2 = learners[1].updateRepresentation(convs)

        convs.clear()
        print("    Updated representation for learners.")

        print("    Calculating distance matrices at: {:d}.".format(iters))

        dist1 = learners[0].getDistanceMatrix(vocab)
        dist2 = learners[1].getDistanceMatrix(vocab)
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
