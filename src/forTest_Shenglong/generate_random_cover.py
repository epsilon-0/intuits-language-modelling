import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description=
    'Random graph generator and clique finder for using in suimulation')
parser.add_argument(
    '--seed', metavar='r', type=int, default=42, help='seed for randomness')
parser.add_argument(
    '--n', metavar='n', type=int, default=100, help='size of the network')
parser.add_argument(
    '--mn',
    metavar='min',
    type=int,
    default=2,
    help='min size of clique to use')
parser.add_argument(
    '--mx',
    metavar='max',
    type=int,
    default=5,
    help='max size of clique to use')
parser.add_argument(
    '--rFile',
    metavar='rf',
    required=True,
    type=str,
    help='File to read the cliques from')
parser.add_argument(
    '--wFile',
    metavar='wf',
    required=True,
    type=str,
    help='File to write the cover to')
args = parser.parse_args()

np.random.seed(args.seed)

fId = open(args.rFile, "r")
lines = fId.readlines()
fId.close()
allCliques = [list(map(int, i.strip().split("\t"))) for i in lines]


def disjointCovers(covers, n, sizeMin, sizeMax):
    # n total nodes
    # returns list of covers between the range sizeMin to sizeMax
    # which are disjoint

    ret = []
    used = np.zeros(n)
    for i in range(len(covers)):
        ind = np.random.randint(0, len(covers))
        size = len(covers[ind])
        if (size < sizeMin):
            continue
        disjoint = True
        for a in range(len(covers[ind])):
            if (used[covers[ind][a]] == 1):
                disjoint = False
        if disjoint:
            if (sizeMax < len(covers[ind])):
                asd = []
                count = 0
                while (count < sizeMax):
                    ind2 = np.random.randint(0, len(covers[ind]))
                    if (used[covers[ind][ind2]] == 0):
                        asd.append(covers[ind][ind2])
                        used[covers[ind][ind2]] = 1
                        count += 1
                ret.append(asd)
            else:
                for b in range(len(covers[ind])):
                    used[covers[ind][b]] = 1
                ret.append(covers[ind])
    return ret


covers = disjointCovers(allCliques, args.n, args.mn, args.mx)

print("\n".join(["\t".join(map(str, i)) for i in covers]))
