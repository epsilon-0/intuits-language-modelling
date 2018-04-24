import argparse
from scipy.spatial import distance_matrix

parser = argparse.ArgumentParser(
    description='Simulation for learners in echo chamber')
parser.add_argument(
    '--infile',
    metavar='f',
    type=str,
    required=True,
    help='inputfile to read points from (gensim format output)')
parser.add_argument(
    '--outfile',
    metavar='f',
    type=str,
    required=True,
    help='outputfile to output the distance matrix to')
parser.add_argument(
    '--num-neighbours',
    metavar='f',
    type=int,
    default=50,
    help='number of neighbours to use in making the simplicial complex')
args = parser.parse_args()

cloud = []

lines = open(args.infile).readlines()

for i in lines:
    j = list(map(float, i.strip().split()[1:]))
    cloud.append(j)

ndistances = distance_matrix(cloud, cloud)
distances = [[100000.00 for i in range(len(cloud))] for j in range(len(cloud))]
for i in range(len(cloud)):
    dList = [(ndistances[i][j], j) for j in range(len(cloud))]
    dList.sort()
    for j in range(min(len(cloud), args.num_neighbours+1)):
        distances[i][dList[j][1]] = dList[j][0]
        distances[dList[j][1]][i] = dList[j][0]

with open(args.outfile, "w") as out:
    out.write("\n".join([" ".join(list(map(str, i))) for i in distances]))
    out.close()
