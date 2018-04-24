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
args = parser.parse_args()

cloud = []

lines = open(args.infile).readlines()

for i in lines:
    j = list(map(float, i.strip().split()[1:]))
    cloud.append(j)

distances = distance_matrix(cloud, cloud)

with open(args.outfile, "w") as out:
    out.write("\n".join([" ".join(list(map(str, i))) for i in distances]))
    out.close()
