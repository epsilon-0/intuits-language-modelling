import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import csv
import pickle 
# takes 4 arguments
# number of people, prob of connection, and where to read from, where to write to, for Abhi part
n = int(sys.argv[1])
minGroup = int(sys.argv[2])# 10  # people
maxGroup = int(sys.argv[3])
FileToRead = sys.argv[4] # read where the "All cliques are"
FileToWrite = sys.argv[5] # write a file for Abhinav, which is a list of disjoint covers
seed = 1
allCliques = []
def readcliques():
	with open(FileToRead) as f:
	    stringCliques = f.read().splitlines()
	for i in xrange(0 , len(stringCliques)):
		if not stringCliques[i]:
			continue
		print stringCliques[i]

		allCliques.append([])
		temp = stringCliques[i].split(",")
		#print temp
		#allCliques.append(list(map(temp, int)))

	print stringCliques
	print allCliques
with open(FileToRead, "rb") as new_filename:
	allCliques = pickle.load(new_filename)
print allCliques

def read(nameTxt):
	return np.loadtxt(nameTxt)


def disjointCovers(covers, n, sizeMin, sizeMax):
    # n total nodes
    # returns list of covers between the range sizeMin to SizeMax, which are disjoint

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
        if (disjoint == True):
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


print disjointCovers(allCliques, n, minGroup,maxGroup)

