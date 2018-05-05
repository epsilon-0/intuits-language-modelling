import numpy as np
from scipy.stats import wasserstein_distance
from numpy.linalg import norm
import sys
# takes as an argument the folder that it is going to comput the list for
import shutil

import os

def printOut(toFile, text):
    with open(toFile, 'w') as f:
        f.write(text)

def removeFolders(folder,N):
    cwd = os.getcwd()
    f = cwd+"/"+folder
    print(f)
    if(os.path.exists(f)):
        shutil.rmtree(f)
    '''
    for i in range(1,N+1):
        f = cwd+"/"+folder+str(i)
        print(folder)
        if(os.path.exists(f)):
            shutil.rmtree(f)
    '''

base = sys.argv[1]
print("the base is "+ base)
N = int(sys.argv[2])
time = int(sys.argv[3])
wassersteinDist = []
#bottleneckDist =[]

for i in range(1,time+1): # steps (epochs of conversations)
    wass = 0.
    #bott = 0.
    x = []
    for j in range(N): # number of people
        #l = 1
        #print(base+ str(i)+"/"+str(j))
        x.append(np.loadtxt(base+ str(i)+"/"+str(j), delimiter="\t"))
    count = 1.
    for j in range(N):
        for k in range(j+1, N):
            count+=1.
            m = min(len(x[j]) ,len(x[k]) ) 
            for z in range(m):
                
                #print(wasserstein_distance(x[j][z], x[k][z]))
                wass += wasserstein_distance(x[j][z], x[k][z])
                #bott += wasserstein_distance(x[j][z], x[k][z])
            #norm_ += norm(x[j] - x[k])
            wass= wass/m
            #print("are all the wass distances the same?")
            #print(wass)

            #print(wass)
    print("wass and count:")
    print(wass)
    print(count)
    print("")
    wassersteinDist.append(wass/(count))


printOut(base[:-1]+"_wasserstein.txt", str(wassersteinDist))
removeFolders(base, time)



