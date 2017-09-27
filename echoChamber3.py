import numpy as np
import scipy
import scipy.spatial.distance as d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import sys 

epsilon = 10**-2
alpha = 0.1 # learning rate
iterations = 3
words = 200
dimensions = 30
rangeVecs = 10.
variance = (2*rangeVecs)/12.
stdDev = np.sqrt(variance) # we know that this will be the standard deviation of the gaussian as well, because summing is 
#linear operation, and the gaussian is simply the sum of n random variables
printEvery = 100 # print every 100 word pairs
conversationLength = 201 # how many words are said on a topic
NumConversations = 500
folderName = "today/"
cooccurrenceMat = np.zeros((words,words)) +1
totalWordsSaid = 0
wantPrint = True
# need an accelearation of learning rate, based on how frequently the coocur
#need very specific definition of co occurence, in order for everything not to converge to 1 point

def generateNVectors(vecs,dimension,low, high):
	tot = abs(high -low)
	mid = (high - low)/2
	return (np.random.random((vecs,dimension))*tot - mid)

def cosSimilarity(vec1, vec2):
	return (1- d.cosine(vec1,vec2))

def norm(vec): #l2 norm
	summer = 0.
	for i in xrange(len(vec)):
		summer+= (vec[i]*vec[i])
	return np.sqrt(summer)
def printNorms(Mat):
	total = 0.
	norms = np.zeros(Mat.shape[0])
	for i in xrange(0,Mat.shape[0]):
		norms[i] = norm(Mat[i])
		total+= norms[i]
	#print norms
	a= (total)/(Mat.shape[0])
	print "average norm :" ,a
	return a
def IsotropyMetric1(percent, distMatrix, metric): # distMatrix is a 2d matrix that is n x n and tells you the distance between vector i and j
	# find average distance from 1 vector to the nearest 10% of the vectors
	isoVec = np.zeros(distMatrix.shape[0])
	dis = np.zeros(distMatrix.shape[0])
	minCount = 5
	#print "M[0]" , M.shape[0]*percent
	for i in xrange(distMatrix.shape[0]): # for each vector
		#dis[i] =metric(distMatrix[])
		#beep boop baaaaap
		dis = distMatrix[distMatrix[:,i].argsort(kind='mergesort')]
		 # check for the nearest ~10 vectors #
		a = max(minCount,(int(np.ceil(distMatrix.shape[0]*percent))))
		#print len(distMatrix[i]), "while the mat size is ", distMatrix.shape[0]
		#print sort[0:a]
		isoVec[i] = np.average(dis[1:a+1])
	return isoVec
def IsotropyMetric2(mat): # distMatrix is a 2d matrix that is n x n and tells you the distance between vector i and j
	# find average distance from 1 vector to the nearest 10% of the vectors
	summer = np.zeros(mat.shape[1]) 
	ret = 0
	for i in xrange(mat.shape[0]): # for each vector
		summer +=mat[i] # by central limit theorem, this converges to a gaussian dist. centered at 0
	#for i in xrange (len(summer)):
	#	ret += summer[i]/stdDev
	
	#print summer/(len(summer)*np.sqrt(mat.shape[0])), stdDev

	#ret = norm(summer)/(len(summer) * stdDev)
	#summer = summer/stdDev # normed by the std
	summer = summer/(mat.shape[0])
	return summer
	#return sum(summer)/mat.shape[1] # return the sum of the distance to the origin in each direction
	#return ret
def getIsotropy1(M, metric): # find the isotropy for an aribtrary metric
	return np.average(IsotropyMetric1(.1, distanceArbitraryMatrix(M, metric) ,metric) )

def getIsotropy2(M, metric): # find the isotropy for an aribtrary metric
	return np.average(IsotropyMetric2(M))/stdDev

def distanceArbitraryMatrix(M, metric): # 4 kinds of metrics : euclidean, inverted euclidean, dot, cosine similarity
	S = np.zeros((M.shape[0],M.shape[0]))

	for i in xrange(M.shape[0]):
		for j in xrange(i+1):
			S[i][j] = metric(M[i],M[j])
			S[j][i] = S[i][j]
	return S

def printResults(M, filename, metric):
	with open(filename, "w") as f:
			writer = csv.writer(f)
			writer.writerows(distanceArbitraryMatrix(M, metric))
def printIsotropy(word, filename):
	with open(filename, "a") as f:
			f.write(str(word)+"\n")
def moveMatOnceNew(metric, M, i ,j): # this is what we do if two words are said in the same context move them to increase their similarity
	global epsilon
	global alpha
	global totalWordsSaid
	global cooccurrenceMat
	# takes word i and word j, and moves them clser to one another
	i = int(i)
	j = int(j)
	totalWordsSaid +=2
	cooccurrenceMat[i][j] += 1
	cooccurrenceMat[j][i] += 1
	cooccurrenceMat[i][i] += 1
	cooccurrenceMat[j][j] += 1
	strengthI = 1.*cooccurrenceMat[i][j]/(1.*cooccurrenceMat[i][i]) # more common word, means it moves less
	strengthJ = 1.*cooccurrenceMat[i][j]/(1.*cooccurrenceMat[j][j])
	if (strengthJ >1. or strengthI >1.):
		print "THIS SHOULD NOT HAPPEN"
	prevMet = metric(M[i],M[j])
	#randVec = (np.random.random(M.shape[1]) *2 *alpha) - alpha
	change = (M[i] - M[j])*alpha
	prevMet = metric(M[i],M[j])
	before = norm(M[i])
	before2 = norm(M[j])
	M[i] = M[i] - (change*strengthI)
	M[j] = M[j] + (change*strengthJ)
	if(norm(M[i]) > before and norm(M[j])> before2):
		print "the norm after for both, is larger than the one before \n \n \n "
	#print (metric(M[i],M[j])- prevMet), "this should be positive"
	return
def moveMatOnce(metric, M, i ,j): # this is what we do if two words are said in the same context move them to increase their similarity
	global epsilon
	global alpha
	global totalWordsSaid
	global cooccurrenceMat
	# takes word i and word j, and moves them clser to one another
	i = int(i)
	j = int(j)
	totalWordsSaid +=2
	cooccurrenceMat[i][j] += 1
	cooccurrenceMat[j][i] += 1
	cooccurrenceMat[i][i] += 1
	cooccurrenceMat[j][j] += 1
	strengthI = 1.*cooccurrenceMat[i][j]/(1.*cooccurrenceMat[i][i]) # more common word, means it moves less
	strengthJ = 1.*cooccurrenceMat[i][j]/(1.*cooccurrenceMat[j][j])

	prevMet = metric(M[i],M[j])
	randVec = (np.random.random(M.shape[1]) *2 *alpha) - alpha
	M[i] = M[i] - (randVec*strengthI)
	M[j] = M[j] + (randVec*strengthJ)
	if(metric(M[i],M[j]) > prevMet): # trying to get the vectors to move away from one another # if metric bigger, then more similar
		return
	else:
		M[i] = M[i] + 2*(randVec*strengthI)
		M[j] = M[j] - 2*(randVec*strengthJ)
	return
	#note d.cosine = 1- cos(anglebetween)

def binarySearchSolution(function, epsilon,lower, upper):
	# we are going to find a solution that sets the function abritrarily close to 0 (as a function of epsilon)
	low = lower
	high = upper
	mid = (high + low)/2.
	tempSol = 100.
	for i in xrange(0,64):
		tempSol = function(mid)
	#	print tempSol
		if(abs(tempSol) < epsilon):
			return mid
		if(tempSol >0.):
			high = mid +epsilon
		if(tempSol <=0.):
			low = mid-epsilon
		mid = (high + low)/2.
	return mid
def solveforC(v1, v2,e1,e2): # v1 > v2 by construction , e1 and e2 are the perpindicular vectors constructed
	# solve for the c of the ellipse. Ellipse is structured such that center of ellipse is on the vertical axis, and origin is at a foci
	c = binarySearchSolution( lambda x: (norm(v1)-norm(v2) - norm(v2 - (2*e2 *x))), 10**-14, -norm(v1), norm(v1))
	return c
def solveforR(v1,C): # R is a (major semi axis)
	return norm(v1)-C # c= solveForC(v1,v2,e1,e2) # x = r cos @  y = s sin @
def solveforS(C,R): # s is b (minor semi axis)
	return np.sqrt((C*C) + (R*R))
def solveforNewAngle(v1,v2,e1,e2,angle12, S):
	qwe = norm(v2)*np.sin(angle12)/S
	#print qwe
	if(abs(qwe) >= 1):
		print "problem child", qwe
		print v1, v2
		print e1, e2
		print "angle", angle12
		print "s", S
		return -101
	return np.arcsin(qwe) 

def moveMatOnceCosineWay(metric,M, i,j):# let's see if this way of moving things, for cosine similarity is better.
	#in the plane defined by vec1 and vec2, calculate the polar coordinates to
	#print "i,j", i,j
	global totalWordsSaid
	global cooccurrenceMat
	totalWordsSaid +=2
	cooccurrenceMat[i][j] += 1
	cooccurrenceMat[j][i] += 1
	cooccurrenceMat[i][i] += 1
	cooccurrenceMat[j][j] += 1
	strengthI = 1.*cooccurrenceMat[i][j]/(1.*cooccurrenceMat[i][i]) # more common word, means it moves less
	strengthJ = 1.*cooccurrenceMat[i][j]/(1.*cooccurrenceMat[j][j])
	
	if(norm(M[i]) < norm(M[j])): # we want i to be the bigger one
		temp = i
		i =j
		j = temp
	vec1 = M[i] # take vec 1 to be angle for @ = 0. so y = norm(vec1) *cos(0)
	a1 = norm(vec1)  # for y =a cos @ and x = b sin @
	e1 = M[i]/a1 # basis vector1
	vec2 = M[j] # take which ever way vec 2 is, clockwise or counter clockwise to be positive theta
	a2 = norm(vec2)
	proj= np.inner(vec1,vec2)/(a1*a1) * vec1 # projection of vector 2 onto vec1
	gramSchmitt2 = vec2-proj # this is the second basis vector
	g2 = norm(gramSchmitt2)
	e2 = gramSchmitt2/g2
	angleBetween12 = np.arccos(cosSimilarity(vec1,vec2))

	c =solveforC(vec1, vec2, e1,e2)
	r =solveforR(vec1, c)
	s =solveforS(c,r)
	angleBetween = solveforNewAngle(vec1,vec2,e1,e2 ,angleBetween12,s)
	if(angleBetween == -101):
		print "Skipping 1, because it is too close"
		return
	dTheta = alpha *angleBetween # this is the amount that both vectors move towards each other along their spheres # 10 percent of the angle
	#diff1 = M[i] -   (a1*np.cos(0) *e1) +(b1 *np.sin(0) * e2)
	#diff2 = M[j] - 	(a1*np.cos(angleBetween) *e1) +(b1 *np.sin(angleBetween) * e2) # just checking that the gram schmidt process is good
	#print "angle is ", dTheta
	#print "diffs!" , norm(diff1), norm(diff2)
	prevMet = metric(M[i],M[j])
	s= M[i]
	r = M[j]
	#print "old metric", metric(M[i],M[j])
	M[i] = c*e1+ (r*np.cos(dTheta *strengthI) *e1) +(s*np.sin(dTheta*strengthI) * e2)
	M[j] =  c*e1+ (r*np.cos(angleBetween - (dTheta*strengthJ)) *e1) +(s *np.sin(angleBetween - (dTheta*strengthJ)) * e2) # moves both vectors along their ellipse, towards one another in a way that increases their cosine similarity 
	#print "new metric", metric(M[i],M[j])
	if((metric(M[i],M[j])- prevMet)<0):
		print "this should be positive _WRONG"
	return


def moveMatOnceCosineWayOld(metric,M, i,j):# let's see if this way of moving things, for cosine similarity is better.
	#in the plane defined by vec1 and vec2, calculate the polar coordinates to
	#print "i,j", i,j
	global totalWordsSaid
	global cooccurrenceMat
	totalWordsSaid +=2
	cooccurrenceMat[i][j] += 1
	cooccurrenceMat[j][i] += 1
	cooccurrenceMat[i][i] += 1
	cooccurrenceMat[j][j] += 1
	strengthI = 1.*cooccurrenceMat[i][j]/(1.*cooccurrenceMat[i][i]) # more common word, means it moves less
	strengthJ = 1.*cooccurrenceMat[i][j]/(1.*cooccurrenceMat[j][j])
	
	if(norm(M[i]) < norm(M[j])): # we want i to be the bigger one
		temp = i
		i =j
		j = temp
	vec1 = M[i] # take vec 1 to be angle for @ = 0. so y = norm(vec1) *cos(0)
	a1 = norm(vec1)  # for y =a cos @ and x = b sin @
	e1 = M[i]/a1 # basis vector1
	vec2 = M[j] # take which ever way vec 2 is, clockwise or counter clockwise to be positive theta
	a2 = norm(vec2)
	#print "a1 and a2", a1, a2
	#print "A1 a2", a1, a2
	proj= np.inner(vec1,vec2)/(a1*a1) * vec1 # projection of vector 2 onto vec1
	gramSchmitt2 = vec2-proj # this is the second basis vector
	g2 = norm(gramSchmitt2)
	e2 = gramSchmitt2/g2
	#e2 = M[j]/a2 #basis vector2
	#print "cos sim" , i,j,1-d.cosine(vec1,vec2)
	angleBetween = np.arccos(cosSimilarity(vec1,vec2))

	#now find b sin @ in this plane, by projecting  vec2 onto e2 direction, and 
	x = g2 # this almost gets you b for x = b sin @
	b1 = x/np.sin(angleBetween)

	dTheta = alpha *angleBetween # this is the amount that both vectors move towards each other along their spheres # 10 percent of the angle
	diff1 = M[i] -   (a1*np.cos(0) *e1) +(b1 *np.sin(0) * e2)
	diff2 = M[j] - 	(a1*np.cos(angleBetween) *e1) +(b1 *np.sin(angleBetween) * e2) # just checking that the gram schmidt process is good
	#print "angle is ", dTheta
	#print "diffs!" , norm(diff1), norm(diff2)
	prevMet = metric(M[i],M[j])
	s= M[i]
	r = M[j]
	#print "old metric", metric(M[i],M[j])
	M[i] = (a1*np.cos(dTheta *strengthI) *e1) +(b1 *np.sin(dTheta*strengthI) * e2)
	M[j] = (a1*np.cos(angleBetween - (dTheta*strengthJ)) *e1) +(b1 *np.sin(angleBetween - (dTheta*strengthJ)) * e2) # moves both vectors along their ellipse, towards one another in a way that increases their cosine similarity 
	#print "new metric", metric(M[i],M[j])
	if((metric(M[i],M[j])- prevMet)<0):
		print "this should be positive _WRONG"
	return
	#print "we are in the cosine way"
	#angle between is the angle between vec 1 and vec2

### rand walk approach
def findOneClosestVectors(Mat, i, j, n, metric): # look into pyfly
	# returns one of the n closest vectors to i, in Mat, which is NOT i or j
	dis = np.zeros((Mat.shape[0],2))

	for a in xrange(0,Mat.shape[0]):
		if(a==j or a == i):
			dis[a] = 0

		dis[a] = (metric(Mat[i], Mat[a]),int(a)) 
	dis = dis[dis[:,0].argsort(kind='mergesort')] # sort by the 1st element (distance),  but we will return the actual index tho
	#print int(dis[np.random.randint(1,n)][1])
	return int(dis[np.random.randint(2,n+1)][1])

def converseOnceAbout(M, N,i,j, metric): # converse about specific letters i and j (randomly choose a word that is similar to the previous words)
	if(metric == cosSimilarity):
		moveMatOnceCosineWay(metric, N, i, j)
		moveMatOnceCosineWay(metric, M, i,j)
	else:
		moveMatOnce(metric, N, i, j)
		moveMatOnce(metric, M, i,j)

def converseAlot(M,N,n, metric): # completely random conversations
	for a in xrange(0,n):
		i = np.random.randint(0,M.shape[0]-1)
		j = np.random.randint(0,M.shape[0]-1)
		while (j== i):
			j = np.random.randint(0,M.shape[0]-1)
		#print "converse about args", i,j
		converseOnceAbout(M,N,i,j, metric)

def converseAlotRandWalk(M,N,n,metric): # M is learner 1 # N learner 2; n is number of conversatoins; metric is the metrix used
	global conversationLength # these conversations are all somewhat strung together
	i = np.random.randint(0,M.shape[0]-1)
	j = np.random.randint(0,M.shape[0]-1)
	while (j== i):
		j = np.random.randint(0,M.shape[0]-1)
	converseOnceAbout(M,N,i,j, metric) # first conversation
	count = 0
	for a in xrange(0,n):
		count = count+1
		if(count == conversationLength): # random subject, every x # of word pairs
			count =0
			i = np.random.randint(0,M.shape[0]-1)
			j = np.random.randint(0,M.shape[0]-1)
			while (j== i):
				j = np.random.randint(0,M.shape[0]-1)
		temp = i
		i = findOneClosestVectors(M,i,j,20,metric) # find a new word that is similar to word i, in the 1st speakers mind
		j = findOneClosestVectors(N,temp,i,20,metric) #find a new word that is similar to word i, in the 2nd speakers mind
		while (j == i):
			j = findOneClosestVectors(N,i,i,20,metric)
		#print "converse a lot rand walk is ", i , j
		converseOnceAbout(M,N,i,j, metric)

def negEuc(a,b): # a kind of metric
	#euclidean metrics get smaller when things get closer, but we want them to get bigger, - infinity ->0
	ret = d.euclidean(a,b)
	return -ret


def invertEuc(a,b): # a kind of metric
	#euclidean metrics get smaller when things get closer, but we want them to get bigger, so just scaled them inversely
	global epsilon
	ret = d.euclidean(a,b)
	if(ret <epsilon):
		return (1./epsilon)
	else:
		return (1./ret)
def PMI(i,j):
	global totalWordsSaid
	global cooccurrenceMat
	return np.log((totalWordsSaid/2)* cooccurrenceMat[i][j]/(cooccurrenceMat[i][i] * cooccurrenceMat[j][j])) # =log(p(v1,v2)/p(v1)p(v2))
def calcZ(Mat):
	global NumConversations
	contexts =generateNVectors(NumConversations,Mat.shape[1], -rangeVecs, rangeVecs) 
	Zc = np.zeros(NumConversations) # 1 Zc for each conversation
	for i in xrange(0, len(Zc)):
		for j in xrange (0, Mat.shape[0]): # sum over all words
			x =1
			#Zc[i] += np.exp(PMI(context[i], ))
	return 1

def calcZ1(Mat):
	global cooccurrenceMat
	global totalWordsSaid
	
	ret = np.zeros((Mat.shape[0], Mat.shape[0]))
	
	for i in xrange(0,Mat.shape[0]):
		for j in xrange(0,Mat.shape[0]):
			ret[i][j] = (norm(Mat[i] + Mat[j])**2/(2*Mat.shape[1]) )- (np.log(cooccurrenceMat[i][j]/(totalWordsSaid/2)))
	out = np.average(ret)
	return out
	
def calcZ2(Mat): 
	#calculate Z the inverse way, given formula 2.4
	global cooccurrenceMat
	global totalWordsSaid
	logZ = np.zeros(Mat.shape[0])
	maxe = 0
	mine = 10000000
	for i in xrange (0, Mat.shape[0]):
		if(cooccurrenceMat[i][i] == 0):
			continue
		logZ[i] = (norm(Mat[i])**2/(2*Mat.shape[1])) - (np.log(cooccurrenceMat[i][i]/(totalWordsSaid/2)))
		maxe = max(maxe, logZ[i])
		mine = min(mine, logZ[i])
	#print "max and min Zs ", maxe, mine , "| "
	return np.average(logZ)
	#return (sum(logZ)/len(logZ))

def theorem22Right(Mat,logZ):
	global cooccurrenceMat
	ret = np.zeros((Mat.shape[0],Mat.shape[0]))
	for i in xrange(0,Mat.shape[0]):
		for j in xrange(0,Mat.shape[0]):
			ret[i][j] = (norm(Mat[i] + Mat[j])/(2*Mat.shape[1]) )- (2*logZ)
	return ret
def theorem22Left(Mat):
	global cooccurrenceMat
	global totalWordsSaid
	ret = np.zeros((Mat.shape[0],Mat.shape[0]))
	for i in xrange(0,Mat.shape[0]):
		for j in xrange(0,Mat.shape[0]):
			if(cooccurrenceMat[i][j] == 0):
				ret[i][j] = 0
				continue
			ret[i][j] = np.log(cooccurrenceMat[i][j]/(totalWordsSaid/2))
	return ret

def entirePipelineRandWalk(metric, name): # entire pipeline, but sentences are generally strucutured
	global printEvery # every _PrintEvery_ print the results
	global iterations # how many times do you want to print everything
	global NumConversations
	global conversationLength
	global wantPrint
	#conversations = 10000
	#printEvery = conversationLength+1
	ZsEvery = 10
	Mat= generateNVectors(words,dimensions,-1.*rangeVecs,rangeVecs) #words = 500,dimensions = 40, rangeVecs ={-10,10}
	Mat2= generateNVectors(words,dimensions,-1.*rangeVecs,rangeVecs)
	if(wantPrint):
		printResults(Mat, "1"+metric.__name__+"/1_" +name+"_RandWalk_"+str(printEvery)+"_"+str(0)+ ".txt", metric)
		printResults(Mat2, "1"+metric.__name__+"/2_"+name+"_RandWalk_"+str(printEvery)+"_"+str(0)+  ".txt", metric)
	
	if(wantPrint):
		printIsotropy(printNorms(Mat), "1"+metric.__name__+"/"+name+"averageNorms.txt")
		printIsotropy("New Pipeline RANDOM", "1"+metric.__name__+"/"+name+"RandWalk_kNN_isotropy.txt")
		printIsotropy("New Pipeline RANDOM", "1"+metric.__name__+"/"+name+"RandWalk_CLT_isotropy.txt")

		printIsotropy("Zresults Z value 1 way - Z value another (closer to 0 is best)", "1"+metric.__name__+"/"+name+"RandWalk_Zresults.txt")
	for i in xrange(1, NumConversations+1):
		print i
		if(wantPrint):
			if(i%ZsEvery == 0):
				s = (theorem22Left(Mat)- theorem22Right(Mat,calcZ2(Mat)))
				print i, "Z shit: should be close to 0 \n", s
				print "\n average", np.average(s)
				print "should be rougly: |" ,calcZ2(Mat) ,"|"
				print

		converseAlotRandWalk(Mat, Mat2, printEvery, metric)
		printNorms(Mat)
		if(wantPrint):
			a= getIsotropy1(Mat,metric) #knn
			b= getIsotropy2(Mat,metric) #clt

			print "istropy1 and 2", a,"\t", b
	#		printIsotropy(a,"1"+metric.__name__+"/"+name+"isotropy.txt")
			printIsotropy(b, "1"+metric.__name__+"/"+name+"RandWalk_CLT_isotropy.txt")
			printIsotropy(a, "1"+metric.__name__+"/"+name+"RandWalk_kNN_isotropy.txt")
			printIsotropy(printNorms(Mat), "1"+metric.__name__+"/"+name+"RandWalk_averageNorms.txt")
			printIsotropy(calcZ1(Mat)-calcZ2(Mat), "1"+metric.__name__+"/"+name+"RandWalk_Zresults.txt")
			printResults(Mat, "1"+metric.__name__+"/1_" +name+"_RandWalk_"+str(printEvery)+"_"+str(i)+ ".txt", metric)
			printResults(Mat2, "1"+metric.__name__+"/2_"+name+"_RandWalk_"+str(printEvery)+"_"+str(i)+  ".txt", metric)

def entirePipeline(metric, name): # entire pipeline, but sentences are generally strucutured
	global printEvery # every _PrintEvery_ print the results
	global iterations # how many times do you want to print everything
	#conversations = 10000
	global NumConversations
	global wantPrint
	#printEvery = 100
	ZsEvery = 10
	Mat= generateNVectors(words,dimensions,-1.*rangeVecs,rangeVecs)
	Mat2= generateNVectors(words,dimensions,-1. *rangeVecs,rangeVecs)
	if(wantPrint):
		printResults(Mat, "1"+metric.__name__+"/1_" +name+"_"+str(printEvery)+"_"+str(0)+ ".txt", metric)
		printResults(Mat2, "1"+metric.__name__+"/2_"+name+"_"+str(printEvery)+"_"+str(0)+  ".txt", metric)
		
		printIsotropy(printNorms(Mat), "1"+metric.__name__+"/"+name+"averageNorms.txt")

		printIsotropy("New Pipeline RANDOM", "1"+metric.__name__+"/"+name+"Random_kNN_isotropy.txt")
		printIsotropy("New Pipeline RANDOM", "1"+metric.__name__+"/"+name+"Random_CLT_isotropy.txt")	
		printIsotropy("Zresults Z value 1 way - Z value another (closer to 0 is best)", "1"+metric.__name__+"/"+name+"Random_Zresults.txt")
	for i in xrange(1, NumConversations+1):
		print i
		if(wantPrint):
			if(i%ZsEvery == 0):
				s = (theorem22Left(Mat)- theorem22Right(Mat,calcZ2(Mat)))
				print i, "Z shit:\n", s
				print "\n average", np.average(s)
				print "should be rougly: |" ,calcZ2(Mat) ,"|"
				print
		converseAlot(Mat, Mat2, printEvery, metric)
		b= getIsotropy2(Mat,metric) #CLT
		a= getIsotropy1(Mat,metric) #kNN
		
		print "Isotropy1", getIsotropy1(Mat,metric), "Isotropy2" , getIsotropy2(Mat, metric)
	 
		if(wantPrint):
			print "istropy1 and 2", a,"\t", b
	#		printIsotropy(a,"1"+metric.__name__+"/"+name+"isotropy.txt")
			printIsotropy(b, "1"+metric.__name__+"/"+name+"Random_CLT_isotropy.txt")
			printIsotropy(a, "1"+metric.__name__+"/"+name+"Random_kNN_isotropy.txt")
			printIsotropy(printNorms(Mat), "1"+metric.__name__+"/"+name+"Random_averageNorms.txt")
			printIsotropy(calcZ1(Mat)-calcZ2(Mat), "1"+metric.__name__+"/"+name+"Random_Zresults.txt")
			printResults(Mat, "1"+metric.__name__+"/1_" +name+"_"+str(printEvery)+"_"+str(i)+ ".txt", metric)
			printResults(Mat2, "1"+metric.__name__+"/2_"+name+"_"+str(printEvery)+"_"+str(i)+  ".txt", metric)




entirePipelineRandWalk(invertEuc, "invertEuc_RandWalk")
entirePipeline(invertEuc, "invertEuc_RandWalk")

#entirePipelineRandWalk(negEuc, "negEuc_RandWalk")
#entirePipeline(negEuc, "negEuc_Random")

#entirePipelineRandWalk(cosSimilarity, "Cosine_RandWalk")#
#entirePipeline(cosSimilarity, "Cosine_Random")

#entirePipeline(np.dot, "Dot_Random")
#entirePipelineRandWalk(np.dot, "Dot_RandWalk")



##Start with some representaiton - babble for 100 steps; observe distance, repeate 
##(over time, distance between conversation and previous conversatino goes goes to 0) (this plot will be the statistics)
##Null hypothesis - Random words (this is the code that is below this)
##
##entire pipeline rand walk is doing a random walk within an area (i.e. choose one of the kNN words to converse with)
##Also have a situation where you converse, using RAND walk (choose one of the kNN words)
##Start with some representaiton - babble for 100 steps; observe distance, repeate 
##(over time, distance between conversation and previous conversatino goes goes to 0) (this plot will be the statistics)
##Null hypothesis - Random words
##
##
##Add in the functinoality of doing a random walk within an area (i.e. choose one of the kNN words to converse with)
##Also have a situation where you converse, using RAND walk (choose one of the kNN words)
##
