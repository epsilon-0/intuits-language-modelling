library(TDA)
epsilon <-  10**-3
alpha <- 0.1 # learning rate
#printEvery <-  201
printEvery <- 100
iterations <- 1000
#print(paste("1_metricMatrix_" +printEvery+"_0" +".txt"))
args <- commandArgs(TRUE)
print("wasserstein distances_EntirePipeline")
#print("bottleneck distances_EntirePipeline")
for(i in seq(from=0, to=iterations, by=2)){


	#a <- sprintf("1_trial_5_RandWalk_%d_%d.txt",printEvery,i)
	#b <- sprintf("2_trial_5_RandWalk_%d_%d.txt",printEvery,i)

	a <- sprintf("%s1_%s_%d_%d.txt",args[1],args[2],printEvery,i)
	a <- sprintf("1invertEuc/1_invertEuc_RandWalk_RandWalk_100_228.txt")
	1_invertEuc_RandWalk_RandWalk_100_228.txt
	b <- sprintf("%s2_%s_%d_%d.txt",args[1],args[2], printEvery,i)
	#print(a)
	MyData1 <- read.csv(file=a , header=FALSE, sep = ",")
	MyData2 <- read.csv(file=b, header=FALSE, sep = ",")

	act1  <- ripsDiag(X = MyData1, maxdimension = 1,dist = "arbitrary", maxscale = 10) # dist= arbitrary, means matrix is a distance matrix (other option is euclidean, which is simply positions); creates nx3
	#max dimension of the homological features to be computed. (e.g. 0 for connected components, 1 for connected components and loops, 2 for connected components, loops, voids, etc.)
	act2  <- ripsDiag(X = MyData2, maxdimension = 1,dist = "arbitrary", maxscale = 10)

	#print("%d bottleneck distance ", i)
	#print(paste(a,"and ",b, " are :",bottleneck(act1[["diagram"]], act2[["diagram"]], dimension=1)))
	#print(i)
	#print( "wasserstein distance ")
	#print(paste(a,"and ",b, " are :",wasserstein(act1[["diagram"]], act2[["diagram"]], p=2, dimension=1)))
	print(wasserstein(act1[["diagram"]], act2[["diagram"]], p=2, dimension=1))
}

