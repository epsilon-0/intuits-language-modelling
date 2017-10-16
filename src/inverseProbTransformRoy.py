import numpy as np
import matplotlib.pyplot as plt 

# we are trying to simulat
def circularRadiusPolar(N, radius):
	#returns points from start to end following a distribution of a circle's circularRadius
	#fx = 2 pi r 
	#Fx = pi r^2
	#F^-1 x = sqrt(X/pi) # solved by setting Fx(F^-1(x)) = x

	#proof to prove that this is inverse transform:
	# Fx(X) = U 
	#ret = np.zeros((N,2))
	ret = np.random.random_sample((N,2))
	for i in xrange (N):
		
		ret[i][0] = 2*radius*np.sqrt( ret[i][0]/np.pi)
		ret[i][1] *= 2*np.pi
		temp = ret[i][0] 
		ret[i][0] *=np.cos(ret[i][1])
		ret[i][1] =  temp *np.sin(ret[i][1])
		
	return ret

def circularRadiusCartesian(N,radius ):
	ret = 2*radius*(np.random.random_sample((N,2))-.5)
	for i in xrange (N):
		while((ret[i][0]**2 + ret[i][1]**2)>radius**2):
			ret[i] = radius*np.random.random_sample((1,2))
	return ret

#print a
#print zip(*zip(a))

#print circularRadiusCartesian(100,1.)

fig1 = plt.figure(1)
fig1.add_subplot(2,1,1)
a = circularRadiusPolar(1000,1.)

#plt.plot(a[:,0], a[:,1])
plt.scatter(*zip(*a), s = 1)
#plt.plot(a)
#plt.scatter(a[:,0] * np.cos(a[:1]), a[:,0]* np.sin(a[:1]))
plt.subplot(2,1,2)
b = circularRadiusCartesian(1000,1.)

plt.scatter(*zip(*b), s = 1)

#plt.scatter(b[:,0], b[:,1])
fig1.suptitle("hello")
fig1.show()
plt.show()