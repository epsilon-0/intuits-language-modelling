import autograd.numpy as np
from autograd import grad, optimizers

import scipy.spatial.distance as dist


class VecspaceLearner():
    def __init__(self, vecspace, C):
        # need to learn a model where we can do
        # updates and test convergence
        self.vecspace = vecspace.copy()
        self.voc_size = len(self.vecspace)
        self.C = C
        self.counts = np.array([[0. for i in range(self.voc_size)]
                                for j in range(self.voc_size)])
        self.calculateZ()
        return

    def calculateZ(self, iters=30):
        # calculates value of Z
        total_z = 0.
        for i in range(iters):
            context = np.random.rand(*self.vecspace[0].shape)
            context *= np.sqrt(len(context))
            Zc = 0.
            for vec in self.vecspace:
                Zc += np.exp(np.dot(context, vec))
                total_z += Zc
        self.Z = total_z / iters

    def getPairCount(self, w1, w2):
        # gets the value of X(w1, w2)
        return np.floor(
            np.exp(np.norm(self.vecspace[w1] + self.vecspace[w2])**2 + self.C))

    def getNextWord(self, context):
        # gets a new word from the current context
        probability = np.array(
            [np.exp(np.dot(context, i)) for i in self.vecspace])
        probability /= np.sum(probability)
        next_word = np.choice(self.voc_size, p=probability)
        return next_word

    def setParams(self):
        # get the parameters needed for the optimization
        for i in range(self.voc_size):
            for j in range(i + 1, self.voc_size):
                pair_count = self.getPairCount(i, j)
                if (pair_count > 2):
                    # only do something if it is non trivially contributing
                    self.counts[i][j] = pair_count
                    self.counts[j][i] = self.counts[i][j]
        return

    def optimize(self, params, t):
        # Squared Norm optimization function
        # from the RAND-WALK paper
        # the first len(vecspace) arrays of params are the vectors
        # the last array has only the first element used which holds
        # the value of C
        answer = 0.
        for i in range(self.voc_size):
            for j in range(i + 1, self.voc_size):
                answer += np.min(self.counts[i][j], 150) * (
                    np.log(1 + self.counts[i][j]) -
                    np.square(np.norm(params[i] + params[j])) - params[-1][0])
        return answer

    def updateRepresentation(self, conversations):
        # update the vecspace using the given conversations

        # initialize the current parameters
        self.setParams()

        # update the parameters with given conversations
        self.aggregate(conversations)

        # calculate the gradient
        optim_grad = grad(self.optimize)

        # set the current parameters into a single array
        const_vector = np.array([[0. for i in range(len(self.vecspace[0]))]])
        const_vector[0][0] = self.C
        params = np.concatenate([self.vecspace, const_vector])

        # get the optimized parameters
        new_params = optimizers.adam(
            optim_grad, params, num_iters=500, step_size=0.01)

        # update the parameters
        self.C = new_params[-1][0]
        self.vecspace = new_params[:-1]
        return

    def aggregate(self, conversations):
        # return the total of all conversations
        # for updating the representations
        # will also update self.counts
        for conv in conversations:
            for i in range(len(conv)):
                for j in range(i + 1, len(conv)):
                    self.counts[conv[i]][conv[j]] += 1
                    self.counts[conv[j]][conv[i]] += 1
        return

    def getDistanceMatrix(self, metric='euclidean'):
        # get the distance matrix for the vecspace
        # will return the distance matrix as
        # Cosine distance / or
        # L2 norm
        distances = dist.pdist(self.vecspace, metric=metric)
        dist_array = np.array(
            [[distances[i * j - 1] for i in range(1, self.voc_size + 1)]
             for j in range(1, self.voc_size + 1)])
        return dist_array
