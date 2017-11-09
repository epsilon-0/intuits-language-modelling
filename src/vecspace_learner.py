import autograd.numpy as np
from autograd import grad
from autograd.misc import optimizers

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
        self.non_zero_counts = 0
        self.Z = self.calculateZ()
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
        return total_z / iters

    def getPairCount(self, w1, w2):
        # gets the value of X(w1, w2)
        return int(
            np.round(
                np.exp(
                    np.linalg.norm(self.vecspace[w1] + self.vecspace[w2])**2 +
                    self.C)))

    def getNextWord(self, context_indx):
        # gets a new word from the current context
        # hack work is done to ensure overflow is not happening
        # might still get underflow (?)
        context = self.vecspace[context_indx]
        dot_prods = np.array([np.dot(context, i) for i in self.vecspace])
        max_dot_prod = max(dot_prods)
        probability = np.array([np.exp(i - max_dot_prod) for i in dot_prods])
        probability /= np.sum(probability)
        next_word = np.random.choice(self.voc_size, p=probability)
        return next_word

    def setParams(self):
        # get the parameters needed for the optimization
        self.non_zero_counts = 0
        for i in range(self.voc_size):
            for j in range(i + 1, self.voc_size):
                pair_count = self.getPairCount(i, j)
                if (pair_count > 0):
                    # only do something if it is non trivially contributing
                    self.counts[i][j] = pair_count
                    self.counts[j][i] = self.counts[i][j]
                    self.non_zero_counts += 1
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
                if (self.counts[i][j] != 0):
                    answer += np.min([self.counts[i][j], 150]) * np.square(
                        (np.log(1 + self.counts[i][j]) -
                         np.square(np.linalg.norm(params[i] + params[j])) -
                         params[-1][0]))
        return answer

    def updateRepresentation(self, conversations, num_iters=20,
                             step_size=0.01):
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
            optim_grad, params, num_iters=num_iters, step_size=step_size)

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

    def getDistanceMatrix(self, metric="cosine"):
        # get the distance matrix for the vecspace
        # will return the distance matrix as
        # Cosine distance / or
        # L2 norm
        distances = dist.pdist(self.vecspace, metric=metric)
        dist_array = np.array([[0. for i in range(self.voc_size)]
                               for j in range(self.voc_size)])
        pos = 0
        for i in range(self.voc_size):
            for j in range(i + 1, self.voc_size):
                dist_array[i][j] = distances[pos]
                dist_array[j][i] = dist_array[i][j]
                pos += 1
        return dist_array
