import autograd.numpy as np


class StatisticalLearner():
    def __init__(self, prob_matrix):
        # need to learn a model where we can do
        # updates and test convergence
        # 1) voc_size is the number of word in vocabulary
        # 2) prob_matrix is the matrix of co-occurrences
        self.voc_size = len(prob_matrix)
        self.prob_matrix = prob_matrix.copy()
        return

    # only needed if we are going to make pro_matrix into a list
    #def getPairProb(self, w1, w2):
    #    # gets the value of P(w1, w2)
    #    # gets it from the prob_matrix
    #    # w1, w2 are 1 based indexed
    #    v1 = w1-1
    #    v2 = w2-1
    #    if v1 > v2:
    #        v1, v2 = v2, v1
    #    indx = (self.voc_size) * v1 + v2 - ((v1*(v1+1))//2)
    #    return self.prob_matrix[indx]

    def getNextWord(self, context):
        # gets a new word from the current context
        probability = np.array([self.prob_matrix[context][i] for i in range(self.voc_size)])
        tot = np.sum(probability)
        if tot < 0.000000001:
            return context
        probability /= tot
        next_word = np.random.choice(self.voc_size, p=probability)
        return next_word

    def updateRepresentation(self, conversations, N = 5000):
        # update the statistics using the given conversations

        # conversations is a list of lists of numbers (words)

        occurrence_counts = self.prob_matrix * N
        word_count = N
        for conv in conversations:
            word_count += (len(conv)*(len(conv)-1))/2
            for i in range(len(conv)):
                for j in range(i+1, len(conv)):
                    occurrence_counts[conv[i]][conv[j]] += 1
                    occurrence_counts[conv[j]][conv[i]] += 1
        self.prob_matrix = occurrence_counts / word_count

    def getDistanceMatrix(self):
        return self.prob_matrix[:]
