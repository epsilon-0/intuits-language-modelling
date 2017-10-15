import autograd.numpy as np
from autograd import grad, optimizers


def printDictionary(dists, vocab):
    # given a dictionary of distances (sparse matrix)
    # returns a string of the matrix
    new_dists = [[1000.0 for i in range(len(vocab))]
                 for j in range(len(vocab))]
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if (vocab[j] in dists[vocab[i]]):
                new_dists[i][j] = float(dists[vocab[i]][vocab[j]])
                new_dists[j][i] = float(dists[vocab[i]][vocab[j]])
    return '\n'.join(['\t'.join(map(str, i)) for i in new_dists])


def generateRandomCorpus(size, density=0.2, prob=0.02):
    # generates a text corpus with randomized counts
    # density : number of pairs that actually get non-zero value
    # prob : TODO : make bernoulli
    corpus = np.array([[0. for i in range(size)] for j in range(size)])
    for i in range(size):
        for j in range(i + 1, size):
            if (np.random.random() < density):
                count = np.random.randint(1, 50)
                corpus[i][j] = count
                corpus[j][i] = corpus[i][j]
    return corpus


def squaredNorm(corpus, dimension=50, num_iters=100, step_size=0.001):
    # given a corpus make word2vec using squaredNorm
    # from RAND-WALK paper
    num_vectors = len(corpus)

    def optimization(params, t=0):
        answer = 0.
        for i in range(num_vectors):
            for j in range(num_vectors):
                answer += np.min([corpus[i][j], 150]) * (
                    np.log(1 + corpus[i][j]) - np.square(
                        np.linalg.norm(params[i] + params[j])) - params[-1][0])
        return answer

    vectors = np.array([[2 * np.random.random() for i in range(dimension)]
                        for j in range(num_vectors)])
    params = np.concatenate(
        [vectors, np.array([[0. for i in range(dimension)]])])
    ograd = grad(optimization)
    new_params = optimizers.adam(
        ograd, params, num_iters=num_iters, step_size=step_size)
    return (new_params[:-1], new_params[-1][0])


def readVecspaceFile(file_name):
    inp = open(file_name, "r")
    lines = inp.readlines()
    vecspace = np.array([list(map(lambda x: float(x), i)) for i in lines[:-1]])
    C = float(lines[-1].strip())
    inp.close()
    return (vecspace, C)
