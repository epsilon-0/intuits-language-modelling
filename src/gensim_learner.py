import random
from gensim.models import Word2Vec, KeyedVectors
import gensim
model = Word2Vec(size=250, window=6, min_count=1, workers=4)

sentence_batches = []

## this would be the list of monthly sentences
## because this is too big instead you can make a generator using yeild
## or you can make a simple iterator, as below
## which reads the data month by month

# an example:


class SentenceIterator:
    def __init__(self):
        self.m = 0
    def __iter__(self):
        return self

    def __next__(self):
        if self.m > 10:
            raise StopIteration
        self.m += 1
        return [["cat", "says", "meow"], ["dog", "says", "woof"]]


c = 0
update = False
for batch in SentenceIterator():
    #a batch is a list of sentences
    #a sentence is a list of words.
    if c == 1:
        update = True
    c += 1
    model.build_vocab(batch, update=update)
    model.train(batch, total_examples=model.corpus_count, epochs=100)
    KeyedVectors.save_word2vec_format(model.wv, fname=str(c)+".txt" , binary=False)






class SentenceIterator:
    def _init__(self, inFile):
        self.fId = open(inFile, "r")
    def _iter__(self):
        return self
    def __next__(self):
    line = self.fId.readline()
    if not line:
        self.fId.close()
        raise StopIteration
        sentences = gensim.summarization.textcleaner.split_sentences(line) # first split comment into sentences
        # then split each sentence into words
        for i in sentences:

    return line.strip().split(" ")


class SentenceIterator:
    def __init__(self, inFile):
        self.fId = open(inFile, "r")
    def __iter__(self):
        return self
    def __next__(self):
        line = self.fId.readline()
        if not line:
            self.fId.close()
            raise StopIteration
        sentences = gensim.summarization.textcleaner.split_sentences(line)
        sentencewords = [list(gensim.summarization.textcleaner.tokenize_by_word(sent)) for sent in sentences]
        return sentencewords



word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format




