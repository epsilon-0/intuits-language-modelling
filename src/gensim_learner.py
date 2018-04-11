import random
from gensim.models import Word2Vec, KeyedVectors

model = Word2Vec(size=250, window=6, min_count=1, workers=4)

sentence_batches = []

## this would be the list of monthly sentences
## because this is too big instead you can make a generator using yeild
## or you can make a simple iterator, as below
## which reads the data month by month

# an example:


class SentenceIterator:
    def __iter__(self):
        return self

    def __next__(self):
        if random.choice([1, 2, 3]) == 3:
            raise StopIteration
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
