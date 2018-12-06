import operator

from gensim.models import Word2Vec as w2v
from gensim.models import KeyedVectors

def load_w2v(f):
    '''
        Loads a file in the format of the Google pretrained binary.
    '''

    return KeyedVectors.load_word2vec_format(f, binary=True)

def load_glove(f, k=None):
    '''
        Loads a file in the format provided through the GloVe website.

        k: optional cap on vocabulary size.
    '''

    embeddings = {}

    lines = open(f, 'r').readlines()

    if k != None:
        lines = lines[:k]
    
    for line in lines:
        # assume words are sorted by frequency
        splits = line.split()

        word = splits[0]
        vec = np.array([float(x) for x in splits[1:]])

        embeddings[word] = vec

    return embeddings

def train_w2v(f, tokenizer='nltk',
              sz=100, win=5, mc=2, wk=4,
              save_file='trained.model'):
    '''
        Trains a gensim word2vec model on the contents of a file. Assumes one
        sentence per line.
    '''

    sentences = []

    for line in open(f, 'r').readlines():
        # split up lines... somehow
        if tokenizer == 'nltk':
            sentence = nltk.word_tokenize(line)
        elif tokenizer == 'spacy':
            # TODO
            pass
        else:
            sentence = line.split()

        sentences.append(sentence)

    # train the model
    model = Word2Vec(sentences, size=sz, window=win, min_count=mc, workers=wk)

    if save_file:
        model.save(save_file)

    return model

def find_top_k(v, k, keys):
    '''
        Returns top k words and top k vectors of a vocabulary, the latter as
        rows of a numpy matrix.
    '''

    counts = {}

    for word, vo in keys:
        counts[word] = vo.count

    # items will be (word, 
    counts = sorted(counts.items(), key=operator.itemgetter(1))

    print counts[:10]

    words = [c[0] for c in counts[:k]]

    vectors = np.matrix([v[w] for w in words])

    return words, vectors
