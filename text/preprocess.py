from os import listdir
from os.path import isfile, join

shakespeare = '../data/ShakespearePlaysPlus'

def compile_shakespeare():
    files = []
    
    for category in listdir(shakespeare):
        files += [join(shakespeare, category, f) for f in listdir(join(shakespeare, category)) if f.endswith('.txt')]

    sentences = []
        
    for f in files:
        sentences += [line.strip().lower().split() for line in open(f, 'r').readlines() if not line.startswith('<')]

    return sentences
