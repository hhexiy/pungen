# parsed words
from enum import IntEnum
Word = IntEnum('Word', [(x, i) for i, x in enumerate('SURFACE TOKEN LEMMA TAG'.split())])

def sentence_iterator(file_, n=-1):
    with open(file_, 'r') as fin:
        for i, line in enumerate(fin):
            if i == n:
                break
            line = line.strip().split()
            words = []
            for w in line:
                tags = w.split('|')
                if len(tags) == 4 and tags[-1] != 'SPACE':
                    words.append(tags)
            yield words

