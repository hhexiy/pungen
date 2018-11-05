import os
from enum import IntEnum
Word = IntEnum('Word', [(x, i) for i, x in enumerate('TOKEN LEMMA TAG'.split())])
Word_ner = IntEnum('Word', [(x, i) for i, x in enumerate('SURFACE TOKEN LEMMA TAG'.split())])

def sentence_iterator(file_, n=-1, ner=False):
    with open(file_, 'r') as fin:
        for i, line in enumerate(fin):
            if i == n:
                break
            line = line.strip().split()
            words = []
            for w in line:
                tags = w.split('|')
                if ((ner and len(tags) == 4) or (not ner and len(tags) == 3)) \
                        and tags[-1] != 'SPACE':
                    words.append(tags)
            yield words

def get_lemma(word, props):
    if word[props.LEMMA] == '-PRON-':
        return word[props.TOKEN]
    return word[props.LEMMA]

def ensure_exist(path, is_dir=False):
    if not is_dir:
        dir_ = os.path.dirname(path)
    else:
        dir_ = path
    if not os.path.exists(dir_):
        os.makedirs(dir_)
