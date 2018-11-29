import os
import logging
import sys
from enum import IntEnum
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en.stop_words import STOP_WORDS

Word = IntEnum('Word', [(x, i) for i, x in enumerate('TOKEN LEMMA TAG'.split())])

whitespace_tokenizer = lambda nlp: Tokenizer(nlp.vocab, prefix_search=None, suffix_search=None, infix_finditer=None, token_match=None)

EPS = 1e-12

def get_spacy_nlp(tokenizer='whitespace', disable=['ner', 'parser']):
    nlp = spacy.load('en_core_web_sm', disable=disable)
    if tokenizer == 'whitespace':
        nlp.tokenizer = whitespace_tokenizer(nlp)
    elif tokenizer == 'default':
        pass
    else:
        raise ValueError('unknown tokenizer {}'.format(tokenizer))
    return nlp

nlp = get_spacy_nlp()

def get_lemma(word, parsed=False):
    if not parsed:
        _word = nlp(word)[0]
    else:
        _word = word

    if _word.lemma_ != '-PRON-':
        lemma = _word.lemma_
    else:
        lemma = _word.text

    return lemma

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

def ensure_exist(path, is_dir=False):
    if not is_dir:
        dir_ = os.path.dirname(path)
    else:
        dir_ = path
    if not os.path.exists(dir_):
        os.makedirs(dir_)

# Adapted from https://github.com/dmlc/gluon-nlp/blob/master/scripts/machine_translation/utils.py
def logging_config(filename=None,
                   level=logging.DEBUG,
                   console_level=logging.INFO,
                   no_console=False):
    logger = logging.getLogger('pungen')

    # Remove all the current handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.handlers = []

    logger.setLevel(level)
    formatter = logging.Formatter('%(filename)s %(funcName)s: %(message)s')

    if filename is not None:
        ensure_exist(filename)
        logpath = filename
        print('All Logs will be saved to {}'.format(logpath))
        logfile = logging.FileHandler(logpath, mode='w')
        logfile.setLevel(level)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)

    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logger.addHandler(logconsole)


