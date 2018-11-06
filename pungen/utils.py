import os
import logging
import inspect
import sys
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

# Copied from https://github.com/dmlc/gluon-nlp/blob/master/scripts/machine_translation/utils.py
def logging_config(filename=None,
                   level=logging.DEBUG,
                   console_level=logging.INFO,
                   no_console=False):
    """ Config the logging.

    Parameters
    ----------
    level : int
    console_level
    no_console: bool
        Whether to disable the console log
    """
    # Remove all the current handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(filename)s %(funcName)s: %(message)s')

    if filename is not None:
        ensure_exist(filename)
        logpath = filename
        print('All Logs will be saved to {}'.format(logpath))
        logfile = logging.FileHandler(logpath)
        logfile.setLevel(level)
        logfile.setFormatter(formatter)
        logging.root.addHandler(logfile)

    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)


