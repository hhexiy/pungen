import json
import requests
import os
from collections import defaultdict
from nltk.corpus import wordnet as wn
import logging
logger = logging.getLogger('pungen')

from .utils import ensure_exist

class TypeRecognizer(object):
    tags = {
            'noun': wn.NOUN,
            'verb': wn.VERB,
            'adj': wn.ADJ,
            'adv': wn.ADV,
            }

    person_words = set(['we', 'he', 'she', 'i', 'you', 'they', 'who', 'him'])

    def __init__(self, max_num_senses=2, threshold=0.2):
        self.max_num_senses = max_num_senses
        self.threshold = threshold
        self.person = wn.synsets('person')[0]

    def get_type(self, word, tag):
        if word in self.person_words:
            return [self.person]
        pos = self.tags.get(tag)
        s = wn.synsets(word, pos=pos)
        return s

    def is_types(self, word, types, tag):
        types1 = types
        types2 = self.get_type(word, tag)
        scores = []
        for t1 in types1[:self.max_num_senses]:
            for t2 in types2[:self.max_num_senses]:
                scores.append(t1.path_similarity(t2))
        if not scores or max(scores) < self.threshold:
            return False
        return True
