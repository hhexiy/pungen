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

    def __init__(self, max_num_senses=3, threshold=0.2):
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

    def save(self):
        pass

class TypeRecognizer2(object):

    types = ['person', 'group', 'organization', 'location']

    my_types = {
        'person': ['we', 'he', 'she', 'i', 'you', 'they', '<norp>', '<person>', 'who'],
        'group': ['we', 'they', '<norp>'],
        'organization': ['we', 'they', '<org>'],
        'location': ['<gpe>', '<loc>'],
        }

    # TODO: don't need type_dict_path given that we can the cache_path
    def __init__(self, type_dict_path='models/types.json', cache_path='.cache/concept.json'):
        if os.path.exists(type_dict_path):
            self.type_dict = json.load(open(type_dict_path))
        else:
            self.type_dict = {}
        for t, words in self.my_types.items():
            for w in words:
                self.add_type(w, [t])
        self.type_dict_path = type_dict_path

        self.cache_path = cache_path
        if os.path.exists(self.cache_path):
            self.cache = json.load(open(self.cache_path))
        else:
            self.cache = defaultdict(lambda : defaultdict())
            ensure_exist(self.cache_path)

    def save(self):
        ensure_exist(self.type_dict_path)
        json.dump(self.type_dict, open(self.type_dict_path, 'w'))
        json.dump(self.cache, open(self.cache_path, 'w'))

    def add_type(self, word, types):
        if not word in self.type_dict:
            self.type_dict[word] = types
        else:
            for t in types:
                if not t in self.type_dict[word]:
                    self.type_dict[word].append(t)

    def get_type(self, word):
        if word in self.type_dict:
            return self.type_dict[word]
        types = []
        for type in self.types:
            if self.is_type(word, type):
                types.append(type)
                break
        self.type_dict[word] = types
        return types

    def is_types(self, word, types):
        for type in types:
            if self.is_type(word, type):
                self.add_type(word, [type])
                return True
        return False

    def is_type(self, word, type):
        if word in self.type_dict and type in self.type_dict[word]:
            return True
        if not word in self.cache:
            self.cache[word] = {}
        if type in self.cache[word]:
            return self.cache[word][type]
        q = 'http://api.conceptnet.io/query?start=/c/en/{word}&end=/c/en/{type}&rel=/r/IsA'.format(word=word, type=type)
        obj = requests.get(q)
        if not obj:
            self.cache[word][type] = False
            return False
        else:
            obj = obj.json()
        if len(obj['edges']) > 0:
            if not word in self.type_dict:
                self.type_dict[word] = [type]
            else:
                self.type_dict[word].append(type)
            self.cache[word][type] = True
            return True
        self.cache[word][type] = False
        return False
