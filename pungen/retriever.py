import argparse
import os, sys
import numpy as np
import time
import pickle
from functools import total_ordering
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import sentence_iterator, Word

import logging
logger = logging.getLogger('pungen')


@total_ordering
class Template(object):
    def __init__(self, tokens, keyword, id_):
        self.id = int(id_)
        self.tokens = tokens
        self.keyword_positions = [i for i, w in enumerate(tokens) if w == keyword]
        self.num_key = len(self.keyword_positions)
        self.keyword_id = None if self.num_key == 0 else max(self.keyword_positions)

    def __len__(self):
        return len(self.tokens)

    def replace_keyword(self, word):
        tokens = list(self.tokens)
        tokens[self.keyword_id] = word
        return tokens

    def __str__(self):
        return ' '.join(['[{}]'.format(w) if i == self.keyword_id else w for i, w in enumerate(self.tokens)])

    def __lt__(self, other):
        # Containing keyword is better
        if self.num_key == 0:
            return True
        # Fewer keywords is better
        if self.num_key > other.num_key:
            return True
        # Later keywords is better
        if self.keyword_id < other.keyword_id:
            return True
        return False

    def __eq__(self, other):
        if self.num_key == 0 and other.num_key == 0:
            return True
        if self.num_key == other.num_key and self.keyword_id == other.keyword_id:
            return True
        return False


class Retriever(object):
    def __init__(self, doc_files, path=None, overwrite=False):
        logger.info('reading retriever docs from {}'.format(' '.join(doc_files)))
        self.docs = [line.strip() for line in open(doc_files[0], 'r')]

        if overwrite or (path is None or not os.path.exists(path)):
            logger.info('building retriever index')
            self.vectorizer = TfidfVectorizer(analyzer=str.split)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)
            if path is not None:
                self.save(path)
        else:
            logger.info('loading retriever index from {}'.format(path))
            with open(path, 'rb') as fin:
                obj = pickle.load(fin)
                self.vectorizer = obj['vectorizer']
                self.tfidf_matrix = obj['tfidf_mat']

    def save(self, path):
        with open(path, 'wb') as fout:
            obj = {
                    'vectorizer': self.vectorizer,
                    'tfidf_mat': self.tfidf_matrix,
                    }
            pickle.dump(obj, fout)

    def query(self, keywords, k=1):
        features = self.vectorizer.transform([keywords])
        scores = self.tfidf_matrix * features.T
        scores = scores.todense()
        scores = np.squeeze(np.array(scores), axis=1)
        ids = np.argsort(scores)[-k:][::-1]
        return ids

    def valid_template(self, template):
        return template.num_key == 1

    def retrieve_pun_template(self, alter_word, len_threshold=10, pos_threshold=0.5, num_cands=500, num_templates=None):
        ids = self.query(alter_word, num_cands)
        templates = [Template(self.docs[id_].split(), alter_word, id_) for id_ in ids]
        templates = [t for t in templates if t.num_key > 0 and len(t.tokens) > len_threshold]
        if len(templates) == 0:
            logger.info('FAIL: no retrieved sentence contains the keyword {}.'.format(alter_word))
            return []

        valid_templates = [t for t in templates if self.valid_template(t)]
        if len(valid_templates) == 0:
            valid_templates = templates
        templates = sorted(valid_templates, reverse=True)[:num_templates]
        return templates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc-file', nargs='+', help='training corpus')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--path', default='models/retriever.pkl', help='retriever model path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--keywords', help='file containing keywords')
    parser.add_argument('--alterwords', help='file containing alternative words')
    parser.add_argument('--outfile', help='output file for the retrieved sentences')
    args = parser.parse_args()

    retriever = Retriever(args.doc_file, args.path, args.overwrite)

    if args.interactive:
        while True:
            alter_word, pun_word = input('Keywords:\n').split()
            alter_sents, pun_sents, pun_word_ids, alter_ori_sents = retriever.retrieve_pun_template(alter_word, num_cands=100)
            for ori_sent, sent in zip(alter_ori_sents, alter_sents):
                print(' '.join(sent))
                print(' '.join(ori_sent))
            if not alter_sents:
                print('No candidates found')
    elif args.outfile:
        with open(args.keywords, 'r') as fin, open(args.alterwords, 'r') as afin, open(args.outfile, 'w') as outf:
            for key, alter in zip(fin, afin):
                key = key.strip()
                alter = alter.strip()
                ids = retriever.query(key, 10)
                contents = [retriever.ori_docs[id_] for id_ in ids]
                contents.sort(key = lambda s: len(s), reverse=True)
#                print(line.strip())
                for ct in contents:
                    ct_list = ct.split()
                    try:
                        idx = ct_list.index(key)
                    except:
                        sys.stderr.write('cannot find the word %s!\n' % key)
                        continue
                    ct_list[idx] = alter
                    outf.write(ct.lower() + '\n')
                    print(' '.join(ct_list).lower())
                    break
