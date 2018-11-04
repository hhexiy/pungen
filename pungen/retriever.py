import argparse
import os, sys
import numpy as np
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import sentence_iterator, Word


class Retriever(object):
    def __init__(self, doc_files, path=None, overwrite=False):
        print('reading docs')
        self.docs = [line.strip() for line in open(doc_files[0], 'r')]
        # TODO: in future we will not do NER abstraction, so no need to have ori_docs
        if len(doc_files) > 1:
            self.ori_docs = [line.strip() for line in open(doc_files[1], 'r')]
        else:
            self.ori_docs = self.docs

        if overwrite or (path is None or not os.path.exists(path)):
            print('building retriever index')
            self.vectorizer = TfidfVectorizer(analyzer=str.split)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)
            if path is not None:
                self.save(path)
        else:
            print('loading retriever index')
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

    def check_pos(self, sent, word, pos_threshold):
        pos = [i for i, w in enumerate(sent) if w == word]
        if len(pos) != 1:
            return False
        p = pos[0]
        if p < min(int(len(sent) * pos_threshold), len(sent) - 1):
            return False
        return True

    def check_len(self, sent, len_threshold):
        if len(sent) < len_threshold:
            return False
        return True

    def retrieve_pun_template(self, pun_word, alter_word, len_threshold=10, pos_threshold=0.5, num_cands=500, num_templates=None):
        ids = self.query(alter_word, num_cands)
        print('retriever returned {} candidates.'.format(len(ids)))
        sents = [self.docs[id_].split() for id_ in ids]
        ori_sents = [self.ori_docs[id_].split() for id_ in ids]
        alter_sents = []
        alter_ori_sents = []
        pun_sents = []
        pun_word_ids = []
        count = 0
        for ori_sent, sent in zip(ori_sents, sents):
            if self.check_len(sent, len_threshold) and \
                    self.check_pos(sent, alter_word, pos_threshold):
                count += 1
                if num_templates and count > num_templates:
                    break
                alter_sents.append(sent)
                alter_ori_sents.append(ori_sent)
                pun_sents.append([x if x != alter_word else pun_word for x in sent])
                id_ = [i for i, w in enumerate(sent) if w == alter_word][0]
                pun_word_ids.append(id_)
        print('{} satisfies pun constraints.'.format(count))
        return alter_sents, pun_sents, pun_word_ids, alter_ori_sents


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
            alter_sents, pun_sents, pun_word_ids, alter_ori_sents = retriever.retrieve_pun_template(pun_word, alter_word, num_cands=100)
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
