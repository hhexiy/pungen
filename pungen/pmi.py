import argparse
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer

class WordPMI(object):
    def __init__(self, docs, path=None):
        if path is None:
            self.vectorizer = CountVectorizer(analyzer=str.split)
            doc_term_matrix = self.vectorizer.fit_transform(docs)
            self.co_matrix = np.matmul(doc_term_matrix.T, doc_term_matrix)
        else:
            with open(path, 'rb') as fin:
                obj = pickle.load(fin)
                self.vectorizer = obj['vectorizer']
                self.co_matrix = obj['co_matrix']
        self.vocab = self.vectorizer.vocabulary_

    def save(self, path):
        with open(path, 'wb') as fout:
            obj = {
                    'vectorizer': self.vectorizer,
                    'co_matrix': self.co_matrix,
                    }
            pickle.dump(obj, fout)

    def top_pmi_words(self, word, k):
        id_ = self.vocab[word]
        co_ids = self.co_matrix[id_] > 0
        co_ids = [i for i, val in enumerate(co_ids[0]) if val]
        word = np.sum(self.co_matrix[id_], axis=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc-file')
    parser.add_argument('--path', default='models/pmi.pkl')
    args = parser.parse_args()
    return args

def main(args):
    start = time.time()
    print('Reading docs')
    with open(args.doc_file, 'r') as fin:
        docs = [line.strip() for line in fin]
    print('{} s'.format(time.time() - start))

    start = time.time()
    print('Building matrix')
    if args.path and os.path.exists(args.path):
        print('Loading from', args.path)
        word_pmi = WordPMI(docs, args.path)
    else:
        word_pmi = WordPMI(docs)
        if args.path is not None:
            word_pmi.save(args.path)
    print('{} s'.format(time.time() - start))


if __name__ == "__main__":
    args = parse_args()
    main(args)

