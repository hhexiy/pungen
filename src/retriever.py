import argparse
import os
import numpy as np
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class Retriever(object):
    def __init__(self, docs, path=None):
        if path is None:
            self.vectorizer = TfidfVectorizer(analyzer=str.split)
            self.tfidf_matrix = self.vectorizer.fit_transform(docs)
        else:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc-file')
    parser.add_argument('--keywords')
    parser.add_argument('--path', default='models/retriever.pkl')
    args = parser.parse_args()

    start = time.time()
    print('Reading docs')
    with open(args.doc_file, 'r') as fin:
        docs = [line.strip() for line in fin]
    print('{} s'.format(time.time() - start))

    start = time.time()
    print('Building matrix')
    if args.path and os.path.exists(args.path):
        print('Loading from', args.path)
        retriever = Retriever(docs, args.path)
    else:
        retriever = Retriever(docs)
        if args.path is not None:
            retriever.save(args.path)
    print('{} s'.format(time.time() - start))

    with open(args.keywords, 'r') as fin:
        for line in fin:
            ids = retriever.query(line.strip(), 5)
            print(line.strip())
            for id_ in ids:
                print(docs[id_])
            print('')

    #while True:
    #    keywords = input('Keywords:\n')
    #    ids = retriever.query(keywords, 20)
    #    for id_ in ids:
    #        print(docs[id_])

