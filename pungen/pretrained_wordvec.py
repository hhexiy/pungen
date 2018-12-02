import argparse
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from fairseq.data.dictionary import Dictionary

class Glove(object):
    def __init__(self, vectors, vocab):
        self.vectors = vectors
        self.vocab = vocab

    def save(self, file_):
        obj = {
                'vectors': self.vectors,
                }
        with open(file_, 'wb') as fout:
            pickle.dump(obj, fout)

    @classmethod
    def from_pickle(cls, pkl_file, vocab_file):
        d = Dictionary.load(vocab_file)
        with open(pkl_file, 'rb') as fin:
            obj = pickle.load(fin)
            return cls(obj['vectors'], d)

    @classmethod
    def from_file(cls, vector_file, vocab, vec_size=300):
        #d = Dictionary.load(vocab_file)
        d = vocab

        vectors = np.ones((len(d), vec_size), dtype=np.float32)
        idx_to_token = []
        with open(vector_file, 'r', errors='ignore') as fin:
            for line in fin:
                ss = line.strip().split()
                num_tokens = len(ss) - vec_size
                word = ' '.join(ss[:num_tokens])
                if word in d.indices:
                    try:
                        #vectors.append([float(x) for x in ss[num_tokens:]])
                        vectors[d.index(word)] = [float(x) for x in ss[num_tokens:]]
                    except ValueError:
                        print(ss[0])
                        print(ss[1:])
                        import sys; sys.exit()
                    idx_to_token.append(word)

        return cls(vectors, d)

    def cosine_similarity(self, words1, words2):
        embeddings1 = [self.vectors[self.vocab.index(w)] for w in words1]
        embeddings2 = [self.vectors[self.vocab.index(w)] for w in words2]
        return cosine_similarity(embeddings1, embeddings2)

    def similarity_scores(self, word):
        word_id = self.vocab.index(word)
        #query_vec = self.vectors[word_id]
        #scores = np.matmul(self.vectors, query_vec)
        #scores = np.expand_dims(scores, 1)
        query_vec = np.expand_dims(self.vectors[word_id], 0)
        scores = cosine_similarity(self.vectors, query_vec)
        return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', default='vocab.txt', type=str)
    parser.add_argument('--vectors', default='vectors.txt', type=str)
    parser.add_argument('-n', default=20, type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    return args

def main(args):
    if os.path.exists(args.output):
        wordvec = Glove.from_pickle(args.output)
    else:
        wordvec = Glove.from_file(args.vectors, args.vocab)
        wordvec.save(args.output)
    wordvec.similarity_scores('people')

if __name__ == "__main__":
    args = parse_args()
    main(args)

