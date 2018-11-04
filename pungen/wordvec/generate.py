import argparse
import os
import pickle
from nltk.corpus import wordnet as wn
import logging
logger = logging.getLogger('pungen')
logger.setLevel(logging.INFO)

import torch
from torch import LongTensor as LT
from torch import FloatTensor as FT
from fairseq.data.dictionary import Dictionary

from .model import Word2Vec, SGNS
from ..pretrained_wordvec import Glove

class SkipGram(object):
    def __init__(self, model, vocab, glove, use_cuda):
        self.model = model
        self.vocab = vocab
        self.glove = glove
        if use_cuda:
            self.model.cuda()

    @classmethod
    def load_model(cls, vocab_path, model_path, cpu=False):
        d = Dictionary.load(vocab_path)
        vocab_size = len(d)
        model = Word2Vec(vocab_size=vocab_size, embedding_size=300)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=1, weights=None)
        print('| loading skipgram model')
        sgns.load_state_dict(torch.load(model_path))
        sgns.eval()
        use_cuda = torch.cuda.is_available() and not cpu
        glove = Glove.from_pickle('data/onebillion/glove.pkl', vocab_path)
        return cls(sgns, d, glove, use_cuda)

    def predict_neighbors(self, word, k=20, sim_words=None, cands=None):
        if cands:
            owords = [self.vocab.index(w) for w in cands]
        else:
            owords = range(len(self.vocab))
        neighbors = self.topk_neighbors([word], owords, k=k, swords=sim_words, wordvec=self.glove)
        return neighbors

    def topk_neighbors(self, words, owords, wordvec=None, swords=None, k=10):
        """Find words in `owords` that are neighbors of `words` and are similar to `swords`.
        """
        vocab = self.vocab
        iwords = [vocab.index(word) for word in words]
        for iword, w in zip(iwords, words):
            if iword == vocab.unk():
                logger.info('unknown input word: {}'.format(w))
                return []

        ovectors = self.model.embedding.forward_o(owords)
        scores = 0
        for iword in iwords:
            ivectors = self.model.embedding.forward_i([iword])
            score = torch.matmul(ovectors, ivectors.t())
            # TODO: figure out why this line doesn't work
            #score = FT(wordvec.similarity_scores(iword)).cuda()
            scores += score
        probs = scores.squeeze().sigmoid()

        # Compute similary by word vectors (i.e. forward_i)
        #ovectors = self.model.embedding.forward_i(owords)
        #if swords:
        #    swords = [vocab.index(word) for word in swords]
        #    for sword in swords:
        #        svectors = self.model.embedding.forward_i([sword])
        #        score = torch.matmul(ovectors, svectors.t())
        #        scores += score
        if swords:
            scores = 0
            for sword in swords:
                # TODO: decide on cuda
                score = FT(wordvec.similarity_scores(sword)).cuda()
                scores += score
            probs2 = scores.squeeze().sigmoid()
            #probs2 = probs2[owords]
        else:
            probs2 = None

        #if probs2 is not None:
        #    probs = 0.3*probs + 0.7*probs2

        to_remove = iwords + [vocab.unk(), 0]
        # TODO: use mask for cands
        #probs[to_remove] = 0.
        topk_prob, topk_id = torch.topk(probs, min(k, len(owords)))
        return [vocab[id_] for id_ in topk_id]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--vocab', type=str, default='./data/', help="data directory path")
    parser.add_argument('--model-path', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--pun-words')
    parser.add_argument('--puns')
    parser.add_argument('--homo', action='store_true')
    parser.add_argument('--interact', action='store_true')
    parser.add_argument('-k', help='number of neighbors to query', default=2, type=int)
    parser.add_argument('--output')
    return parser.parse_args()

def get_sense(word):
    try:
        synset = wn.lemma_from_key(word).synset().lemma_names()
    except Exception:
        return word
    for w in synset:
        if w != word:
            return w
    return word

def read_pun_word(filename, homo):
    with open(filename, 'r') as fin:
        for line in fin:
            ss = line.strip().split('\t')
            pun_word = ss[1].split('%')[0]
            alter_word = ss[2].split('%')[0]
            if homo:
                pun_sense = get_sense(ss[1])
                alter_sense = get_sense(ss[2])
            else:
                pun_sense, alter_sense = pun_word, alter_word
            yield pun_word, alter_word, pun_sense, alter_sense

def read_pun(filename):
    with open(filename, 'r') as fin:
        for line in fin:
            yield line.strip()

def main(args):
    print('loading dict')
    d = Dictionary.load(args.vocab)
    vocab_size = len(d)
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=1, weights=None)
    glove = Glove.from_pickle('data/onebillion/glove.pkl', args.vocab)
    print('loading model')
    sgns.load_state_dict(torch.load(args.model_path))
    if args.cuda:
        sgns = sgns.cuda()
    sgns.eval()

    if args.interact:
        owords = range(vocab_size)
        while True:
            #s = input('word: ')
            #iwords = s.split()
            #s = input('sim word: ')
            #swords = s.split()
            iwords = ['gene']
            swords = ['gene']
            neighbors = topk_neighbors(iwords, owords, d, sgns, k=args.k + 5, swords=swords, wordvec=glove)
            print(neighbors)
            import sys; sys.exit()


    owords = range(vocab_size)
    fout = open(args.output, 'w')
    i = 0
    for (pun_word, alter_word, pun_sense, alter_sense), sent in zip(read_pun_word(args.pun_words, args.homo), read_pun(args.puns)):
        if pun_sense == alter_sense:
            continue
        pun_word_neighbors = topk_neighbors([pun_sense], owords, d, sgns, k=args.k + 5)
        alter_word_neighbors = topk_neighbors([alter_sense], owords, d, sgns, k=args.k + 5)
        if not pun_word_neighbors or not alter_word_neighbors:
            continue
        #print(pun_word_neighbors)
        #print(alter_word_neighbors)
        words = pun_word_neighbors[:1] + alter_word_neighbors[:1] + [alter_word]
        fout.write('{} | {} | {}\n'.format(' '.join([pun_sense, alter_sense, pun_word, alter_word]), ' '.join(words), sent))
        i += 1
    fout.close()

if __name__ == '__main__':
    main(parse_args())
