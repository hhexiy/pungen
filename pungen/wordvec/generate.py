import argparse
import os
import pickle
from nltk.corpus import wordnet as wn

import torch
from torch import LongTensor as LT
from torch import FloatTensor as FT
from fairseq.data.dictionary import Dictionary

from pungen.utils import get_lemma, STOP_WORDS
from .model import Word2Vec, SGNS

import logging
logger = logging.getLogger('pungen')

class SkipGram(object):
    def __init__(self, model, vocab, use_cuda):
        self.model = model
        self.vocab = vocab
        if use_cuda:
            self.model.cuda()
        self.model.eval()

    @classmethod
    def load_model(cls, vocab_path, model_path, embedding_size=300, cpu=False):
        d = Dictionary.load(vocab_path)
        vocab_size = len(d)
        model = Word2Vec(vocab_size=vocab_size, embedding_size=embedding_size)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=1, weights=None)
        logger.info('loading skipgram model')
        sgns.load_state_dict(torch.load(model_path))
        sgns.eval()
        use_cuda = torch.cuda.is_available() and not cpu
        return cls(sgns, d, use_cuda)

    def predict_neighbors(self, word, k=20, masked_words=None):
        # take lemma because skipgram is trained on lemmas
        lemma = get_lemma(word)
        word = lemma

        owords = range(len(self.vocab))

        # NOTE: 0 is <Lua heritage> in fairseq.data.dictionary
        masked_inds = [self.vocab.index(word), self.vocab.unk(), self.vocab.eos(), 0] + [self.vocab.index(w) for w in STOP_WORDS]
        if masked_words:
            masked_inds += [self.vocab.index(w) for w in masked_words]
        masked_inds = set(masked_inds)
        owords = [w for w in owords if not w in masked_inds and self.vocab.count[w] > 100]
        neighbors = self.topk_neighbors([word], owords, k=k)

        return neighbors

    def score(self, iwords, owords, lemma=False):
        """p(oword | iword)
        """
        if not lemma:
            iwords = [get_lemma(w) for w in iwords]
            owords = [get_lemma(w) for w in owords]
        iwords = [self.vocab.index(w) for w in iwords]
        owords = [self.vocab.index(w) for w in owords]
        ovectors = self.model.embedding.forward_o(owords)
        ivectors = self.model.embedding.forward_i(iwords)
        scores = torch.matmul(ovectors, ivectors.t())
        probs = scores.squeeze().sigmoid()
        return probs.data.cpu().numpy()

    # TODO: use self.score()
    def topk_neighbors(self, words, owords, k=10):
        """Find words in `owords` that are neighbors of `words` and are similar to `swords`.
        """
        vocab = self.vocab
        iwords = [vocab.index(word) for word in words]
        for iword, w in zip(iwords, words):
            if iword == vocab.unk():
                return []

        ovectors = self.model.embedding.forward_o(owords)
        scores = 0
        for iword in iwords:
            ivectors = self.model.embedding.forward_i([iword])
            score = torch.matmul(ovectors, ivectors.t())
            scores += score
        probs = scores.squeeze()#.sigmoid()

        topk_prob, topk_id = torch.topk(probs, min(k, len(owords)))
        return [vocab[owords[id_]] for id_ in topk_id]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skipgram-model', nargs=2, help='pretrained skipgram model [vocab, model]')
    parser.add_argument('--skipgram-embed-size', type=int, default=300, help='word embedding size in skipgram model')
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--pun-words')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('-k', help='number of neighbors to query', default=2, type=int)
    parser.add_argument('-n', help='number of examples to process', default=-1, type=int)
    parser.add_argument('--output')
    parser.add_argument('--logfile')
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
    skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1], embedding_size=args.skipgram_embed_size, cpu=args.cpu)
    if args.interactive:
        while True:
            word = input('word: ')
            topic_words = skipgram.predict_neighbors(word, k=args.k)
            print(topic_words)

    puns = json.load(open(args.pun_words))
    results = []
    for i, example in enumerate(puns):
        if i == args.n:
            break
        pun_word, alter_word = example['pun_word'], example['alter_word']
        alter_topic_words = skipgram.predict_neighbors(alter_word, k=args.k)
        pun_topic_words = skipgram.predict_neighbors(pun_word, k=args.k)
        logger.debug(alter_word)
        logger.debug(alter_topic_words)
        logger.debug(pun_word)
        logger.debug(pun_topic_words)
        r = {
                'id': example['id'],
                'pun_word': pun_word,
                'alter_word': alter_word,
                'pun_topic_words': pun_topic_words,
                'alter_topic_words': alter_topic_words
            }
        results.append(r)
    json.dump(results, open(args.output, 'w'))


if __name__ == '__main__':
    import json
    from pungen.utils import logging_config
    args = parse_args()
    logging_config(filename=args.logfile)
    main(args)
