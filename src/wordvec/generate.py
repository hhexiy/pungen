import argparse
import os
import pickle
from nltk.corpus import wordnet as wn

import torch
from torch import LongTensor as LT
from torch import FloatTensor as FT

from model import Word2Vec, SGNS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--model_path', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--pun-words')
    parser.add_argument('--puns')
    parser.add_argument('--homo', action='store_true')
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

def topk_neighbors(word, owords, word2idx, idx2word, sgns, k=10):
    try:
        iword = [word2idx[word]]
    except KeyError:
        print('unknown word: {}'.format(word))
        return None
    ivectors = sgns.embedding.forward_i(iword)
    ovectors = sgns.embedding.forward_o(owords)
    scores = torch.matmul(ovectors, ivectors.t())
    probs = scores.squeeze().sigmoid()
    topk_prob, topk_id = torch.topk(probs, k)
    return [idx2word[id_] for id_ in topk_id if not idx2word[id_] in ('<unk>', word)]

def main(args):
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(args.data_dir, 'word2idx.dat'), 'rb'))
    vocab_size = len(idx2word)
    # TODO: use saved e_dim
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=1, weights=None)
    sgns.load_state_dict(torch.load(args.model_path))
    if args.cuda:
        sgns = sgns.cuda()
    sgns.eval()

    owords = range(vocab_size)
    fout = open(args.output, 'w')
    for (pun_word, alter_word, pun_sense, alter_sense), sent in zip(read_pun_word(args.pun_words, args.homo), read_pun(args.puns)):
        if pun_sense == alter_sense:
            continue
        pun_word_neighbors = topk_neighbors(pun_sense, owords, word2idx, idx2word, sgns, k=args.k + 5)
        alter_word_neighbors = topk_neighbors(alter_sense, owords, word2idx, idx2word, sgns, k=args.k + 5)
        if not pun_word_neighbors or not alter_word_neighbors:
            continue
        words = alter_word_neighbors[:2] + pun_word_neighbors[:2] + [alter_word]
        fout.write('{} | {} | {}\n'.format(' '.join([pun_sense, alter_sense, pun_word, alter_word]), ' '.join(words), sent))
    fout.close()

if __name__ == '__main__':
    main(parse_args())
