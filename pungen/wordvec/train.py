# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch as t
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from fairseq.data.dictionary import Dictionary

from .model import Word2Vec, SGNS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    return parser.parse_args()


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None, window=5):
        self.window = window
        if ws is not None:
            self.data = []
            for iword, owords in self.read_data(datapath):
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = [(iword, owords) for iword, owords in self.read_data(datapath)]

    def read_data(self, datapath):
        n = 2 * self.window + 1
        with open(datapath, 'rb') as fin:
            print('Reading binary data...')
            data = np.fromfile(fin, dtype=np.uint16, count=-1)
            for i in range(0, len(data), n):
                yield data[i], data[i+1:i+n]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return int(iword), np.array(owords, dtype=np.int)


def train(args):
    d = Dictionary.load(args.vocab)
    wf = np.array(d.count)
    wf[wf == 0] = 1
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(d)
    weights = wf if args.weights else None
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, weights=weights, pad=d.unk())
    if os.path.isfile(modelpath) and args.conti:
        sgns.load_state_dict(t.load(modelpath))
    if args.cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters())
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and args.conti:
        optim.load_state_dict(t.load(optimpath))
    dataset = PermutedSubsampledCorpus(args.data, ws=ws)
    dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True, num_workers=0)
    for epoch in range(1, args.epoch + 1):
        total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())

        t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}-e{}.pt'.format(args.name, epoch)))
        t.save(optim.state_dict(), os.path.join(args.save_dir, '{}-e{}.optim.pt'.format(args.name, epoch)))


if __name__ == '__main__':
    train(parse_args())
