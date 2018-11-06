import argparse
import random
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--split', nargs='+', default=[0.8, 0.1, 0.1])
    parser.add_argument('--split-names', nargs='+', default=['train', 'valid', 'test'])
    args = parser.parse_args()
    return args

def main(args):
    assert len(args.split) == len(args.split_names)
    split_prob = [float(x) for x in args.split]
    assert sum(split_prob) == 1
    fouts = [open('{}/{}.txt'.format(args.output, name), 'w') for name in args.split_names]

    if args.shuffle:
        draws = np.random.multinomial(1, split_prob, size=1000)
        draws = np.where(draws > 0)[1]
        with open(args.input, 'r') as fin:
            for i, line in tqdm(enumerate(fin)):
                fouts[draws[i % 1000]].write(line)
    else:
        n = 0
        with open(args.input, 'r') as fin:
            for line in fin:
                n += 1
        split_cumsum = np.cumsum(split_prob) * n
        split_cumsum[-1] = n
        curr_split = 0
        with open(args.input, 'r') as fin:
            for i, line in enumerate(fin):
                if i >= split_cumsum[curr_split]:
                    curr_split += 1
                fouts[curr_split].write(line)

    for fout in fouts:
        fout.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
