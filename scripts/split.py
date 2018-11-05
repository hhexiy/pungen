import argparse
import random
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--split', nargs='+', default=[0.8, 0.1, 0.1])
    parser.add_argument('--split-names', nargs='+', default=['train', 'valid', 'test'])
    args = parser.parse_args()
    return args

def main(args):
    assert len(args.split) == len(args.split_names)
    split_prob = [float(x) for x in args.split]
    assert sum(split_prob) == 1
    draws = np.random.multinomial(1, split_prob, size=1000)
    draws = np.where(draws > 0)[1]
    fouts = [open('{}/{}.txt'.format(args.output, name), 'w') for name in args.split_names]
    with open(args.input, 'r') as fin:
        for i, line in tqdm(enumerate(fin)):
            fouts[draws[i % 1000]].write(line)

if __name__ == '__main__':
    args = parse_args()
    main(args)
