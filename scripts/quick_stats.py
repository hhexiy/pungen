#!/usr/bin/python

import sys
import numpy as np
import re

def load_data(infile, delim='[ppl| ]+'):
    ppls = []
    with open(infile) as inf:
        for line in inf:
            elem = re.split(delim, line.strip())
            if len(elem) != 2:
                continue
            ppls.append(float(elem[1]))
    return np.array(ppls)


def load_prob(infile):
    freqs = []
    with open(infile) as inf:
        for line in inf:
            elem = line.strip().split(' ')
            if len(elem) != 2:
                print('abnormal line!')
                continue
            freqs.append(map(lambda x:log(float(x)), elem))
    return freqs 


if __name__ == '__main__':
    ppl_arrays = []
    for arg in sys.argv[1: -1]:
        ppl_arrays.append((load_data(arg+'.true.log'), load_data(arg+'.hypo.log')))
    freqs = np.array(load_prob(sys.argv[-1]))
    for pairs in ppl_arrays:
        assert len(pairs[0]) == len(pairs[1])
        print(sum(pairs[0]/freqs[0] > pairs[1]/freqs[1]))
        print(sum(np.array(ppl_arrays[-1][0]) < np.array(ppl_arrays[-1][1])))
    pun_diff_array = [np.array(ppl_arrays[i][0]) - np.array(ppl_arrays[j][0]) for i in range(len(ppl_arrays)-1) for j in range(i+1, len(ppl_arrays))]
    alter_diff_array = [np.array(ppl_arrays[i][1]) - np.array(ppl_arrays[j][1]) for i in range(len(ppl_arrays)-1) for j in range(i+1, len(ppl_arrays))]
    print('-'*89)
    for pel, ael in zip(pun_diff_array, alter_diff_array):
        print(sum(pel > 0), sum(ael < 0), sum(np.logical_and(pel > 0, ael < 0)), sum(np.logical_or(pel > 0, ael < 0)))
        #print(sum(np.logical_xor(pel > 0, ael > 0)))
        #print(sum(abs(pel) > abs(ael)))
