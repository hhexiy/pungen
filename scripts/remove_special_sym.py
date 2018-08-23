#!/usr/bin/python

import sys, re

if __name__ == '__main__':
    symbols = '[#\.\*]+'
    with open(sys.argv[1]) as inf, open(sys.argv[2], 'w') as outf:
        for line in inf:
            elems = re.split(symbols, line)
            outf.write(' '.join(elems))
