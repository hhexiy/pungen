import argparse
import random
from collections import OrderedDict
from spacy.lang.en.stop_words import STOP_WORDS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keywords', nargs='+', default=['NOUN'])
    parser.add_argument('--subsample', action='store_true')
    parser.add_argument('--min-keywords', type=int, default=1)
    parser.add_argument('--max-keywords', type=int, default=8)
    parser.add_argument('--max-len', type=int, default=20)
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()
    return args

def valid_keyword(token):
    if token in STOP_WORDS:
        return False
    if token.startswith("'"):
        return False
    return True

def main(args):
    with open(args.input, 'r') as fin, \
         open(args.output+'.src', 'w') as fsrc, \
         open(args.output+'.tgt', 'w') as ftgt:
        for line in fin:
            s = [x.split('|') for x in line.strip().split()]
            if len(s) > args.max_len:
                continue
            try:
                sent = [token.lower() for token, tag in s]
            except ValueError:
                print(sent)
                continue
            s = [(token.lower(), tag) for token, tag in s if valid_keyword(token.lower())]
            if args.subsample:
                keywords = []
                for key_tag in args.keywords:
                    new_keywords = set([token for token, tag in s if tag == key_tag])
                    if len(new_keywords) > 0:
                        keywords.extend(list(new_keywords))
                        random.shuffle(keywords)
                        fsrc.write(' '.join(keywords) + '\n')
                        ftgt.write(' '.join(sent) + '\n')
            else:
                keywords = [token for token, tag in s if tag in args.keywords]
                # Remove duplicates
                keywords = list(OrderedDict.fromkeys(keywords))
                fsrc.write(' '.join(keywords) + '\n')
                ftgt.write(' '.join(sent) + '\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)
