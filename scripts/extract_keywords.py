import argparse
import random
from collections import OrderedDict
from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS.update(['doesn', 'hadn', 'don'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keywords', nargs='+', default=['NOUN'])
    parser.add_argument('--subsample', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--tgt-only', action='store_true')
    parser.add_argument('--keywords-only', action='store_true')
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--max-len', type=int, default=20)
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()
    return args

def valid_keyword(token):
    if len(token) < 3:
        return False
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
            words = line.strip().split()
            if len(words) > args.max_len:
                continue
            s = []
            for word in words:
                tags = word.split('|')
                if len(tags) == 4 and tags[-1] != 'SPACE':
                    s.append(tags)
            try:
                sent = [token for _, token, lemma, tag in s]
            except ValueError:
                print(line)
                continue
            if args.tgt_only:
                ftgt.write(' '.join(sent) + '\n')
            #s = [(token.lower(), tag) for token, tag in s if valid_keyword(token.lower())]
            elif args.subsample:
                keywords = []
                for key_tag in args.keywords:
                    new_keywords = set([token for token, tag in s if tag == key_tag])
                    if len(new_keywords) > 0:
                        keywords.extend(list(new_keywords))
                        random.shuffle(keywords)
                        fsrc.write(' '.join(keywords) + '\n')
                        ftgt.write(' '.join(sent) + '\n')
            elif args.final:
                keywords = [token for token, tag in s if tag in args.keywords]
                # Remove duplicates
                keywords = list(OrderedDict.fromkeys(keywords))
                if not keywords:
                    continue
                n = int(len(keywords) * 0.2)
                for i in range(1, max(1, n)+1):
                    fsrc.write(keywords[-i] + '\n')
                    if args.target:
                        new_sent = ['<target>' if x == keywords[-i] else x for x in sent]
                    else:
                        new_sent = sent
                    if args.keywords_only:
                        ftgt.write(' '.join(keywords) + '\n')
                    else:
                        ftgt.write(' '.join(new_sent) + '\n')
            else:
                keywords = [token for _, token, lemma, tag in s if tag in args.keywords and valid_keyword(token)]
                if '-PRON-' in keywords:
                    print(s)
                    print(args.keywords)
                    print(keywords)
                    import sys; sys.exit()
                # Remove duplicates
                keywords = list(OrderedDict.fromkeys(keywords))
                if not keywords:
                    continue
                # Cut half
                #if len(keywords) > 8:
                #    keywords = [x for i, x in enumerate(keywords) if i % 2 == 0]
                fsrc.write(' '.join(keywords) + '\n')
                ftgt.write(' '.join(sent) + '\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)
