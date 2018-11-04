import argparse

from pungen.utils import sentence_iterator, Word

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--ner', action='store_true')
    args = parser.parse_args()
    with open(args.output, 'w') as fout:
        for s in sentence_iterator(args.input, ner=args.ner):
            if args.ner:
                ss = []
                [ss.extend(w[Word.TOKEN].split('_')) for w in s]
                ss = ' '.join([x.lower() for x in ss])
            else:
                ss = ' '.join([w[Word.TOKEN] for w in s])
            fout.write(ss + '\n')


