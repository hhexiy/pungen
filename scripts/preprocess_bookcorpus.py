import argparse
import re
from unidecode import unidecode
from tqdm import tqdm

from pungen.utils import get_spacy_nlp
nlp = get_spacy_nlp()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--min-len', type=int, default=15, help='minimum sentence length')
    parser.add_argument('--max-len', type=int, default=30, help='maximum sentence length')
    args = parser.parse_args()
    return args

def sentence_iter(file_):
    with open(file_, 'r', errors='ignore') as fin:
        cache = [re.sub('\s+', ' ', unidecode(line.strip()).strip()) for line in fin]
        docs = nlp.pipe(cache)
        for doc in docs:
            yield doc

def write_parsed_sentence(doc):
    all_tokens = [x for x in doc]

    def tok_to_str(tok):
        return '{ori}|{lemma}|{pos}'.format(ori=tok.text, lemma=tok.lemma_, pos=tok.pos_)

    def ent_to_str(e):
        type_ = '<{}>'.format(e.label_.lower())
        return '{ori}|{low}|{lemma}|{pos}'.format(
                ori=e.text.replace(' ', '_'),
                low=type_,
                lemma=type_, pos=e.label_)

    if not doc.ents:
        tokens = [tok_to_str(w) for w in all_tokens]
    else:
        last_tok = 0
        tokens = []
        for e in doc.ents:
            for i in range(last_tok, e.start):
                tokens.append(tok_to_str(all_tokens[i]))
            tokens.append(ent_to_str(e))
            last_tok = e.end
        for i in range(last_tok, len(all_tokens)):
            tokens.append(tok_to_str(all_tokens[i]))

    return ' '.join(tokens)

def main(args):
    count = 0
    with open(args.output, 'w') as fout, open(args.input, 'r') as fin:
        for s in tqdm(sentence_iter(args.input)):
            if len(s) >= args.min_len and len(s) <= args.max_len:
                s = write_parsed_sentence(s)
                fout.write('{}\n'.format(s))
                #count += 1
                #if count > 10:
                #    break

if __name__ == '__main__':
    args = parse_args()
    main(args)
