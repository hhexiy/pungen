import argparse
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

from pungen.utils import sentence_iterator, Word, get_lemma

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed-corpus', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--max-examples', '-n', type=int, default=-1)
    parser.add_argument('--delete-frac', type=float, default=0.3, help="Pick the word to delete from the first X% words")
    parser.add_argument('--window-size', type=int, default=1, help="Number of additional words to delete")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

def split_sent(words, delete_frac, window_size=1):
    """Generate two input parts.
    """
    N = len(words)
    n = max(1, int(delete_frac * N))
    deleted_keyword = None
    for i, w in enumerate(words):
        if i < n and w[Word.TAG] in ('NOUN', 'PROPN', 'PRON') and not w[Word.TOKEN] in STOP_WORDS:
            left, right = i, i+1
            deleted_keyword = get_lemma(w, Word)

            # delete [left, right)
            left = max(0, i - window_size)
            right = min(N, i + window_size + 1)

    if not deleted_keyword:
        return None, None, None

    template = [w[Word.TOKEN] for w in words[:left]] + \
            ['<placeholder>'] + \
            [w[Word.TOKEN] for w in words[right:]]

    deleted = [w[Word.TOKEN] for w in words[left:right]]

    return template, deleted_keyword, deleted


def main(args):
    with open(args.parsed_corpus, 'r') as fin, \
         open(args.output+'.src', 'w') as fsrc, \
         open(args.output+'.tgt', 'w') as ftgt:
        for words in sentence_iterator(args.parsed_corpus, args.max_examples):
            template, del_word, deleted = split_sent(words, args.delete_frac, args.window_size)
            if not template:
                continue
            # target
            ftgt.write('{}\n'.format(' '.join(deleted)))
            # source
            fsrc.write('{deleted}\n{temp}\n'.format(
                deleted=del_word,
                temp=' '.join(template)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
