import argparse
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

from src.utils import sentence_iterator, Word

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

def build_tfidf_vectorizer(file_, n):
    vectorizer = TfidfVectorizer(analyzer=str.split)
    def doc_iterator(file_, n):
        for words in sentence_iterator(file_, n):
            yield ' '.join([w[1] for w in words])
    vectorizer.fit(doc_iterator(file_, n))
    return vectorizer

def split_sent(words, delete_frac, window_size=1):
    """Generate two input parts.
    """
    N = len(words)
    n = max(1, int(delete_frac * N))
    #delete_ids = []
    deleted_word = None
    for i, w in enumerate(words):
        if i < n and w[Word.TAG in ('NOUN', 'PROPN', 'PRON') and not w[Word.TOKEN in STOP_WORDS:
            left, right = i, i+1
            #delete_ids.append(i)
            deleted_word = w[Word.TOKEN

            # delete [left, right)
            left = max(0, i - window_size)
            right = min(N, i + window_size + 1)
            #if i > 0: #and not words[i-1][Word.TAG in ('VERB',):
            #    delete_ids.insert(0, i-1)
            #if i < N - 1: #and not words[i+1][Word.TAG in ('VERB', 'NOUN', 'PROPN', 'PRON'):
            #    delete_ids.append(i+1)
            #break

    if not deleted_word:
        return None, None, None, None

    #template = [w[Word.TOKEN for w in words[:delete_ids[0]]] + \
    #        ['<placeholder>'] + \
    #        [w[Word.TOKEN for w in words[delete_ids[-1]+1:]]
    template = [w[Word.TOKEN for w in words[:left]] + \
            ['<placeholder>'] + \
            [w[Word.TOKEN for w in words[right:]]

    #if insert_mode == 'none':
    #    insert = None
    #elif insert_mode == 'self':
    #    insert = delete_word
    #elif insert_mode == 'related':
    #    for i, w in enumerate(words[::-1]):
    #        if i < n and w[Word.TAG in ('NOUN', 'VERB', 'ADJ'):
    #            insert = w[Word.TOKEN
    #            break
    related_word = '<empty>'
    for i, w in enumerate(words[::-1]):
        if i < n and w[Word.TAG in ('NOUN', 'VERB', 'ADJ') and not w[Word.TOKEN in STOP_WORDS:
            related_word = w[Word.TOKEN
            break

    original = [w[Word.TOKEN for w in words]

    return template, deleted_word, related_word, original


def main(args):
    with open(args.parsed_corpus, 'r') as fin, \
         open(args.output+'.src', 'w') as fsrc, \
         open(args.output+'.tgt', 'w') as ftgt:
        for words in sentence_iterator(args.parsed_corpus, args.max_examples):
            template, del_word, rel_word, original = split_sent(words, args.delete_frac, args.window_size)
            if not template:
                continue
            # target
            sent = ' '.join([w[Word.TOKEN for w in words])
            ftgt.write('{}\n'.format(sent))
            # source
            fsrc.write('{deleted}\n{related}\n{temp}\n'.format(
                deleted=del_word,
                related=rel_word,
                temp=' '.join(template)))
            if args.debug:
                fsrc.write('{}\n'.format(' '.join(original)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
