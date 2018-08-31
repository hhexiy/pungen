import argparse
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

# parsed words
SURFACE, TOKEN, LEMMA, TAG = 0, 1, 2, 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed-corpus', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--max-examples', '-n', type=int, default=-1)
    parser.add_argument('--insert', default='related', choices=['self', 'related', 'none'])
    args = parser.parse_args()
    return args

def sentence_iterator(file_, n=-1):
    with open(file_, 'r') as fin:
        for i, line in enumerate(fin):
            if i == n:
                break
            line = line.strip().split()
            words = []
            for w in line:
                tags = w.split('|')
                if len(tags) == 4 and tags[-1] != 'SPACE':
                    words.append(tags)
            yield words

def build_tfidf_vectorizer(file_, n):
    vectorizer = TfidfVectorizer(analyzer=str.split)
    def doc_iterator(file_, n):
        for words in sentence_iterator(file_, n):
            yield ' '.join([w[1] for w in words])
    vectorizer.fit(doc_iterator(file_, n))
    return vectorizer

def split_sent(words, insert_mode):
    """Generate two input parts.
    """
    N = len(words)
    n = max(0, int(0.3 * N))
    delete_ids = []
    deleted_word = None
    for i, w in enumerate(words):
        if i < n and w[TAG] in ('NOUN', 'PROPN', 'PRON') and not w[TOKEN] in STOP_WORDS:
            delete_ids.append(i)
            deleted_word = w[TOKEN]
            if i > 0: #and not words[i-1][TAG] in ('VERB',):
                delete_ids.insert(i-1, 0)
            if i < N - 1: #and not words[i+1][TAG] in ('VERB', 'NOUN', 'PROPN', 'PRON'):
                delete_ids.append(i+1)
            break

    if not deleted_word:
        return None, None, None

    template = [w[TOKEN] for w in words[:delete_ids[0]]] + \
            ['<placeholder>'] + \
            [w[TOKEN] for w in words[delete_ids[-1]+1:]]

    #if insert_mode == 'none':
    #    insert = None
    #elif insert_mode == 'self':
    #    insert = delete_word
    #elif insert_mode == 'related':
    #    for i, w in enumerate(words[::-1]):
    #        if i < n and w[TAG] in ('NOUN', 'VERB', 'ADJ'):
    #            insert = w[TOKEN]
    #            break
    related_word = '<empty>'
    for i, w in enumerate(words[::-1]):
        if i < n and w[TAG] in ('NOUN', 'VERB', 'ADJ') and not w[TOKEN] in STOP_WORDS:
            related_word = w[TOKEN]
            break

    return template, deleted_word, related_word


def main(args):
    with open(args.parsed_corpus, 'r') as fin, \
         open(args.output+'.src', 'w') as fsrc, \
         open(args.output+'.tgt', 'w') as ftgt:
        for words in sentence_iterator(args.parsed_corpus, args.max_examples):
            template, del_word, rel_word = split_sent(words, args.insert)
            if not template:
                continue
            # target
            sent = ' '.join([w[TOKEN] for w in words])
            ftgt.write('{}\n'.format(sent))
            # source
            fsrc.write('{deleted}\n{related}\n{temp}\n'.format(
                deleted=del_word,
                related=rel_word,
                temp=' '.join(template)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
