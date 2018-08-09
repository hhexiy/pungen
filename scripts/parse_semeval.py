from argparse import ArgumentParser
from collections import defaultdict
from unidecode import unidecode
import xml.etree.ElementTree
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
is_stop = lambda x: x in STOP_WORDS

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--xml')
    parser.add_argument('--output')
    args = parser.parse_args()

    e = xml.etree.ElementTree.parse(args.xml).getroot()
    sents = defaultdict(list)
    pun_words = defaultdict(list)
    for w in e.iter('word'):
        token = unidecode(w.text).lower()
        pun_word = w.attrib['senses'] == '2'
        sent_id = int(w.attrib['id'].split('_')[1])
        sents[sent_id].append(token)
        if pun_word:
            pun_words[sent_id].append(token)

    with open(args.output, 'w') as fout:
        for id_ in sents:
            text = ' '.join(sents[id_])
            #doc = nlp(text)
            #content_words = set([x.text.lower() for x in doc if x.pos_ in ('NOUN', 'VERB') and not is_stop(x.text.lower()) and len(x.text) > 2])
            #pun_word = pun_words[id_]
            #for w in pun_word:
            #    if not w in content_words:
            #        content_words.add(w)
            #fout.write(' '.join(pun_word) + '\t' + ' '.join(list(content_words)) + '\t' + text + '\n')
            fout.write(text + '\n')

