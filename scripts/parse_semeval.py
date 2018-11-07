import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from unidecode import unidecode
import xml.etree.ElementTree

from pungen.utils import ensure_exist


def parse_gold_file(path):
    word_pairs = {}
    with open(path) as fin:
        for line in fin:
            ss = line.strip().split()
            sent_id = '_'.join(ss[0].split('_')[:2])
            word_id = int(ss[0].split('_')[-1])
            pun_word = ss[1].split('%')[0]
            alter_word = ss[2].split('%')[0]
            word_pairs[sent_id] = {'pun': pun_word, 'alter': alter_word, 'id': word_id}
    return word_pairs

def parse_xml_file(path):
    e = xml.etree.ElementTree.parse(path).getroot()
    sents = defaultdict(list)
    for w in e.iter('word'):
        token = unidecode(w.text).lower()
        sent_id = '_'.join(w.attrib['id'].split('_')[:2])
        sents[sent_id].append(token)
    return sents

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--xml', help='.xml input from semeval containing pun sentences')
    parser.add_argument('--gold', help='.gold input from semeval containing paired words')
    parser.add_argument('--output-dir', help='path to output file')
    parser.add_argument('--split', default=0.2, help='percentage of data for development')
    args = parser.parse_args()

    word_pairs = parse_gold_file(args.gold)
    sents = parse_xml_file(args.xml)
    ids = sorted(word_pairs.keys(), key=lambda x: int(x.split('_')[1]))
    puns = []
    for id_ in ids:
        # NOTE: take pun word from the sentence directly because the one in the gold file is lemma
        pun = {
                'id': id_,
                'tokens': sents[id_],
                'pun_word': sents[id_][word_pairs[id_]['id']-1],
                'alter_word': word_pairs[id_]['alter'],
                'pun_word_id': word_pairs[id_]['id']-1,
                }
        puns.append(pun)

    ensure_exist(args.output_dir, is_dir=True)
    ndev = int(len(puns) * args.split)

    path = os.path.join(args.output_dir, 'dev.json')
    print('Write {} puns for dev to {}.'.format(ndev, path))
    json.dump(puns[:ndev], open(path, 'w'))

    path = os.path.join(args.output_dir, 'test.json')
    print('Write {} puns for dev to {}.'.format(len(puns) - ndev, path))
    json.dump(puns[ndev:], open(path, 'w'))
