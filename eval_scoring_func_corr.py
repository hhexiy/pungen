"""Compute the correlation of scorers given by our scoring function and those
given by humans.
"""

import argparse
import random
from scipy.stats import spearmanr

from pungen.scorer import LMScorer, PunScorer, UnigramModel, GoodmanPunScorer
from pungen.options import add_scorer_args
from pungen.utils import logging_config
from pungen.wordvec.generate import SkipGram

import logging
logger = logging.getLogger('pungen')

def parse_args():
    parser = argparse.ArgumentParser()
    add_scorer_args(parser)
    parser.add_argument('--skipgram-model', nargs=2, help='pretrained skipgram model [vocab, model]')
    parser.add_argument('--skipgram-embed-size', type=int, default=300, help='word embedding size in skipgram model')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--human-eval')
    args = parser.parse_args()
    return args

def find_first_diff(s1, s2):
    for i, w in enumerate(s1):
        if w != s2[i]:
            return i
    return False

pair_dict = {
        'pun': 'depun',
        'depun': 'pun',
        'retrieved_aw': 'retrieved_aw_alter',
        'retrieved_aw_alter': 'retrieved_aw',
        'retrieved_pw': 'retrieved_pw_alter',
        'retrieved_pw_alter': 'retrieved_pw',
        }

def parse_human_eval_data(path):
    def get_pairs(fin):
        pairs = []
        for line in fin:
            ss = line.strip().split('\t')
            text, type_, score = ss[0].split(), ss[1], float(ss[2])
            if len(pairs) == 0:
                pairs.append((type_, text, score))
            elif type_ != pair_dict[pairs[0][0]] or len(text) != len(pairs[0][1]):
                # NOTE: always discard the previous one
                pairs = [(type_, text, score)]
            else:
                pairs.append((type_, text, score))
            if len(pairs) == 2:
                yield pairs
                pairs = []

    candidates = []
    with open(path, 'r') as fin:
        for pairs in get_pairs(fin):
            t1, s1, f1 = pairs[0]
            t2, s2, f2 = pairs[1]
            if not len(s1) == len(s2):
                print(s1)
                print(s2)
                raise Exception
            id_ = find_first_diff(s1, s2)
            c1 = {
                    'pun_sent': s1,
                    'pun_word_id': id_,
                    'alter_word': s2[id_],
                    'human_score': f1,
                    'type': t1,
                    }
            c2 = {
                    'pun_sent': s2,
                    'pun_word_id': id_,
                    'alter_word': s1[id_],
                    'human_score': f2,
                    'type': t2,
                    }
            candidates.append(c1)
            candidates.append(c2)
    return candidates


def main(args):
    lm = LMScorer.load_model(args.lm_path)
    unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1], embedding_size=args.skipgram_embed_size, cpu=args.cpu)

    scorers = [PunScorer(lm, unigram_model),
                GoodmanPunScorer(lm, unigram_model, skipgram)]

    candidates = parse_human_eval_data(args.human_eval)
    for c in candidates:
        c['scores'] = {}
        for scorer in scorers:
            scores = scorer.score(c['pun_sent'], c['pun_word_id'], c['alter_word'])
            c['scores'].update(scores)

    all_types = pair_dict.keys()
    for types in [('pun', 'depun'), ('pun',), all_types]:
        _candidates = [c for c in candidates if c['type'] in types]
        human_scores = [c['human_score'] for c in _candidates]
        print('correlation for {} sentences of types {}'.format(len(_candidates), str(types)))
        for metric in _candidates[0]['scores'].keys():
            _scores = [c['scores'][metric] for c in _candidates]
            corr = spearmanr(human_scores, _scores)
            print('{:<15s}: {:<8.4f} p={:<8.4f}'.format(metric, corr.correlation, corr.pvalue))

if __name__ == '__main__':
    args = parse_args()
    logging_config(console_level=logging.DEBUG)
    main(args)
