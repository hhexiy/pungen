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
    parser.add_argument('--types', nargs='+')
    args = parser.parse_args()
    return args

def find_first_diff(s1, s2):
    for i, w in enumerate(s1):
        if w != s2[i]:
            return i
    return False

def parse_human_eval_data(path, types):
    def get_pairs(fin):
        pairs = []
        for line in fin:
            ss = line.strip().split('\t')
            text, type_, score = ss
            if type_ not in types:
                continue
            pairs.append((type_, text.split(), float(score)))
            if len(pairs) == 2:
                yield pairs
                pairs = []

    candidates = []
    with open(path, 'r') as fin:
        for pairs in get_pairs(fin):
            t1, pun_sent, pun_score = pairs[0]
            t2, depun_sent, depun_score = pairs[1]
            if not len(pun_sent) == len(depun_sent):
                print(pun_sent)
                print(depun_sent)
            assert len(pun_sent) == len(depun_sent)
            id_ = find_first_diff(pun_sent, depun_sent)
            c1 = {
                    'pun_sent': pun_sent,
                    'pun_word_id': id_,
                    'alter_word': depun_sent[id_],
                    'human_score': pun_score,
                    'type': t1,
                    }
            c2 = {
                    'pun_sent': depun_sent,
                    'pun_word_id': id_,
                    'alter_word': pun_sent[id_],
                    'human_score': depun_score,
                    'type': t2,
                    }
            candidates.append(c1)
            candidates.append(c2)
    return candidates


def main(args):
    lm = LMScorer.load_model(args.lm_path)
    unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1], embedding_size=args.skipgram_embed_size, cpu=args.cpu)

    scorer = PunScorer(lm, unigram_model)
    goodman_scorer = GoodmanPunScorer(lm, unigram_model, skipgram)

    candidates = parse_human_eval_data(args.human_eval, args.types)
    for c in candidates:
        c['model_score'] = scorer.score(c['pun_sent'], c['pun_word_id'], c['alter_word'])
        c['goodman_model_score'] = goodman_scorer.score(c['pun_sent'], c['pun_word_id'], c['alter_word'])
        #c['model_score'] = random.random()
    print('correlation for {} sentences'.format(len(candidates)))
    human_scores = [c['human_score'] for c in candidates]
    model_scores = [c['model_score'] for c in candidates]
    goodman_model_scores = [c['goodman_model_score'] for c in candidates]
    corr = spearmanr(human_scores, model_scores)
    print(corr)
    corr = spearmanr(human_scores, goodman_model_scores)
    print(corr)

if __name__ == '__main__':
    args = parse_args()
    logging_config(console_level=logging.DEBUG)
    main(args)
