"""Compute the correlation of scorers given by our scoring function and those
given by humans.
"""

import os
import argparse
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import numpy as np
import pickle as pkl
import json
import matplotlib.pyplot as plt

from pungen.scorer import LMScorer, PunScorer, UnigramModel, GoodmanPunScorer
from pungen.options import add_scorer_args, add_generic_args
from pungen.utils import logging_config
from pungen.wordvec.generate import SkipGram

import logging
logger = logging.getLogger('pungen')

def parse_args():
    parser = argparse.ArgumentParser()
    add_scorer_args(parser)
    add_generic_args(parser)
    parser.add_argument('--skipgram-model', nargs=2, help='pretrained skipgram model [vocab, model]')
    parser.add_argument('--skipgram-embed-size', type=int, default=300, help='word embedding size in skipgram model')
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
                logger.error('The sentence pair has different sizes')
                logger.error(s1)
                logger.error(s2)
                raise Exception
            id_ = find_first_diff(s1, s2)
            c1 = {
                    'pun_sent': s1,
                    'pun_word_id': id_,
                    'alter_word': s2[id_],
                    'scores': {'human': f1},
                    'type': t1,
                    }
            c2 = {
                    'pun_sent': s2,
                    'pun_word_id': id_,
                    'alter_word': s1[id_],
                    'scores': {'human': f2},
                    'type': t2,
                    }
            candidates.append(c1)
            candidates.append(c2)
    return candidates


def linear_regression(scores, features, target='human'):
    X = np.array([[s[f] for f in features] for s in scores])
    y = np.array([s[target] for s in scores])
    model = LinearRegression().fit(X, y)
    coeffs = {k: v for k, v in zip(features, model.coef_)}
    r2 = model.score(X, y)
    f_scores, p_values = f_regression(X, y)
    f_scores = {k: {'coeff': c, 'fscore': f, 'pvalue': p}
            for k, c, f, p in zip(features, model.coef_, f_scores, p_values)}
    return model, r2, f_scores

def plot(scores, x_features, y_features, path='./', file_format='jpg'):
    plt.figure(1)
    ind = 0
    nrows = len(x_features)
    ncols = len(y_features)
    f, axarr = plt.subplots(nrows, ncols)
    for i, x_feature in enumerate(x_features):
        for j, y_feature in enumerate(y_features):
            #plt.subplot(nrows, ncols, ind)
            x = np.array([s[x_feature] for s in scores])
            y = np.array([s[y_feature] for s in scores])
            axarr[ind].scatter(x, y)
            axarr[ind].set_xlabel(x_feature)
            axarr[ind].set_ylabel(y_feature)
            #axarr[ind].set_title('{} vs {}'.format(x_feature, y_feature))
            ind += 1
    f.subplots_adjust(hspace=0.8)
    plt.savefig('x_human.jpg', format=file_format)

def score_examples(args):
    lm = LMScorer.load_model(args.lm_path)
    unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1], embedding_size=args.skipgram_embed_size, cpu=args.cpu)

    scorers = [PunScorer(lm, unigram_model, skipgram=skipgram, local_window_size=2),
                GoodmanPunScorer(lm, unigram_model, skipgram)]

    candidates = parse_human_eval_data(args.human_eval)
    for c in candidates:
        for scorer in scorers:
            scores = scorer.score(c['pun_sent'], c['pun_word_id'], c['alter_word'])
            c['scores'].update(scores)
    return candidates


def main(args):
    filename = os.path.join(args.outdir, 'scores.json')
    if not os.path.exists(filename):
        candidates = score_examples(args)
        json.dump(candidates, open(filename, 'w'))
    else:
        candidates = json.load(open(filename))

    # Correlation
    all_types = pair_dict.keys()
    for types in [('pun', 'depun'), ('pun',), all_types]:
        _candidates = [c for c in candidates if c['type'] in types]
        human_scores = [c['scores']['human'] for c in _candidates]
        logger.info('correlation for {} sentences of types {}'.format(len(_candidates), str(types)))
        for metric in _candidates[0]['scores'].keys():
            if metric == 'human':
                continue
            _scores = [c['scores'][metric] for c in _candidates]
            corr = spearmanr(human_scores, _scores)
            logger.info('{:<15s}: {:>8.4f} p={:>8.4f}'.format(metric, corr.correlation, corr.pvalue))

    # All scores
    scores = [c['scores'] for c in candidates]

    # Plot
    plot(scores, x_features=['local', 'global', 'ambiguity', 'grammar'], y_features=['human'])

    # Linear regression
    #features = ['global', 'local', 'ratio', 'grammar', 'ambiguity', 'global_pun_skipgram', 'global_alter_skipgram', 'global_skipgram']
    features = ['global', 'local', 'grammar', 'ambiguity']
    #features = ['grammar', 'local_ambiguity', 'global_ambiguity']
    #features = ['grammar', 'ambiguity']
    #features = ['global_alter', 'global_pun', 'local_alter', 'local_pun', 'grammar']
    model, r2, feature_stats = linear_regression(scores, features)
    pkl.dump(model, open(os.path.join(args.outdir, 'lr_model.pkl'), 'wb'))

    logger.info('linear regression')
    for name, stats in feature_stats.items():
        stats_format = '{name:<15s}' + ' '.join(['{}={{{}:>8.4f}}'.format(s, s) for s in stats.keys()])
        logger.info(stats_format.format(name=name, **stats))
    logger.info('R^2={:.4f}'.format(r2))

if __name__ == '__main__':
    args = parse_args()
    logging_config(os.path.join(args.outdir, 'console.log'), console_level=logging.DEBUG)
    main(args)
