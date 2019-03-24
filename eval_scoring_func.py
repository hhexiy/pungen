"""Compute the correlation of funniness scorers given by a scoring function and those
given by humans.
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

from fairseq.data.dictionary import Dictionary

from pungen.scorer import LMScorer, SurprisalScorer, UnigramModel, GoodmanScorer
from pungen.options import add_scorer_args, add_generic_args
from pungen.utils import logging_config, get_spacy_nlp
from pungen.wordvec.generate import SkipGram
from pungen.pretrained_wordvec import Glove

import logging
logger = logging.getLogger('pungen')

nlp = get_spacy_nlp(tokenizer='default', disable=['tagger', 'ner', 'parser'])

def parse_args():
    parser = argparse.ArgumentParser()
    add_scorer_args(parser)
    add_generic_args(parser)
    parser.add_argument('--skipgram-model', nargs=2, help='pretrained skipgram model [vocab, model]')
    parser.add_argument('--skipgram-embed-size', type=int, default=300, help='word embedding size in skipgram model')
    parser.add_argument('--human-eval', help='path to human score file')
    parser.add_argument('--features', nargs='+', default=['ratio', 'grammar', 'ambiguity'], help='features to analyze')
    parser.add_argument('--ignore-cache', action='store_true', help='ignore cached scores. cache path: `args.outdir`/scores.json')
    parser.add_argument('--analysis', action='store_true', help='using analysis data instead of generated data')
    parser.add_argument('--tokenized', action='store_true', help='whether the sentences are tokenized')
    args = parser.parse_args()
    return args


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
            x = np.array([s[x_feature] for s in scores])
            y = np.array([s[y_feature] for s in scores])
            axarr[ind].scatter(x, y)
            axarr[ind].set_xlabel(x_feature)
            axarr[ind].set_ylabel(y_feature)
            ind += 1
    f.subplots_adjust(hspace=0.8)
    plt.savefig('x_human.jpg', format=file_format)

def parse_human_eval_data(path, tokenized):
    candidates = []
    with open(path) as fin:
        for line in fin:
            ss = line.strip().split('\t')
            if tokenized:
                sent = ss[0].split()
            else:
                sent = [x.text for x in nlp(ss[0])]
            sent = [x.lower() for x in sent]
            pun_word, alter_word = ss[1].split('-')
            method = ss[2]
            score = float(ss[3])
            # TODO: move to preprocess
            try:
                pun_word_id = sent.index(pun_word)
            except ValueError:
                continue
            c = {
                    'pun_sent': sent,
                    'pun_word_id': pun_word_id,
                    'alter_word': alter_word,
                    'scores': {'human': score},
                    'type': method,
                }
            candidates.append(c)
    return candidates

def score_examples(args):
    lm = LMScorer.load_model(args.lm_path)
    unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1], embedding_size=args.skipgram_embed_size, cpu=args.cpu)

    candidates = parse_human_eval_data(args.human_eval, args.tokenized)

    #vocab = build_vocab(candidates)
    #glove = Glove.from_file('/u/scr/nlp/data/glove_vecs/glove.840B.300d.txt', vocab)
    glove = None

    scorers = [SurprisalScorer(lm, unigram_model, local_window_size=args.local_window_size),
               GoodmanScorer(unigram_model, skipgram, glove)]
    #scorers = [GoodmanScorer(unigram_model, skipgram, glove)]

    for c in candidates:
        for scorer in scorers:
            scores = scorer.analyze(c['pun_sent'], c['pun_word_id'], c['alter_word'])
            c['scores'].update(scores)
    return candidates

def build_vocab(candidates):
    d = Dictionary()
    for c in candidates:
        for w in c['pun_sent']:
            d.add_symbol(w)
    return d

def compute_stats(candidates):
    results_by_type = defaultdict(lambda : defaultdict(float))
    for c in candidates:
        results_by_type[c['type']]['count'] += 1
        results_by_type[c['type']]['score'] += c['scores']['human']
    logger.info('Stats')
    str_format = '{type:<10s}{count:10d}{score:10.2f}'
    for k, v in results_by_type.items():
        logger.info(str_format.format(type=k, count=int(v['count']), score=v['score'] / v['count']))


def main(args):
    json.dump(vars(args), open(os.path.join(args.outdir, 'config.json'), 'w'))

    filename = os.path.join(args.outdir, 'scores.json')
    if not os.path.exists(filename) or args.ignore_cache:
        candidates = score_examples(args)
        json.dump(candidates, open(filename, 'w'))
    else:
        candidates = json.load(open(filename))

    candidates = [c for c in candidates if c['type'] in ('pun', 'depun', 'retrieved_pw', 'retrieved_aw', 'nonpun')]

    compute_stats(candidates)

    # Normalization, cutoff 2-\sigma
    features = candidates[0]['scores'].keys()
    for feat in features:
        if feat != 'human':
            _scores = zscore([c['scores'][feat] for c in candidates])
            for _score, c in zip(_scores, candidates):
                if _score > 0:
                    _score = min(2, _score)
                elif _score < 0:
                    _score = max(-2, _score)
                c['scores'][feat] = _score

    # Correlation
    if args.analysis:
        all_types = tuple(set([c['type'] for c in candidates]))
        for types in [('pun',), ('pun', 'depun'), ('pun', 'nonpun', 'retrieved_pw', 'retrieved_aw')]:
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
    #plot(scores, x_features=args.features, y_features=['human'])

    # Linear regression
    features = args.features
    scores = [c['scores'] for c in candidates]
    model, r2, feature_stats = linear_regression(scores, features)

    logger.info('linear regression')
    for name, stats in feature_stats.items():
        stats_format = '{name:<15s}' + ' '.join(['{}={{{}:>8.4f}}'.format(s, s) for s in stats.keys()])
        logger.info(stats_format.format(name=name, **stats))
    logger.info('R^2={:.4f}'.format(r2))

if __name__ == '__main__':
    args = parse_args()
    logging_config(os.path.join(args.outdir, 'console.log'), console_level=logging.DEBUG)
    main(args)
