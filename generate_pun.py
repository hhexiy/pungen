import os
import pickle
import argparse
import json
from collections import defaultdict

from fairseq import options

from pungen.retriever import Retriever
from pungen.generator import SkipGram, RulebasedGenerator, NeuralCombinerGenerator, RetrieveGenerator, RetrieveSwapGenerator
from pungen.scorer import LMScorer, PunScorer, UnigramModel, RandomScorer
from pungen.type import TypeRecognizer
from pungen.options import add_scorer_args, add_editor_args, add_retriever_args, add_generic_args
from pungen.utils import logging_config, get_lemma, ensure_exist, get_spacy_nlp

import logging
logger = logging.getLogger('pungen')

nlp = get_spacy_nlp()

def parse_args():
    parser = options.get_generation_parser(interactive=True)
    add_scorer_args(parser)
    add_editor_args(parser)
    add_retriever_args(parser)
    add_generic_args(parser)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--pun-words')
    parser.add_argument('--system', default='rule')
    parser.add_argument('--num-workers', type=int, default=1)
    args = options.parse_args_and_arch(parser)
    return args

def iter_keywords(file_):
    with open(file_, 'r') as fin:
        for line in fin:
            alter_word, pun_word = line.strip().split()
            yield alter_word, pun_word

def feasible_pun_words(pun_word, alter_word, unigram_model, skipgram=None, freq_threshold=1000):
    if len(alter_word.split('_')) > 1 or len(pun_word.split('_')) > 1:
        logger.info('FAIL: phrase')
        return False, 'phrase'

    #if nlp(pun_word)[0].pos_[0] != nlp(alter_word)[0].pos_[0]:
    #    logger.info('FAIL: different POS tags')
    #    return False, 'pos tag'

    if unigram_model.word_counts.get(pun_word, 0) < freq_threshold or \
        unigram_model.word_counts.get(alter_word, 0) < freq_threshold:
        logger.info('FAIL: rare words')
        return False, 'rare'

    if skipgram and skipgram.vocab.index(get_lemma(pun_word)) == skipgram.vocab.unk():
        logger.info('FAIL: unknown pun word: {}'.format(pun_word))
        return False, 'unk to skipgram'

    return True, None


def main(args):
    ensure_exist(args.outdir, is_dir=True)
    json.dump(vars(args), open(os.path.join(args.outdir, 'config.json'), 'w'))

    unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    retriever = Retriever(args.doc_file, path=args.retriever_model, overwrite=args.overwrite_retriever_model)

    if args.system.startswith('rule'):
        skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1], embedding_size=args.skipgram_embed_size, cpu=args.cpu)
    else:
        skipgram = None

    if args.scorer == 'random':
        scorer = RandomScorer()
    elif args.scroer == 'surprisal':
        lm = LMScorer.load_model(args.lm_path)
        scorer = PunScorer(lm, unigram_model)

    type_recognizer = TypeRecognizer()

    if args.system == 'rule':
        generator = RulebasedGenerator(retriever, skipgram, type_recognizer, scorer, dist_to_pun=args.distance_to_pun_word)
    elif args.system == 'rule+neural':
        generator = NeuralCombinerGenerator(retriever, skipgram, type_recognizer, scorer, args.distance_to_pun_word, args)
    elif args.system == 'retrieve':
        generator = RetrieveGenerator(retriever, scorer)
    elif args.system == 'retrieve+swap':
        generator = RetrieveSwapGenerator(retriever, scorer)

    puns = json.load(open(args.pun_words))
    for example in puns:
        pun_word, alter_word = example['pun_word'], example['alter_word']
        logger.info('-'*50)
        logger.info('INPUT: alter={} pun={}'.format(alter_word, pun_word))
        logger.info('REFERENCE: {}'.format(' '.join(example['tokens'])))
        logger.info('-'*50)

        feasible, reason = feasible_pun_words(pun_word, alter_word, unigram_model, skipgram=skipgram, freq_threshold=args.pun_freq_threshold)
        if not feasible:
            example['fail'] = reason
            continue

        results = generator.generate(alter_word, pun_word, k=args.num_topic_words, ncands=args.num_candidates, ntemps=args.num_templates, pos_th=args.pos_threshold)
        example['results'] = results
        if not results:
            continue

        results = [r for r in results if r.get('score') is not None]

        # group by template
        if args.system.startswith('rule'):
            result_groups = defaultdict(list)
            for r in results:
                result_groups[r['template-id']].append(r)

            sorted_group_results = {}
            # sort within a template
            for id_, results_ in result_groups.items():
                results_ = sorted(results_,
                        key=lambda x: x['score'], reverse=True)[:3]
                sorted_group_results[id_] = results_
            # sort across templates
            sorted_groups = sorted(sorted_group_results.values(), key=lambda x: sum([x_['score'] for x_ in x]), reverse=True)

            for results in sorted_groups:
                logger.debug('RETRIEVED: {}'.format(results[0]['retrieved']))
                for r in results:
                    logger.debug('{} -> {}'.format(r['deleted'], r['inserted']))
                    logger.debug('{:.2f} {}'.format(r['score'], ' '.join(r['output'])))
                logger.debug(' ')

    json.dump(puns, open(os.path.join(args.outdir, 'results.json'), 'w'))


if __name__ == '__main__':
    args = parse_args()
    logging_config(os.path.join(args.outdir, 'generate_pun.log'), console_level=logging.DEBUG)
    main(args)
