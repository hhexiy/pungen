import os
import pickle
import argparse
import json
from collections import defaultdict
import fuzzy

from fairseq import options

from pungen.retriever import Retriever
from pungen.generator import SkipGram, RulebasedGenerator, NeuralCombinerGenerator, RetrieveGenerator, RetrieveSwapGenerator, KeywordsGenerator
from pungen.scorer import LMScorer, SurprisalScorer, UnigramModel, RandomScorer, GoodmanScorer
from pungen.type import TypeRecognizer
from pungen.options import add_scorer_args, add_editor_args, add_retriever_args, add_generic_args, add_type_checker_args
from pungen.utils import logging_config, get_lemma, ensure_exist, get_spacy_nlp

import logging
logger = logging.getLogger('pungen')

nlp = get_spacy_nlp()

def parse_args():
    parser = options.get_generation_parser(interactive=True)
    add_scorer_args(parser)
    add_editor_args(parser)
    add_retriever_args(parser)
    add_type_checker_args(parser)
    add_generic_args(parser)
    parser.add_argument('--pun-words')
    parser.add_argument('--system', default='rule')
    parser.add_argument('--max-num-examples', type=int, default=-1)
    args = options.parse_args_and_arch(parser)
    return args

def iter_keywords(file_):
    with open(file_, 'r') as fin:
        for line in fin:
            alter_word, pun_word = line.strip().split()
            yield alter_word, pun_word

def feasible_pun_words(pun_word, alter_word, unigram_model, skipgram=None, freq_threshold=1000):
    # Pun / alternative word cannot be phrases
    if len(alter_word.split('_')) > 1 or len(pun_word.split('_')) > 1:
        logger.info('FAIL: phrase')
        return False, 'phrase'

    if skipgram and skipgram.vocab.index(get_lemma(pun_word)) == skipgram.vocab.unk():
        logger.info('FAIL: unknown pun word: {}'.format(pun_word))
        return False, 'unk to skipgram'

    return True, None


def main(args):
    ensure_exist(args.outdir, is_dir=True)
    json.dump(vars(args), open(os.path.join(args.outdir, 'config.json'), 'w'))

    unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    retriever = Retriever(args.doc_file, path=args.retriever_model, overwrite=args.overwrite_retriever_model)

    if args.system.startswith('rule') or args.system == 'keywords' or args.scorer in ('goodman',):
        skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1], embedding_size=args.skipgram_embed_size, cpu=args.cpu)
    else:
        skipgram = None

    if args.scorer == 'random':
        scorer = RandomScorer()
    elif args.scorer == 'surprisal':
        lm = LMScorer.load_model(args.lm_path)
        scorer = SurprisalScorer(lm, unigram_model, local_window_size=args.local_window_size)
    elif args.scorer == 'goodman':
        scorer = GoodmanScorer(unigram_model, skipgram)

    type_recognizer = TypeRecognizer(threshold=args.type_consistency_threshold)

    if args.system == 'rule':
        generator = RulebasedGenerator(retriever, skipgram, type_recognizer, scorer, dist_to_pun=args.distance_to_pun_word)
    elif args.system == 'rule+neural':
        generator = NeuralCombinerGenerator(retriever, skipgram, type_recognizer, scorer, args.distance_to_pun_word, args)
    elif args.system == 'retrieve':
        generator = RetrieveGenerator(retriever, scorer)
    elif args.system == 'retrieve+swap':
        generator = RetrieveSwapGenerator(retriever, scorer)

    puns = json.load(open(args.pun_words))
    # Uniq
    d = {}
    for e in puns:
        d[e['pun_word']] = e
    puns = d.values()
    # Sorting by quality of pun words
    dmeta = fuzzy.DMetaphone()
    homophone = lambda x, y: float(dmeta(x)[0] == dmeta(y)[0])
    length = lambda x, y: float(len(x) > 2 and len(y) > 2)
    freq = lambda x, y: unigram_model.word_counts.get(x, 0) * unigram_model.word_counts.get(y, 0)
    puns = sorted(puns, key=lambda e: (length(e['pun_word'], e['alter_word']),
                                       homophone(e['pun_word'], e['alter_word']),
                                       freq(e['pun_word'], e['alter_word'])),
                  reverse=True)
    num_success = 0
    processed_examples = []
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

        results = generator.generate(alter_word, pun_word, k=args.num_topic_words, ncands=args.num_candidates, ntemps=args.num_templates)
        example['results'] = results
        if not results:
            continue

        results = [r for r in results if r.get('score') is not None]
        results = sorted(results, key=lambda r: r['score'], reverse=True)
        for r in results[:3]:
            logger.info('{:<8.2f}{}'.format(r['score'], ' '.join(r['output'])))

        processed_examples.append(example)
        num_success += 1
        if args.max_num_examples > 0 and num_success >= args.max_num_examples:
            break

    json.dump(processed_examples, open(os.path.join(args.outdir, 'results.json'), 'w'))


if __name__ == '__main__':
    args = parse_args()
    logging_config(os.path.join(args.outdir, 'generate_pun.log'), console_level=logging.INFO)
    main(args)
