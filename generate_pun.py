import pickle
import argparse
import json
from collections import defaultdict

from fairseq import options

from pungen.retriever import Retriever
from pungen.generator import SkipGram, RulebasedGenerator, NeuralCombinerGenerator
from pungen.scorer import LMScorer, PunScorer, UnigramModel
from pungen.type import TypeRecognizer
from pungen.options import add_scorer_args, add_editor_args, add_retriever_args


def parse_args():
    parser = options.get_generation_parser(interactive=True)
    add_scorer_args(parser)
    add_editor_args(parser)
    add_retriever_args(parser)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--pun-words')
    parser.add_argument('--output', default='output.txt')
    parser.add_argument('--system', default='rule')
    parser.add_argument('--num-workers', type=int, default=1)
    args = options.parse_args_and_arch(parser)
    return args

def iter_keywords(file_):
    with open(file_, 'r') as fin:
        for line in fin:
            alter_word, pun_word = line.strip().split()
            yield alter_word, pun_word

def main(args):
    retriever = Retriever(args.doc_file, path=args.retriever_model, overwrite=args.overwrite_retriever_model)
    skipgram = SkipGram.load_model(args.skipgram_model[0], args.skipgram_model[1], embedding_size=args.skipgram_embed_size, cpu=args.cpu)
    lm = LMScorer.load_model(args.lm_path)
    #lm = None
    unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    scorer = PunScorer(lm, unigram_model)
    type_recognizer = TypeRecognizer(args.type_dict_path)
    if args.system == 'rule':
        generator = RulebasedGenerator(retriever, skipgram, type_recognizer, scorer)
    else:
        generator = NeuralCombinerGenerator(retriever, skipgram, type_recognizer, scorer, args)

    puns = json.load(open(args.pun_words))
    for example in puns:
        alter_word, pun_word = example['pun_word'], example['alter_word']
        print('INPUT:', alter_word, pun_word)
        results = generator.generate(alter_word, pun_word, k=100, ncands=500)
        if not results:
            continue
        results = [r for r in results if r.get('score') is not None]
        # group by template
        result_groups = defaultdict(list)
        for r in results:
            result_groups[r['template-id']].append(r)
        sorted_group_results = {}
        for id_, results_ in result_groups.items():
            results_ = sorted(results_,
                    key=lambda x: x['score'], reverse=True)[:3]
            sorted_group_results[id_] = results_
        sorted_groups = sorted(sorted_group_results.values(), key=lambda x: sum([x_['score'] for x_ in x]), reverse=True)
        for results in sorted_groups:
            print('RETRIEVED:', ' '.join(results[0]['retrieved']))
            print(' '.join(results[0]['retrieved-ori']))
            for r in results:
                print('{} -> {}'.format(r['deleted'], r['inserted']))
                print(r['score'], ' '.join(r['output']))
            print()
    #fout.close()

    # Cache queried types
    type_recognizer.save()



if __name__ == '__main__':
    args = parse_args()
    main(args)
