import torch
import pickle
import argparse
from collections import defaultdict

from fairseq import options

from pungen.retriever import Retriever
from pungen.generator import SkipGram, RulebasedGenerator, NeuralGenerator
from pungen.scorer import LMScorer, PunScorer, UnigramModel
from pungen.options import add_scorer_args


def parse_args():
    parser = options.get_generation_parser(interactive=True)
    add_scorer_args(parser)
    parser.add_argument('--insert')
    parser.add_argument('--skipgram-path', nargs=2, help='pretrained skipgram model [vocab, model]')
    parser.add_argument('--doc-file', nargs=2, help='tokenized training corpus to retrieve from')
    parser.add_argument('--retriever-path', help='retriever model path')
    parser.add_argument('--keywords', help='file containing keywords')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--output', default='output.txt')
    parser.add_argument('--system', default='rule')
    args = options.parse_args_and_arch(parser)
    return args

def iter_keywords(file_):
    with open(file_, 'r') as fin:
        for line in fin:
            alter_word, pun_word = line.strip().split()
            yield alter_word, pun_word

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    retriever = Retriever(args.doc_file, args.retriever_path)
    skipgram = SkipGram.load_model(args.skipgram_path[0], args.skipgram_path[1], args.cpu)
    lm = LMScorer.load_model(args.lm_path)
    #lm = None
    unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    scorer = PunScorer(lm, unigram_model)
    if args.system == 'rule':
        generator = RulebasedGenerator(retriever, skipgram, scorer)
    else:
        generator = NeuralGenerator(retriever, skipgram, scorer, args)

    #fout = open(args.output, 'w')
    for alter_word, pun_word in iter_keywords(args.keywords):
        print('INPUT:', alter_word, pun_word)
        results = generator.generate(alter_word, pun_word, k=50, ncands=500)
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



if __name__ == '__main__':
    args = parse_args()
    main(args)
