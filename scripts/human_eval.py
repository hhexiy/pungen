import argparse
from collections import defaultdict, Counter
import random
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-outputs')
    parser.add_argument('--num', default=10, type=int)
    parser.add_argument('--db', default='eval.json')
    parser.add_argument('--readable-result', default='eval.txt')
    args = parser.parse_args()
    return args

def parse_results(file_):
    results = defaultdict(lambda : {})
    with open(file_, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line.startswith('S-'):
                s = line[2:].split('\t')
                id_ = int(s[0])
                results[id_]['src'] = s[1]
            elif line.startswith('T-'):
                s = line[2:].split('\t')
                id_ = int(s[0])
                results[id_]['tgt'] = s[1]
            elif line.startswith('H-'):
                s = line[2:].split('\t')
                id_ = int(s[0])
                results[id_]['hypo'] = s[1]
    return results

def main(args):
    if os.path.exists(args.db):
        with open(args.db, 'r') as fin:
            results = json.load(fin)
    else:
        results = parse_results(args.model_outputs)

    category_dict = {
        1: 'wrong sentiment',
        2: 'wrong event sequences / contradiction',
        3: 'repetition',
        4: 'irrelevence',
        5: 'ilogical sentence',
    }

    ids = list(results.keys())
    eval_sample_ids = random.choices(ids, k=args.num)
    for id_ in eval_sample_ids:
        sample = results[id_]
        print(sample['src'])
        print(sample['hypo'])
        print('1', 'wrong sentiment')
        print('2', 'wrong event sequences / contradiction')
        print('3', 'repetition')
        print('4', 'irrelevence')
        print('5', 'ilogical sentence')
        selections = input('Please select: ')
        selections = [int(x) for x in selections.split()]
        if 'problems' not in sample:
            sample['problems'] = selections
        else:
            for s in selections:
                if not s in sample['problems']:
                    sample['problems'].append(s)

    num_eval = 0
    for id_, result in results.items():
        if 'problems' in result:
            num_eval += 1

    with open(args.readable_result, 'w') as fout:
        for id_, result in results.items():
            if 'problems' in result:
                fout.write(result['src']+'\n')
                fout.write(result['hypo']+'\n')
                for p in result['problems']:
                    fout.write('{} {}\n'.format(p, category_dict[p]))
                fout.write('\n')


    all_problems = Counter()
    for id_, result in results.items():
        all_problems.update(result.get('problems', []))
    print(num_eval)
    print(all_problems)

    with open(args.db, 'w') as fout:
        json.dump(results, fout)

if __name__ == '__main__':
    args = parse_args()
    main(args)
