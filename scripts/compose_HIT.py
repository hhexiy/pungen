#!/usr/bin/python

import json
import random
import argparse
import sys, re
from random import shuffle
from wordfreq import word_frequency
from mosestokenizer import *


def compose_collaborative_pun_hit(data_dict, key_filter, outfile, top_k=5):
    with open(outfile, 'w') as outf:
        header = ['Pun_alter']
        for i in range(top_k):
            header.append('sentence_'+str(i+1))
        assert len(header) == top_k + 1
        outf.write(','.join(header)+'\n')
        for key in key_filter:
            results = data_dict[key]
            if word_frequency(key[0], 'en') < 1e-6 or word_frequency(key[1], 'en') < 1e-6 :
                print('skip the keyword pair:', ' '.join(key))
                continue
            contents = []
            contents.append('-'.join(key))
            if type(results) is tuple:
                results = results[0]
            for res in results[:top_k]:
                contents.append(res)
            #print(type(contents), contents)
            outf.write(','.join(contents)+'\n')


def compose_funniness_justin_data(dataset, outfile):
    with open(outfile, 'w') as outf:
        header = []
        for i in range(10):
            header.append('sentence_'+str(i+1))
            header.append('sentence_info_'+str(i+1))
        assert len(header) == 20
        outf.write(','.join(header)+'\n')
        for i, pair in enumerate(dataset):
            print(pair)
            outf.write(','.join(pair))
            if (i+1)%10 == 0:
                outf.write('\n')
            else:
                outf.write(',')


def compose_funniness_anno_hit(names, outfile, *datasets):
    #names = ['pun', 'depun', 'retrieved_pw', 'retrieved_aw']
    indexes = list(range(len(names)))
    temp = datasets
    with open(outfile, 'w') as outf:
        header = []
        for i in range(10):
            header.append('sentence_'+str(i+1))
            header.append('sentence_info_'+str(i+1))
        assert len(header) == 20
        outf.write(','.join(header)+'\n')
        contents = []
        for i, elems in enumerate(zip(*datasets)): #zip(puns, depuns, retrieve_punwords, retrieve_alters)):
            #print(type(elems), len(elems))
            sentences = elems #[p, dp, rp, ra]
            shuffle(indexes)
            contents.extend([(sentences[ii], names[ii]+'_'+str(i)) for ii in indexes])
        for i, pair in enumerate(contents):
            outf.write(','.join(pair))
            if (i+1)%10 == 0:
                outf.write('\n')
            else:
                outf.write(',')

def compose_eval_hit(sentences, pun_word, outfile, group_per_page=2):
    indexes = list(range(len(sentences)))
    instance_order = list(range(len(pun_word)))
    shuffle(instance_order)
    with open(outfile, 'w') as outf:
        header = []
        for i in range(group_per_page):
            header.append('order_info_'+str(i+1))
            header.append('Pun_word_'+str(i+1))
            header.append('Pun_alter_'+str(i+1))
            for j in range(len(indexes)):
                header.append('Sentence'+str(i+1)+'_'+str(j+1))
        assert len(header) == (len(indexes)+3)*group_per_page, len(header)
        outf.write(','.join(header)+'\n')
        zipped_sentences = list(zip(*sentences))
        #print(len(sentences), len(zipped_sentences))
        count = 0
        for i in instance_order:
            sents = zipped_sentences[i]
            assert len(sents) == len(indexes), len(sents)
            shuffle(indexes)
            sents = [sents[ii] for ii in indexes] 
            outf.write('-'.join(list(map(str, indexes))) + ',')
            outf.write(pun_word[i][0]+',')
            outf.write('-'.join(pun_word[i])+',')
            outf.write(','.join(sents))
            if (count+1)%group_per_page == 0:
                outf.write('\n')
            else:
                outf.write(',')
            count += 1
            #print(count)

def compose_eval_human_hit(outfile, group_per_page=2, *data_dicts):
    indexes = list(range(len(data_dicts)))
    with open(outfile, 'w') as outf:
        header = []
        for i in range(group_per_page):
            header.append('order_info_'+str(i+1))
            header.append('Pun_alter_'+str(i+1))
            for j in range(len(indexes)):
                header.append('Sentence'+str(i+1)+'_'+str(j+1))
                header.append('TurkerID'+str(i+1)+'_'+str(j+1))
        assert len(header) == (2*len(indexes)+2)*group_per_page, len(header)
        outf.write(','.join(header)+'\n')
        ds0 = data_dicts[0]
        count = 0
        for k, v in ds0.items():
            check = [k in dic for dic in data_dicts]
            if sum(check) < len(data_dicts):
                print('some dataset cannot generate words:', k)
                continue
            sents = [ddict[k] for ddict in data_dicts] 
            assert len(sents) == len(indexes), len(sents)
            shuffle(indexes)
            sents = [','.join(sents[ii]).lower() for ii in indexes] 
            outf.write('-'.join(list(map(str, indexes))) + ',')
            outf.write(k+',')
            outf.write(','.join(sents))
            if (count+1)%group_per_page == 0:
                outf.write('\n')
            else:
                outf.write(',')
            count += 1

def load_sentences(infile):
    contents = []
    with open(infile) as inf:
        for line in inf:
            contents.append('\"'+re.sub('\"', '\'\'', ' '.join(line.strip().split('\t')))+'\"')
    return contents

def load_justin(infile):
    contents = []
    with open(infile) as inf:
        for line in inf:
            contents.append(('\"'+re.sub('\"', '\'\'', line.strip().split('\t')[0])+'\"', '\t'.join(line.strip().split('\t')[1:])))
    return contents


def load_human(infile):
    data_dict = dict()
    print('loading from', infile)
    with open(infile) as inf, MosesDetokenizer('en') as detokenize:
        for line in inf:
            elems = line.strip().split('\t')
            assert (len(elems) == 2 or len(elems) == 5), len(elems)
            key = elems[0]
            sent = '"' + re.sub('\"', '\'\'', detokenize(elems[1].split())) + '"'
            if len(elems) == 5:
                turker = elems[-1]
            else:
                turker = 'placeholder'
            data_dict[key] = (sent, turker)
    return data_dict


def load_json(infile, top_k=1):
    # data_dict: key=(pun_word, alter_word), value=(top_k_results, gold_sentence)
    data_dict = dict()
    with open(infile) as inf, MosesDetokenizer('en') as detokenize:
        data = json.load(inf)
        for line in data:
            if len(line.get('results', [])) == 0:
                continue
            pw = line['pun_word']
            aw = line['alter_word']
            ref = ' '.join(line['tokens'])
            #results = [(' '.join(item.get('output', [])), item.get('score', float("inf"))) for item in line['results']]
            results = [(detokenize(lctx)+' # '+' # '.join(random.sample(item.get('topic_words', []), top_k//2)), random.random()) for item in line['results'] for lctx in item.get('local_contexts', [])] # + ' '.join([:5])
            #results = [(detokenize(item.get('output', [])), item.get('score', float("inf"))) for item in line['results']]
            results = sorted(results, key=lambda x: x[1])[:top_k]
            results = ['"' + '\''.join(res[0].split('"'))+'"' for res in results]
            key = (pw, aw)
            try:
                assert key not in data_dict 
            except:
                sys.stderr.write(str(key) + 'has already in the dictionary!\n')
            data_dict[key] = (results, ref)
    return data_dict


def load_pku(kw_file, sent_file, every=100, top_k=1):
    # data_dict: key=(pun_word, alter_word), value=top_k_results
    data_dict = dict()
    key_array = load_keyword(kw_file)
    print('key array size:', len(key_array))
    results_array = []
    with open(sent_file) as sf, MosesDetokenizer('en') as detokenize:
        local = []
        for line in sf:
            elems = line.strip().split()
            #sent = ' '.join(elems[:-1])
            sent = detokenize(elems[:-1])
            score = random.random()
            local.append((sent, score))
            if len(local) == every:
                results_array.append(local[:(every//2)])
                local = []
    assert len(key_array) == len(results_array)
    for key, results in zip(key_array, results_array):
        '''if word_frequency(key[0], 'en') < 1e-6 or word_frequency(key[1], 'en') < 1e-6 :
            print('skip the keyword pair:', ' '.join(key))
            continue
        '''
        results = sorted(results, key=lambda x: x[1])[:top_k]
        results = ['"' + '\''.join(res[0].split('"'))+'"' for res in results]
        try:
            assert key not in data_dict
        except:
            sys.stderr.write(str(key) + '\n')
        data_dict[key] = results
    return data_dict


def load_keyword(kw_file):
    key_array = []
    with open(kw_file) as kf:
        keys = []
        for line in kf:
            keys.append(line.strip())
            if len(keys) == 2:
                key_array.append(tuple(keys))
                keys = []
    return key_array
            

def combine_results(pku_dict, top_k=1, *other_dicts):
    pun_words = []
    sentences = [[] for i in range(len(other_dicts) + 2)]
    count = 0 
    reference = None
    for key, val in pku_dict.items():
        check = [key in dic for dic in other_dicts]
        if sum(check) < len(other_dicts):
            print('some dataset cannot generate words:', key)
            continue
        for i, dic in enumerate(other_dicts):
            #if key not in dic:
            #    #print(key)
            #    count += 1
            #    break
            generated = dic[key][0]
            try:
                assert len(generated) == top_k, len(generated)
            except:
                generated += [generated[-1]] * (top_k-len(generated))
            sentences[i+1].extend(generated)
            if i == 0:
                reference = dic[key][1]
        #if i < len(other_dicts)-1 or key not in dic:
        #    # roll back
        #    continue
        pun_words.extend([key] * len(val))
        sentences[0].extend(val)
        sentences[-1].extend(['"' + '\''.join(reference.split('"'))+'"'] * len(val))
        #print(len(pun_words), [len(sents) for sents in sentences])
    print(len(pun_words), len(sentences), [len(sent) for sent in sentences])
    return pun_words, sentences


def load_key_filter(fkeyfilter, top=20):
    data_array = []
    with open(fkeyfilter) as inf:
        for line in inf:
            elems = line.strip().split('\t')
            data_array.append(elems[0])
    return [tuple(key.split('-')) for key in data_array[:top]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='compose_eval_hit', help='which hit to compose')
    parser.add_argument('--files', nargs='+', help='the input files. can take multiple files.')
    parser.add_argument('--top-k', default=1, type=int, help='get top k from each method')
    parser.add_argument('--keywords', help='file containing keywords')
    parser.add_argument('--fkeyfilter', help='file containing filtered keywords')
    parser.add_argument('--outfile', default='test.csv', help='output file for the retrieved sentences')
    args = parser.parse_args()
    func = eval(args.task)
   
    # quick hack to get keywords from He's file
    '''data_dict = load_json(args.files[0])
    for k, v in data_dict.items():
        print(k[0])
        print(k[1])
    exit(0)
    '''
    if args.task == 'compose_eval_human_hit':
        names = ['turker', 'turker-pku', 'turker-surprisal', 'expert']
        data_array = [load_human(infile) for infile in args.files]
        compose_eval_human_hit(args.outfile, 2, *data_array)

    if args.task == 'compose_collaborative_pun_hit':
        filename = args.files[0]
        if 'final' in filename:
            data_dict = load_pku(args.keywords, filename, top_k=args.top_k)
        else:
            data_dict = load_json(filename, top_k=args.top_k)
        key_filter = load_key_filter(args.fkeyfilter)
        print(len(key_filter))
        func(data_dict, key_filter, args.outfile, args.top_k)

    if args.task == 'compose_eval_hit':
        names = ['pku', 'retrieved', 'retrieve_alter', 'rule', 'neural', 'human']
        retrieve = load_json(args.files[0], top_k=args.top_k)
        retrieve_repl = load_json(args.files[1], top_k=args.top_k)
        rule = load_json(args.files[2], top_k=args.top_k)
        neural = load_json(args.files[3], top_k=args.top_k)
        pku = load_pku(args.keywords, args.files[4], top_k=args.top_k)
        pun_words, sentences = combine_results(pku, args.top_k, retrieve, retrieve_repl, rule, neural)
        func(sentences, pun_words, args.outfile)
    
    if args.task == 'compose_funniness_anno_hit':
        sentences_array = [load_sentences(fname) for fname in args.files] 
        # the actual order: incremental, title-to-text, title-keywords-text, human_story
        names = ['pun', 'depun', 'retrieved_pw', 'retrieved_pw_alter', 'retrieved_aw', 'retrieved_aw_alter']
        func(names, args.outfile, *sentences_array) 

    if args.task == 'compose_funniness_justin_data':
        sentences = load_justin(args.files[0])
        func(sentences, args.outfile)
