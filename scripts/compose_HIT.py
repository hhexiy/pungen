#!/usr/bin/python

import json
import random
import sys, re
from random import shuffle
from wordfreq import word_frequency
from mosestokenizer import *

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
            for j in range(6):
                header.append('Sentence'+str(i+1)+'_'+str(j+1))
        assert len(header) == 9*group_per_page, len(header)
        outf.write(','.join(header)+'\n')
        zipped_sentences = list(zip(*sentences))
        print(len(sentences), len(zipped_sentences))
        count = 0
        for i in instance_order:
            sents = zipped_sentences[i]
            assert len(sents) == 6, len(sents)
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
            print(count)


def load_sentences(infile):
    contents = []
    with open(infile) as inf:
        for line in inf:
            contents.append('\"'+re.sub('\"', '\'\'', ' '.join(line.strip().split('\t')))+'\"')
    return contents


def load_json(infile, top_k=1):
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
            results = [(detokenize(item.get('output', [])), item.get('score', float("inf"))) for item in line['results']]
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
        if word_frequency(key[0], 'en') < 1e-6 or word_frequency(key[1], 'en') < 1e-6 :
            print('skip the keyword pair:', ' '.join(key))
            continue
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
        print(len(pun_words), [len(sents) for sents in sentences])
    print(len(pun_words), len(sentences), [len(sent) for sent in sentences])
    return pun_words, sentences


if __name__ == '__main__':
    retrieve = load_json(sys.argv[1])
    retrieve_repl = load_json(sys.argv[2])
    rule = load_json(sys.argv[3])
    neural = load_json(sys.argv[4])
    pku = load_pku(sys.argv[5], sys.argv[6])
    pun_words, sentences = combine_results(pku, 1, retrieve, retrieve_repl, rule, neural)
    outfile = sys.argv[7]
    compose_eval_hit(sentences, pun_words, outfile)
    exit(0)
    sentences_array = [load_sentences(fname) for fname in sys.argv[1:-1]] 
    # the actual order: incremental, title-to-text, title-keywords-text, human_story
    names = ['pun', 'depun', 'retrieved_pw', 'retrieved_pw_alter', 'retrieved_aw', 'retrieved_aw_alter']
    compose_funniness_anno_hit(names, sys.argv[-1], *sentences_array) 
