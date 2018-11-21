#!/user/bin/python

import sys, csv, re
import copy
import numpy as np
from operator import itemgetter
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pearsonr

#names = {'pun':0, 'depun':1, 'retrieved_pw':2, 'retrieved_pw_alter':3, 'retrieved_aw':4, 'retrieved_aw_alter':5}
def load_analysis_eval(infile):
    sentence_dict = dict()
    turker_dict = dict()
    line_num = 0
    with open(infile) as csvfile:
        inf = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for line in inf:
            line_num += 1
            if line_num == 1:
                header_dict = read_header(line)
                continue
            elems = line
            for i in range(1, 11):
                key = 'Answer.Story'+str(i)
                assert key in header_dict, key
                answer = int(elems[header_dict[key]])
                if answer != '':
                    if 'Input.sentence_'+str(i) in header_dict:
                        sentence = elems[header_dict['Input.sentence_'+str(i)]]
                    if 'Input.sentence_info_'+str(i) in header_dict:
                        sentence_info = elems[header_dict['Input.sentence_info_'+str(i)]].strip()
                    turker_id = elems[header_dict['WorkerId']] 
                    sent_key = sentence_info + '#' + turker_id
                    if sentence_info not in sentence_dict:
                        sentence_dict[sentence_info] = [sentence]
                    sentence_dict[sentence_info].append(answer)
                    if turker_id not in turker_dict:
                        turker_dict[turker_id] = []
                    turker_dict[turker_id].append((sentence_info, answer))
    return sentence_dict, turker_dict

def load_generation_eval(infile, num_methods, group_per_page):
    sentence_dict = dict()
    pun_dict = dict()
    turker_dict = dict()
    line_num = 0
    counter = 0
    with open(infile) as csvfile:
        inf = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for line in inf:
            line_num += 1
            if line_num == 1:
                header_dict = read_header(line)
                print (header_dict)
                continue
            elems = line
            turker_id = elems[header_dict['WorkerId']] 
            for i in range(1, 1+group_per_page):
                key = 'Input.Pun_alter_'+str(i)
                assert key in header_dict, key
                pun_alter_key = elems[header_dict[key]]
                if pun_alter_key not in pun_dict:
                    pun_dict[pun_alter_key] = [{} for i in range(num_methods)]
                else:
                    print(pun_alter_key, "has already be in the corpora!")
                key = 'Input.order_info_'+str(i) 
                assert key in header_dict, key
                order_info = elems[header_dict[key]].strip().split('-')
                temp_results = [None for od in order_info]
                for j in range(num_methods):
                    key = 'Answer.Sentence'+str(i)+'_'+str(j+1)
                    assert key in header_dict, key
                    answer = int(elems[header_dict[key]])
                    key = 'Input.Sentence'+str(i)+'_'+str(j+1) 
                    assert key in header_dict, key
                    sentence = elems[header_dict[key]]
                    temp_results[int(order_info[j])] = (sentence, answer)
                for j in range(num_methods):
                    sentence, score = temp_results[j]
                    if sentence not in pun_dict[pun_alter_key][j]:
                        pun_dict[pun_alter_key][j][sentence] = []
                    if j != num_methods - 1:
                        try:
                            assert len(pun_dict[pun_alter_key][j][sentence]) < 3
                        except:
                            print('loading data! Abnormal data:', j, temp_results[j], pun_dict[pun_alter_key][j][sentence], counter)
                            #sentence = '_'.join([sentence, str(j)])
                            counter += 1
                            '''if sentence not in pun_dict[pun_alter_key][j]:
                                pun_dict[pun_alter_key][j][sentence] = []
                            else:
                                assert len(pun_dict[pun_alter_key][j][sentence]) < 53                            '''
                    pun_dict[pun_alter_key][j][sentence].append(score)
                    if sentence not in sentence_dict:
                        sentence_dict[sentence] = [sentence]
                    sentence_dict[sentence].append(score)
                    if turker_id not in turker_dict:
                        turker_dict[turker_id] = []
                    turker_dict[turker_id].append((sentence, score))
    return sentence_dict, pun_dict, turker_dict

def compute_generated_pun_results(pun_dict, sentence_dict, names, scale):
    print ('total pun numbers:', len(pun_dict))
    print ('total annotation numbers:', len(sentence_dict))
    names = {0:'pku', 1:'retrieve', 2:'retrieve_repl', 3:'rule', 4:'neural', 5:'gold'}
    scores = [[] for i in range(len(names))]
    counts = [0] * len(names)
    annotations = []
    sentences_with_scores = []
    for key, value in pun_dict.items():
        assert len(value) == 6, len(value)
        #temp_scores = [[] for i in range(len(names))]
        for ii, sent_dic in enumerate(value):
            temp_scores = []
            #print(len(sent_dic))
            for k, v in sent_dic.items():
                assert k in sentence_dict, k
                if ii < len(names) -1 and len(sentence_dict[k]) > 6:
                    print('abnormal sentences!!', ii, sentence_dict[k], v)
                score = np.sum(sentence_dict[k][1:])/np.count_nonzero(sentence_dict[k][1:]) if np.count_nonzero(sentence_dict[k][1:]) > 0 else 1
                sentences_with_scores.append((k, names[ii], score))
                temp_scores.append(score)
            # this one can be changed to different strategies.
            scores[ii].append(np.max(temp_scores))
            counts[ii] += 1 
            #annotations.append(value)
    #annotations = np.array(annotations)
    #print(annotations.shape)
    '''kappa_array, spearman_array = [], []
    for i in range(annotations.shape[1]):
        for j in range(i+1, annotations.shape[1]):
            kappa_array.append(cohen_kappa_score(annotations[:,i], annotations[:, j]))
            spearman_array.append(spearmanr(annotations[:,i], annotations[:, j]))
    sorted_sents = sorted(sentences_with_scores, key=itemgetter(0))
    for item in sorted_sents:
        print('\t'.join(list(map(str, item))))
    '''
    for i in range(len(scores)-1):
        print(i, [(sum(np.array(sc)<np.array(scores[i])), sum(np.array(sc)>np.array(scores[i]))) for sc in scores[i+1:]])
    print([np.mean(sc)/scale for sc in scores])
    #return np.array(scores) / np.array(counts), np.mean(kappa_array), np.mean(spearman_array)

def filter_bad_turker(turker_dict, sentence_dict, thres=0.2):
    print ('total turker numbers:', len(turker_dict))
    for tk, v in turker_dict.items():
        tary, aary = [], []
        for sid, a in v:
            assert sid in sentence_dict
            #avg = np.mean(sentence_dict[sid])
            tary.append(a)
            aary.append(sentence_dict[sid][1:])
        correlations = []
        count = 0
        for elem in zip(*aary):
            corr = spearmanr(tary, elem)[0]
            #print(corr, count)
            if corr > 0.99 and count == 0:
                count += 1
                continue
            correlations.append(0 if np.isnan(corr) else corr)
        print(correlations)
        if len(correlations) != 0 and np.max(correlations) < thres:
            print ('turker', tk, 'is a bad turker with agreement', correlations, tary, aary)
            for i, (sid, a) in enumerate(v):
                annos = sentence_dict[sid][1:]
                idx = annos.index(a)
                del sentence_dict[sid][idx+1]
                #annos[idx] = round(aary[i])

def compute_results(sentence_dict):
    print ('total annotation numbers:', len(sentence_dict))
    scores = [0.0] * len(names)
    counts = [0] * len(names)
    annotations = []
    count = 0
    sentences_with_scores = []
    for key, value in sentence_dict.items():
        type_id = names['_'.join(key.split('_')[:-1])]
        if len(value) != 6:
            avg = np.mean(value[1:])
            value += [round(avg)] * (6-len(value))
        scores[type_id] += np.mean(value[1:])
        counts[type_id] += 1 
        annotations.append(value)
        sentences_with_scores.append((value[0], '_'.join(key.split('_')[:-1]), np.mean(value[1:])))
    annotations = np.array(annotations)
    print(annotations.shape)
    kappa_array, spearman_array = [], []
    for i in range(annotations.shape[1]):
        for j in range(i+1, annotations.shape[1]):
            kappa_array.append(cohen_kappa_score(annotations[:,i], annotations[:, j]))
            spearman_array.append(spearmanr(annotations[:,i], annotations[:, j]))
    sorted_sents = sorted(sentences_with_scores, key=itemgetter(0))
    for item in sorted_sents:
        print('\t'.join(list(map(str, item))))
    return np.array(scores) / np.array(counts), np.mean(kappa_array), np.mean(spearman_array)


def read_header(header_array):
    header_dict = dict()
    for i, header in enumerate(header_array):
        if header.startswith('Answer') or header.startswith('WorkerId') or header.startswith('Input'):
            header_dict[header] = i
    #print len(header_dict), header_dict
    return header_dict

# get statistics for each story.
# get statistics for each turker.
def load_data(infile):
    sentence_dict = dict()
    turker_dict = dict()
    line_num = 0
    with open(infile) as csvfile:
        inf = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for line in inf:
            line_num += 1
            if line_num == 1:
                header_dict = read_header(line)
                continue
            elems = line
            for i in range(1, 11):
                key = 'Answer.Ending'+str(i)
                assert key in header_dict
                answer = elems[header_dict[key]]
                if answer != '':
                    storyid = elems[header_dict['Input.storyid'+str(i)]]
                    story_content = []
                    for j in range(1, 6):
                        story_content.append(elems[header_dict['Input.sentence'+str(i)+'_'+str(j)]])
                    if storyid not in sentence_dict:
                        sentence_dict[storyid] = ['\t'.join(story_content)]
                    turker_id = elems[header_dict['WorkerId']] 
                    sentence_dict[storyid].append([turker_id, answer])
                    if turker_id not in turker_dict:
                        turker_dict[turker_id] = []
                    turker_dict[turker_id].append((storyid, answer))
    return sentence_dict, turker_dict

def load_gold(infile):
    gold_sentence_dict = dict()
    with open(infile, 'rU') as csvfile:
        inf = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for line in inf:
            assert line[0] not in gold_sentence_dict
            gold_sentence_dict[line[0]] = line[-1]
    print (gold_sentence_dict)
    return gold_sentence_dict

def load_data_computer_generated(infile):
    sentence_dict = dict()
    turker_dict = dict()
    line_num = 0
    with open(infile) as csvfile:
        inf = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for line in inf:
            line_num += 1
            if line_num == 1:
                header_dict = read_header(line)
                continue
            elems = line
            for i in range(1, 11):
                key = 'Answer.Ending'+str(i)
                assert key in header_dict
                answer = elems[header_dict[key]]
                if answer != '':
                    story_content = []
                    for j in range(1, 6):
                        story_content.append(elems[header_dict['Input.sentence'+str(i)+'_'+str(j)]])
                    story_content = '\t'.join(story_content)
                    if story_content not in sentence_dict:
                        sentence_dict[story_content] = ['Dummy']
                    turker_id = elems[header_dict['WorkerId']] 
                    sentence_dict[story_content].append([turker_id, answer])
                    if turker_id not in turker_dict:
                        turker_dict[turker_id] = []
                    turker_dict[turker_id].append((story_content, answer))
    print ('story dict size:', len(sentence_dict), 'turker dict size:', len(turker_dict))
    return sentence_dict, turker_dict

def load_gold_computer_generated(infile):
    gold_sentence_dict = dict()
    with open(infile) as inf:
        for line in inf:
            elems = re.split(r'([\.!\?])', line.strip())
            assert len(elems) == 11, elems
            label = elems[0].split(' ')[0][1:-6]
            #elems[0] = ' '.join(elems[0].split(' ')[1:])
            elems = [elems[i]+elems[i+1] for i in range(0,10,2)]
            assert len(elems) == 5
            key = '\t'.join(elems)
            assert key not in gold_sentence_dict
            gold_sentence_dict[key] = label
    print ('gold dict size:', len(gold_sentence_dict))
    return gold_sentence_dict

def decide_label(story_info):
    if len(story_info) > 2:
        if story_info[-2][-1] == story_info[-1][-1]:
            gold = story_info[-1][-1]
        else:
            gold = 'Other'
    else:
        gold = story_info[-1][-1]
    return gold

def parse_keywords_eval_hit(infile):
    with open(infile) as csvfile:
        inf = csv.reader(csvfile, delimiter=',', quotechar='\"')
        header_dict = dict()
        line_num = 0
        for line in inf:
            line_num += 1
            if line_num == 1:
                for i, header in enumerate(line):
                    if header.startswith('story') and header.endswith('2'):
                        header_dict[header] = i
            else:
                for i in range(1,11):
                    print (line[header_dict['story'+str(i)+'_2']])

if __name__ == '__main__':
    infile = sys.argv[1]
    scale = float(sys.argv[2])
    num_methods = 6
    group_per_page = 2
    sentence_dict, pun_dict, turker_dict = load_generation_eval(infile, num_methods, group_per_page)
    print(len(sentence_dict))
    print(len(pun_dict))
    print(len(turker_dict))
    filter_bad_turker(turker_dict, sentence_dict, thres=0.08*scale)
    names = {'pku':0, 'retrieve':1, 'retrieve_repl':2, 'rule':3, 'neural':4, 'gold':5}
    compute_generated_pun_results(pun_dict, sentence_dict, names, scale) 
    exit(0)
    names = {'pun':0, 'depun':1, 'retrieved_pw':2, 'retrieved_aw':3}
    sentence_dict, turker_dict = load_analysis_eval(sys.argv[1])
    filter_bad_turker(turker_dict, sentence_dict, thres=0.08*scale)
    print (compute_results(sentence_dict))
