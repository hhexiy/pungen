import argparse
import os
import numpy as np
import time
import pickle
import torch
from operator import itemgetter
from fairseq import data, options, progress_bar, tasks, utils, tokenizer
from fairseq.sequence_scorer import SequenceScorer
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def load_lm(lm_path, cpu=False):
    # TODO: don't hardcode path
    args = argparse.Namespace(data=lm_path, path=lm_path+'/wiki103.pt', cpu=cpu, task='language_modeling')
    use_cuda = torch.cuda.is_available() and not args.cpu
    task = tasks.setup_task(args)
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task)
    d = task.target_dictionary
    scorer = SequenceScorer(models, d)
    if use_cuda:
        scorer.cuda()
    return task, scorer

def make_batches(lines, src_dict, max_positions):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
        for src_str in lines
    ]
    idx_to_words = {v: k for k, v in src_dict.indices.items()}
    lengths = np.array([t.numel() for t in tokens])
    itr = data.EpochBatchIterator(
        dataset=data.MonolingualDataset([(s[:-1], s[1:]) for s in tokens], lengths, src_dict, False),
        max_tokens=100,
        max_sentences=5,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    return itr

def score_sents(sents, lm_task, scorer, use_cuda=True):
    itr = make_batches(sents, lm_task.target_dictionary, scorer.models[0].max_positions())
    idx_to_words = {v: k for k, v in lm_task.target_dictionary.indices.items()}
    results = scorer.score_batched_itr(itr, cuda=use_cuda)
    scores = []
    new_sents = []
    #print('in score_sents')
    for _, src_tokens, __, hypos in results:
        #print(len(src_tokens), len(hypos[0]['positional_scores']))
        #print(list(map(lambda x: idx_to_words[x], src_tokens.data.cpu().numpy())))
        for hypo in hypos:
            pos_scores = hypo['positional_scores']
            scores.append(pos_scores.data.cpu().numpy())
            new_sents.append(list(map(lambda x: idx_to_words[x], src_tokens.data.cpu().numpy())))
    #for i in range(10):
    #    print(len(new_sents[i]), len(scores[i]))
    #    print(new_sents[i], scores[i])
    return scores, new_sents

def compute_pun_score(alter_sent_score, pun_sent_score, pun_word_id):
    N = len(alter_sent_score)
    alter_score = np.mean(alter_sent_score[max(0, pun_word_id-2):min(pun_word_id+2, N)])
    pun_score = pun_sent_score[pun_word_id]
    return alter_score + pun_score

def print_scored_sent(score, sent):
    if type(sent) == str:
        sent = sent.split()
    a = ['{}|{:.2f}'.format(w, s) for w, s in zip(sent[1:], score)]
    print('{} {}'.format(sent[0], ' '.join(a)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc-file', help='tokenized training corpus')
    parser.add_argument('--lm-path', help='pretrained LM for scoring')
    #parser.add_argument('--path', default='models/retriever.pkl', help='retriever model path')
    parser.add_argument('--save', default='histogram.png', help='save the plot to')
    parser.add_argument('--infile', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()

    start = time.time()
    print('Reading docs')
    with open(args.doc_file, 'r') as fin:
        docs = [line.strip() for line in fin]
    print('{} s'.format(time.time() - start))

    lm_task, scorer = load_lm(args.lm_path)

    corpora = []
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    names = [os.path.basename(f).split('.')[0] for f in args.infile] 
    print(names)
    infs = [open(f, 'r') for f in args.infile]
    for fin in infs:
        sents = []
        for line in fin:
            sents.append(line.strip())
        corpora.append(sents)
    plt.xlabel('Perplexity')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ Perplexity\ Differences:}$')
    data_scores = []
    data_probs = []
    for i, sents in enumerate(corpora):
        scores, new_sents = score_sents(sents, lm_task, scorer)
        #ppls = list(filter(lambda x: (x < 2000 and x > 10), map(lambda x: np.exp(-np.mean(x)), scores)))
        ppls = list(map(lambda x: np.exp(-np.mean(x)), scores))
        print('datasize:', len(ppls), len(scores), len(sents), len(new_sents))
        data_scores.append(ppls)

        data_probs.append(scores)
        ## Plot corpus-wise perplexity statistics 
        '''weights = np.ones_like(ppls)/float(len(ppls))
        n, bins, patches = plt.hist(ppls, 30, weights=weights, alpha=0.5, \
                range=[0, 2000], color=colors[i], label=names[i])
        '''
    #data_weights = [np.ones_like(ppls)/float(len(ppls)) for ppls in data_scores]
    #n, bins, patches = plt.hist(data_scores, 20, weights=data_weights, \
    #            range=[0, 2000], color=colors[:len(names)], label=names) #density=True)
    ## Plot corpus-wise difference statistics
    ppl_diff = [np.array(data_scores[i+1]) - np.array(data_scores[i]) for i in range(0,len(data_scores),2)]
    print(len(ppl_diff), len(corpora), len(data_probs))
    for ii, diff in enumerate(ppl_diff):
        idxs = np.nonzero(abs(diff) > 1000)[0]
        print(len(idxs), idxs)
        sents1, sents2 = itemgetter(*idxs)(corpora[2*ii]), itemgetter(*idxs)(corpora[2*ii+1])
        scores1, scores2 = itemgetter(*idxs)(data_probs[2*ii]), itemgetter(*idxs)(data_probs[2*ii+1])
        '''for s1, s2, c1, c2 in zip(sents1, sents2, scores1, scores2):
            print(s1, c1)
            print_scored_sent(c1, s1)
            print(s2, c2)
            print_scored_sent(c2, s2)
        '''
    diff_weights = [np.ones_like(ppls)/float(len(ppls)) for ppls in ppl_diff] 
    for i, (diff, wt) in enumerate(zip(ppl_diff, diff_weights)):
        n, bins, patches = plt.hist(diff, 30, weights=wt, alpha=0.5, \
                range=[-1000, 1000], color=colors[i], label=names[2*i+1]+' - '+names[2*i]) 
    
    #n, bins, patches = plt.hist(ppl_diff, 20, weights=diff_weights, \
    #        range=[-1000, 1000], color=colors[:len(names)-1], label=names[1:])
    plt.legend()
    plt.savefig(args.save)
    for f in infs:
        f.close()
