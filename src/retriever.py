import argparse
import os
import numpy as np
import time
import pickle
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from fairseq import data, options, progress_bar, tasks, utils, tokenizer
from fairseq.sequence_scorer import SequenceScorer

class Retriever(object):
    def __init__(self, docs, path=None):
        if path is None:
            self.vectorizer = TfidfVectorizer(analyzer=str.split)
            self.tfidf_matrix = self.vectorizer.fit_transform(docs)
        else:
            with open(path, 'rb') as fin:
                obj = pickle.load(fin)
                self.vectorizer = obj['vectorizer']
                self.tfidf_matrix = obj['tfidf_mat']

    def save(self, path):
        with open(path, 'wb') as fout:
            obj = {
                    'vectorizer': self.vectorizer,
                    'tfidf_mat': self.tfidf_matrix,
                    }
            pickle.dump(obj, fout)

    def query(self, keywords, k=1):
        features = self.vectorizer.transform([keywords])
        scores = self.tfidf_matrix * features.T
        scores = scores.todense()
        scores = np.squeeze(np.array(scores), axis=1)
        ids = np.argsort(scores)[-k:][::-1]
        return ids

def keywords_at_the_end(words, sent):
    pos = []
    sent = sent.split()
    if len(sent) < 5:
        return False
    for i, w in enumerate(sent):
        if w in words:
            pos.append(i)
    if len(pos) != len(words):
        return False
    for p in pos:
        if p < min(int(len(sent) * 0.7), len(sent) - 1):
            return False
    return True

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
    results = scorer.score_batched_itr(itr, cuda=use_cuda)
    scores = []
    for _, src_tokens, __, hypos in results:
        for hypo in hypos:
            pos_scores = hypo['positional_scores']
            scores.append(pos_scores.data.cpu().numpy())
    return scores

def compute_pun_score(alter_sent_score, pun_sent_score, pun_word_id):
    N = len(alter_sent_score)
    alter_score = np.mean(alter_sent_score[max(0, pun_word_id-2):min(pun_word_id+2, N)])
    pun_score = pun_sent_score[pun_word_id]
    return alter_score + pun_score

def print_scored_sent(score, sent):
    sent = sent.split()
    a = ['{}|{:.2f}'.format(w, s) for w, s in zip(sent[1:], score)]
    print('{} {}'.format(sent[0], ' '.join(a)))

def retrieve(alter_word, pun_word, lm_task, scorer):
    ids = retriever.query(alter_word, 50)
    alter_sents = []
    pun_sents = []
    alter_scores = []
    pun_scores = []
    pun_word_ids = []
    sent_pun_scores = []
    for id_ in ids:
        if keywords_at_the_end([alter_word], docs[id_]):
            s = docs[id_].split()
            ids = [i for i, w in enumerate(s) if w == alter_word]
            alter_sent = ' '.join(s)
            pun_sent = ' '.join([x if x != alter_word else pun_word for x in s])
            pun_word_ids.append(max(ids))
            alter_score = score_sents([alter_sent], lm_task, scorer)[0]
            pun_score = score_sents([pun_sent], lm_task, scorer)[0]

            alter_sents.append(alter_sent)
            pun_sents.append(pun_sent)
            alter_scores.append(alter_score)
            pun_scores.append(pun_score)
            sent_pun_scores.append(compute_pun_score(alter_score, pun_score, max(ids)))
    sorted_ids = np.argsort(sent_pun_scores)[::-1]
    for id_ in sorted_ids:
        print_scored_sent(alter_scores[id_], alter_sents[id_])
        print_scored_sent(pun_scores[id_], pun_sents[id_])
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc-file', help='tokenized training corpus')
    parser.add_argument('--lm-path', help='pretrained LM for scoring')
    parser.add_argument('--keywords', help='file containing keywords')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--path', default='models/retriever.pkl', help='retriever model path')
    args = parser.parse_args()

    start = time.time()
    print('Reading docs')
    with open(args.doc_file, 'r') as fin:
        docs = [line.strip() for line in fin]
    print('{} s'.format(time.time() - start))

    start = time.time()
    print('Building matrix')
    if args.path and os.path.exists(args.path):
        print('Loading from', args.path)
        retriever = Retriever(docs, args.path)
    else:
        retriever = Retriever(docs)
        if args.path is not None:
            retriever.save(args.path)
    print('{} s'.format(time.time() - start))

    lm_task, scorer = load_lm(args.lm_path)

    if args.interactive:
        while True:
            alter_word, pun_word = input('Keywords:\n').split()
            retrieve(alter_word, pun_word, lm_task, scorer)
            ids = retriever.query(pun_word, 50)
            for id_ in ids:
                if keywords_at_the_end([pun_word], docs[id_]):
                    print(docs[id_])
    else:
        with open(args.keywords, 'r') as fin:
            for line in fin:
                ids = retriever.query(line.strip(), 10)
                contents = [docs[id_] for id_ in ids]
                contents.sort(key = lambda s: len(s), reverse=True)
#                print(line.strip())
                for ct in contents[:1]:
                    print(ct)
#                print('')

