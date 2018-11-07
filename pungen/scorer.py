import argparse
import torch
import numpy as np

from fairseq import data, options, tasks, utils, tokenizer
from fairseq.sequence_scorer import SequenceScorer

import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

import logging
logger = logging.getLogger('pungen')

class LMScorer(object):
    def __init__(self, task, scorer, use_cuda):
        self.task = task
        self.scorer = scorer
        self.use_cuda = use_cuda
        if use_cuda:
            self.scorer.cuda()

    @classmethod
    def load_model(cls, path, cpu=False):
        # TODO: don't hardcode path
        args = argparse.Namespace(data=path, path=path+'/wiki103.pt', cpu=cpu, task='language_modeling',
                output_dictionary_size=-1, self_target=False, future_target=False, past_target=False)
        use_cuda = torch.cuda.is_available() and not cpu
        task = tasks.setup_task(args)
        logger.info('loading language model from {}'.format(args.path))
        models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task)
        d = task.target_dictionary
        scorer = SequenceScorer(models, d)
        return cls(task, scorer, use_cuda)

    def score_sents(self, sents, tokenize=str.split):
        """Return log p at each word
        """
        itr = self.make_batches(sents, self.task.target_dictionary, self.scorer.models[0].max_positions(), tokenize=tokenize)
        results = self.scorer.score_batched_itr(itr, cuda=self.use_cuda)
        scores = []
        for id_, src_tokens, __, hypos in results:
            pos_scores = hypos[0]['positional_scores'].data.cpu().numpy()
            scores.append((int(id_.data.cpu().numpy()), pos_scores))
        # sort by id
        scores = [s[1] for s in sorted(scores, key=lambda x: x[0])]
        return scores

    def make_batches(self, lines, src_dict, max_positions, tokenize=str.split):
        tokens = [
            tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False, tokenize=tokenize).long()
            for src_str in lines
        ]
        lengths = np.array([t.numel() for t in tokens])

        # Load dataset
        # MonolingualDataset[i] = source, future_target, past_target
        # all targets are effectively ignored during inference
        dataset = data.MonolingualDataset(
                dataset=[(s[:-1], s[1:], None) for s in tokens],
                sizes=lengths, src_vocab=src_dict, tgt_vocab=src_dict,
                add_eos_for_other_targets=False, shuffle=False)
        itr = self.task.get_batch_iterator(
            dataset=dataset,
            max_tokens=100,
            max_sentences=5,
            max_positions=max_positions,
        ).next_epoch_itr(shuffle=False)

        return itr

class UnigramModel(object):
    def __init__(self, counts_path, oov_prob=0.03):
        self.word_counts = self.load_model(counts_path)
        self.total_count = sum(self.word_counts.values())
        self.oov_prob = oov_prob
        self._oov_smoothing_prob = self.oov_prob * (1. / self.total_count)

    def load_model(self, dict_path):
        counts = {}
        with open(dict_path, 'r') as fin:
            for line in fin:
                ss = line.strip().split()
                counts[ss[0]] = int(ss[1])
        return counts

    def _score(self, token):
        p = self.word_counts.get(token, 0) / float(self.total_count)
        smoothed_p = (1 - self.oov_prob) * p + self._oov_smoothing_prob
        return np.log(smoothed_p)

    def score(self, tokens):
        return [self._score(token) for token in tokens]


class PunScorer(object):
    def __init__(self, lm, um):
        self.lm = lm
        self.um = um

    def _get_window(self, i, w):
        start = max(0, i - w)
        end = i + w
        return start, end

    def grammaticality_score(self, sent, lm_scores):
        unigram_scores = self.um.score(sent)
        score = (np.sum(lm_scores) - np.sum(unigram_scores)) / len(sent)
        return score

    # TODO: batch
    def score(self, pun_sent, pun_word_id, alter_word, local_window_size=3):
        alter_sent = list(pun_sent)
        alter_sent[pun_word_id] = alter_word
        alter_word_id = pun_word_id
        alter_start, alter_end = self._get_window(alter_word_id, local_window_size)
        pun_start, pun_end = self._get_window(pun_word_id, local_window_size)
        local_pun_sent = pun_sent[pun_start:pun_end]
        local_alter_sent = alter_sent[alter_start:alter_end]
        sents = [alter_sent, pun_sent, local_alter_sent, local_pun_sent]
        scores = self.lm.score_sents(sents, tokenize=lambda x: x)
        # surprisal = logp(alter) - logp(pun)
        global_surprisal = np.mean(scores[0]) - np.mean(scores[1])
        local_surprisal = np.mean(scores[2]) - np.mean(scores[3])
        gram = self.grammaticality_score(pun_sent, scores[1])
        if not (global_surprisal > 0 and local_surprisal > 0):
            #print(' '.join(alter_sent))
            #print(' '.join(pun_sent))
            #print('global:', np.mean(scores[0]), np.mean(scores[1]), global_surprisal)
            #print(' '.join(local_alter_sent))
            #print(' '.join(local_pun_sent))
            #print('local:', np.mean(scores[2]), np.mean(scores[3]), local_surprisal)
            return -1000.
        else:
            r = local_surprisal / global_surprisal  # larger is better
            return float(r + gram)

class GoodmanPunScorer(object):
    def __init__(self, lm, um, skipgram):
        self.lm = lm
        self.um = um
        self.skipgram = skipgram

    def unigram_logprob(self, w):
        return self.um._score(w)

    def assignment_prior(self, sent):
        return len(sent) * np.log(0.5)

    def _word_likelihood(self, w, m, f):
        """\sum_{f \in {0, 1}} p(w | f, m)
        Args:
            w (str): word
            f (int): {0, 1} meaning assignment
            m (str): {pun word, alter word} meaning
        Return:
            probability
        """
        # p(w | m, f=1) = p(w | m)
        if f == 1:
            return self.skipgram.score(iword=m, oword=w)
        # p(w | m, f=0) = p(w)
        else:
            return np.exp(self.unigram_logprob(w))

    def word_likelihood(self, w, m):
        return np.log(self._word_likelihood(w, m, 1) + self._word_likelihood(w, m, 0))

    def meaning_prior(self, meanings):
        probs = [np.exp(self.unigram_logprob(m)) for m in meanings]
        z = sum(probs)
        logprobs = [np.log(p / z) for p in probs]
        return logprobs

    def meaning_posterior(self, m, sent, meaning_prior):
        """p(m | sent)
        """
        #lm_scores = self.lm.score_sents([sent])[0]
        #assert len(lm_scores) == len(sent)
        # NOTE: cannot use LM here because sent only contains content words

        # everything is in the log space
        # NOTE: need to renormalize priors
        #meaning_prior = self.unigram_logprob(m)
        assignment_prior = self.assignment_prior(sent)
        sent_likelihood = np.sum([self.word_likelihood(w, m) for w, lm_score in zip(sent, lm_scores)])

        return meaning_prior + assignment_prior + sent_likelihood

    def ambiguity(self, pun_word, alter_word, sent):
        meanings = [pun_word, alter_word]
        priors = self.meaning_prior(meanings)
        posteriors = [self.meaning_posterior(m, sent, p) for m, p in zip(meanings, priors)]
        entropy = -1 * sum([p * np.log(p) for p in posteriors])
        return entropy

    def kl_div(self, p1, p2):
        return np.sum([p1_ * np.log(p1_ / p2_) for p1_, p2_ in zip(p1, p2)])

    def distinctiveness(self, pun_word, alter_word, sent):
        # NOTE: we cannot use joint distribution
        kl_divs = []
        for w in sent:
            p1 = [self._word_likelihood(w, pun_word, f) for f in (0, 1)]
            p2 = [self._word_likelihood(w, alter_word, f) for f in (0, 1)]
            d = self.kl_div(p1, p2) + self.kl_div(p2, p1)
            kl_divs.append(d)
        return np.mean(kl_divs)

    def is_content(self, tag):
        if tag.startswith('NN') or \
           tag.startswith('VB') or \
           tag.startswith('JJ'):
            return True
        return False

    def score(self, pun_sent, pun_word_id, alter_word):
        pun_word = pun_sent[pun_word_id]
        parsed_sent = nlp(pun_sent)
        content_words = [x.text for x in parsed_sent if self.is_content(x.tag_)]
        ambiguity = self.ambiguity(pun_word, alter_word, content_words)
        distinctiveness = self.distinctiveness(pun_word, alter_word, content_words)
        return ambiguity + distinctiveness


if __name__ == '__main__':
    lm = LMScorer.load_model('models/wikitext')
    scorer = PunScorer(lm)
    pun_sent = 'he is going to dye'.split()
    alter_sent = 'he is going to die'.split()
    pun_word_id = 4
    scorer.score(alter_sent, pun_sent, pun_word_id, local_window_size=2)
