import argparse
import os
import pickle as pkl
import torch
import numpy as np
from collections import defaultdict
import itertools
from scipy.stats import entropy

from fairseq import data, options, tasks, utils, tokenizer
from fairseq.sequence_scorer import SequenceScorer

import logging
logger = logging.getLogger('pungen')

from .utils import get_lemma, get_spacy_nlp, STOP_WORDS
nlp = get_spacy_nlp()

def is_content(word, tag):
    if not word in STOP_WORDS and \
       (tag.startswith('NN') or \
        tag.startswith('VB') or \
        tag.startswith('JJ')):
        return True
    return False

class LMScorer(object):
    def __init__(self, task, scorer, use_cuda):
        self.task = task
        self.scorer = scorer
        self.use_cuda = use_cuda
        if use_cuda:
            self.scorer.cuda()

    @classmethod
    def load_model(cls, path, cpu=False):
        args = argparse.Namespace(data=os.path.dirname(path), path=path, cpu=cpu, task='language_modeling',
                output_dictionary_size=-1, self_target=False, future_target=False, past_target=False)
        use_cuda = torch.cuda.is_available() and not cpu
        logger.info('loading language model from {}'.format(args.path))
        task = tasks.setup_task(args)
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
    def analyze(self, pun_sent, pun_word_id, alter_word):
        """Return multiple scores by category.
        """
        raise NotImplementedError

    def score(self, pun_sent, pun_word_id, alter_word):
        """Return aggregated scores.
        """
        scores = self.analyze(pun_sent, pun_word_id, alter_word)
        return sum(scores.values())

class RandomScorer(PunScorer):
    def analyze(self, pun_sent, pun_word_id, alter_word):
        return {'random': float(np.random.random())}

class SurprisalScorer(PunScorer):
    def __init__(self, lm, um, local_window_size=2):
        self.lm = lm
        self.um = um
        self.local_window_size = local_window_size

    def _get_window(self, i, w):
        start = max(0, i - w)
        end = i + w
        return start, end

    def grammaticality_score(self, sent, lm_scores):
        unigram_scores = self.um.score(sent)
        score = (np.sum(lm_scores) - np.sum(unigram_scores)) / len(sent)
        return score

    def analyze(self, pun_sent, pun_word_id, alter_word):
        def normalize(x, y):
            px = np.exp(x)
            py = np.exp(y)
            z = px + py
            return np.log(px/z), np.log(py/z)

        alter_sent = list(pun_sent)
        alter_sent[pun_word_id] = alter_word

        local_start, local_end = self._get_window(pun_word_id, self.local_window_size)
        local_pun_sent = pun_sent[local_start:local_end]
        local_alter_sent = alter_sent[local_start:local_end]

        sents = [alter_sent, pun_sent, local_alter_sent, local_pun_sent]
        scores = self.lm.score_sents(sents, tokenize=lambda x: x)
        global_surprisal = np.sum(scores[0]) - np.sum(scores[1])
        local_surprisal = np.sum(scores[2]) - np.sum(scores[3])
        grammar = self.grammaticality_score(pun_sent, scores[1])

        # ratio
        if not (global_surprisal > 0 and local_surprisal > 0):
            r = -1.
        else:
            r = local_surprisal / global_surprisal  # larger is better

        res = {'grammar': grammar, 'ratio': r,
               'local': local_surprisal, 'global': global_surprisal,
              }
        res = {k: float(v) for k, v in res.items()}
        return res


class GoodmanScoreCaculator(object):
    def __init__(self, um, skipgram, words, meanings, glove):
        self.words = words
        self.meanings = meanings
        _words = list(set(words + meanings))
        self.unigram_logprobs = {w: um._score(w) for w in _words}
        self.unigram_probs = {w: np.exp(s) for w, s in self.unigram_logprobs.items()}
        self.skipgram_probs = self.skipgram_scores(skipgram, _words, meanings)
        self.meaning_prior = self.meaning_prior()

    def glove_scores(self, glove, words, meanings):
        n = len(words)
        scores = defaultdict(dict)
        _scores = glove.cosine_similarity(meanings, words)
        print(_scores.shape)
        for i, w in enumerate(words):
            for j, m in enumerate(meanings):
                scores[w][m] = np.exp(_scores[j][i] * 0.3 + self.unigram_logprobs[w])
                assert scores[w][m] < 1
        return scores

    def skipgram_scores(self, skipgram, words, meanings):
        n = len(words)
        # p(w | m)
        scores = defaultdict(dict)
        # p(oword | iword)
        _scores = skipgram.score(iwords=meanings, owords=words, lemma=True)
        for i, w in enumerate(words):
            for j, m in enumerate(meanings):
                scores[w][m] = _scores[i][j]
        return scores

    def _word_likelihood_normalizer(self, w):
        meanings = self.meanings
        p_w = self.unigram_probs[w]
        p_m = [self.unigram_probs[m] for m in meanings]
        p_w_m = [self.skipgram_probs[w][m] for m in meanings]
        z = p_w / np.dot(p_m, p_w_m)
        return z

    def _word_likelihood(self, w, m, f):
        """\sum_{f \in {0, 1}} p(w | f, m)
        """
        # p(w | m, f=1) = p(w | m)
        if f == 1:
            z = self._word_likelihood_normalizer(w)
            score = self.skipgram_probs[w][m] * z
            return score
        # p(w | m, f=0) = p(w)
        else:
            return self.unigram_probs[w]

    def word_likelihood(self, w, m):
        return np.log(
                self._word_likelihood(w, m, 1) + self._word_likelihood(w, m, 0)
               )

    def meaning_prior(self):
        probs = [self.unigram_probs[m] for m in self.meanings]
        z = sum(probs)
        logprobs = {m: np.log(p / z) for m, p in zip(self.meanings, probs)}
        return logprobs

    def _meaning_posterior(self, m):
        """p(m | sent)
        """
        sent = self.words
        # NOTE: ignore the assignment prior which is a constant
        sent_likelihood = np.sum([self.word_likelihood(w, m) for w in sent])
        return self.meaning_prior[m] + sent_likelihood

    def meaning_posterior(self):
        posteriors = [self._meaning_posterior(m) for m in self.meanings]
        # Normalize to distribution so that entropy makes sense
        posteriors = [np.exp(p) for p in posteriors]
        z = np.sum(posteriors)
        posteriors = [np.log(p / z) for p in posteriors]
        return {m: p for m, p in zip(self.meanings, posteriors)}

    def ambiguity(self):
        meanings = self.meanings
        sent = self.words
        posteriors = self.meaning_posterior().values()
        #logger.debug('posteriors: {}'.format(posteriors))
        entropy = -1 * sum([np.exp(logp) * logp for logp in posteriors])
        #logger.debug('entropy: {}'.format(entropy))
        return entropy

    def kl_div(self, p1, p2):
        p1 = p1 / np.sum(p1)
        p2 = p2 / np.sum(p2)
        return np.sum([p1_ * np.log(p1_ / p2_) for p1_, p2_ in zip(p1, p2)])

    def distinctiveness(self):
        meanings = self.meanings
        sent = self.words
        kl_divs = []
        for i, w in enumerate(sent):
            p1 = [self._word_likelihood(w, meanings[0], f) for f in (0, 1)]
            p2 = [self._word_likelihood(w, meanings[1], f) for f in (0, 1)]
            d = self.kl_div(p1, p2) + self.kl_div(p2, p1)
            kl_divs.append(d)
        return np.sum(kl_divs)

    def distinctiveness_enum(self):
        words, meanings = self.words, self.meanings
        combinations = ["".join(seq) for seq in itertools.product("01", repeat=len(words))]
        dist_ma = np.zeros(len(combinations))
        dist_mb = np.zeros(len(combinations))
        for j, fvec in enumerate(combinations):
            fvec = [int(i) for i in list(fvec)]
            logp_w_given_m_f = np.array([0.0, 0.0])
            for i, f in enumerate(fvec):
                logp_w_given_m_f[0] += np.log(self._word_likelihood(words[i], meanings[0], f))
                logp_w_given_m_f[1] += np.log(self._word_likelihood(words[i], meanings[1], f))
            dist_ma[j] = np.exp(logp_w_given_m_f[0])
            dist_mb[j] = np.exp(logp_w_given_m_f[1])
        distinctiveness = entropy(dist_ma, dist_mb) +  entropy(dist_mb, dist_ma)
        return distinctiveness

class GoodmanScorer(PunScorer):
    def __init__(self, um, skipgram, glove=None):
        self.um = um
        self.skipgram = skipgram
        self.glove = glove

    def is_content(self, word, tag):
        if not word in STOP_WORDS and \
           (tag.startswith('NN') or \
            tag.startswith('VB') or \
            tag.startswith('JJ')):
            return True
        return False

    def _get_window(self, i, w):
        start = max(0, i - w)
        end = i + w
        return start, end

    def analyze(self, pun_sent, pun_word_id, alter_word):
        pun_word = pun_sent[pun_word_id]
        pun_word = get_lemma(pun_word)
        alter_word = get_lemma(alter_word)
        meanings = [pun_word, alter_word]

        parsed_sent = nlp(' '.join(pun_sent))
        content_words = [get_lemma(x, parsed=True) for x in parsed_sent if self.is_content(x.text, x.tag_)]

        calculator = GoodmanScoreCaculator(self.um, self.skipgram, content_words, meanings, self.glove)
        ambiguity = calculator.ambiguity()
        distinctiveness = calculator.distinctiveness()

        res = {'ambiguity': ambiguity, 'distinctiveness': distinctiveness}
        res = {k: float(v) for k, v in res.items()}
        return res


class LearnedScorer(PunScorer):
    def __init__(self, model, features, scorers):
        self.model = model
        self.features = features
        self.scorers = scorers

    @classmethod
    def from_pickle(cls, model_path, features_path, scorers):
        model = pkl.load(open(model_path, 'rb'))
        features = pkl.load(open(features_path, 'rb'))
        return cls(model, features, scorers)

    def analyze(self, pun_sent, pun_word_id, alter_word):
        res = {}
        for scorer in self.scorers:
            res.update(scorer.analyze(pun_sent, pun_word_id, alter_word))
        return res

    def score(self, pun_sent, pun_word_id, alter_word):
        res = self.analyze(pun_sent, pun_word_id, alter_word)
        score = self.model.predict([[res[f] for f in self.features]])
        return float(score)
