import argparse
import torch
import numpy as np

# TODO: must import options before task??
from fairseq import data, options, tasks, utils, tokenizer
from fairseq.sequence_scorer import SequenceScorer

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
        print('| loading model(s) from {}'.format(args.path))
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
        itr = data.EpochBatchIterator(
            dataset=data.MonolingualDataset([(s[:-1], s[1:]) for s in tokens], lengths, src_dict, False),
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


if __name__ == '__main__':
    lm = LMScorer.load_model('models/wikitext')
    scorer = PunScorer(lm)
    pun_sent = 'he is going to dye'.split()
    alter_sent = 'he is going to die'.split()
    pun_word_id = 4
    scorer.score(alter_sent, pun_sent, pun_word_id, local_window_size=2)
