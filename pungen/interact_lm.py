#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from argparse import Namespace
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import data, options, progress_bar, tasks, utils, tokenizer
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions.adaptive_loss import AdaptiveLoss

def make_batches(lines, args, src_dict, max_positions):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = data.EpochBatchIterator(
        dataset=data.MonolingualDataset([(s[:-1], s[1:]) for s in tokens], lengths, src_dict, False),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    return itr
    #for batch in itr:
    #    yield batch
        #yield Batch(
        #    srcs=[lines[i] for i in batch['id']],
        #    tokens=batch['net_input']['src_tokens'],
        #    lengths=None,
        #    prefix=None,
        #), batch['id']

def main(args):
    assert args.path is not None, '--path required for evaluation!'

    args.tokens_per_sample = getattr(args, 'tokens_per_sample', 1024)
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task)

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    #for model in models:
    #    model.make_generation_fast_()
    #    if args.fp16:
    #        model.half()

    d = task.target_dictionary
    crit_args = Namespace(sentence_avg=False)
    criterion = AdaptiveLoss(crit_args, task)
    scorer = SequenceScorer(models, d)
    lm = models[0]
    model = models[0]
    if use_cuda:
        scorer.cuda()
        criterion.cuda()
        lm.cuda()

    #s = input('text: ')
    s = "he is going to dye ."
    l = 3
    e = 2
    while True:
        itr = make_batches([s], args, d, model.max_positions())
        words = s.split()
        for sample in itr:
            print('use cuda:', use_cuda)
            s = utils.move_to_cuda(sample) if use_cuda else sample
            lm.eval()
            lm.zero_grad()
            wid = d.index(words[e])
            optimizer = torch.optim.Adagrad([lm.decoder.embed_tokens.weight], lr=0.1)
            for step in range(50):
                loss = criterion(lm, s, reduce=False)[0]
                print(loss[l])
                loss[l].backward()
                optimizer.step()
                if step % 10 == 0:
                    all_v = lm.decoder.embed_tokens.weight
                    vocab_size = all_v.size(0)
                    new_v = lm.decoder.embed_tokens.weight[wid]
                    new_v = new_v.repeat(vocab_size, 1)
                    #print(lm.decoder.embed_tokens.weight.grad[wid])
                    #sim = torch.matmul(all_v, new_v.unsqueeze(1)).squeeze().data.cpu().numpy()
                    sim = F.cosine_similarity(all_v, new_v).data.cpu().numpy()
                    ids = np.argsort(sim)[::-1]
                    print([d[i] for i in ids[:10]])
                    print([sim[i] for i in ids[:10]])

        import sys; sys.exit()

        results = scorer.score_batched_itr(itr, cuda=use_cuda)
        for _, src_tokens, __, hypos in results:
            for hypo in hypos:
                pos_scores = hypo['positional_scores']
                lm.zero_grad()
                pos_scores[3].backward()
                wid = d.index('he')
                g = lm.decoder.embed_tokens.weight[wid].grad
                print(g.size())
                inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                if inf_scores.any():
                    print('| Skipping tokens with inf scores:',
                          task.target_dictionary.string(hypo['tokens'][inf_scores.nonzero()]))
                    pos_scores = pos_scores[(~inf_scores).nonzero()]
                words = s.split()[1:]
                print([(w, x) for w, x in zip(words, pos_scores.data.cpu().numpy())])


    score_sum = 0.
    count = 0
    with progress_bar.build_progress_bar(args, itr) as t:
        results = scorer.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
        wps_meter = TimeMeter()
        for _, src_tokens, __, hypos in results:
            for hypo in hypos:
                pos_scores = hypo['positional_scores']
                inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                if inf_scores.any():
                    print('| Skipping tokens with inf scores:',
                          task.target_dictionary.string(hypo['tokens'][inf_scores.nonzero()]))
                    pos_scores = pos_scores[(~inf_scores).nonzero()]
                score_sum += pos_scores.sum()
                count += pos_scores.numel()
            wps_meter.update(src_tokens.size(0))
            t.log({'wps': round(wps_meter.avg)})

    avg_nll_loss = -score_sum / count
    print('| Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))
    print('| Loss: {:.4f}, Perplexity: {:.2f}'.format(avg_nll_loss, np.exp(avg_nll_loss)))


if __name__ == '__main__':
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
