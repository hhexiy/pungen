#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import namedtuple
import numpy as np
import sys
import copy
import os
import pickle

import torch

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator

from wordvec.model import Word2Vec, SGNS


Batch = namedtuple('Batch', 'srcs tokens lengths prefix')
Translation = namedtuple('Translation', 'src_str hypos alignments')


def buffered_read(buffer_size):
    buffer = []
    for src_str in sys.stdin:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, src_dict, max_positions, tgt_str=None, tgt_dict=None):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
        for src_str in lines
    ]
    if not tgt_str is None:
        tgt_tokens = [
            tokenizer.Tokenizer.tokenize(tgt_str, tgt_dict, add_if_not_exist=False).long()
                ]
    else:
        tgt_tokens = None
    lengths = np.array([t.numel() for t in tokens])
    itr = data.EpochBatchIterator(
        dataset=data.LanguagePairDataset(tokens, lengths, src_dict, tgt=tgt_tokens, tgt_sizes=None, tgt_dict=tgt_dict),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        if not tgt_str is None:
            yield Batch(
                srcs=[lines[i] for i in batch['id']],
                tokens=batch['net_input']['src_tokens'],
                lengths=batch['net_input']['src_lengths'],
                prefix=batch['target'][:, :3],
            ), batch['id']
        else:
            yield Batch(
                srcs=[lines[i] for i in batch['id']],
                tokens=batch['net_input']['src_tokens'],
                lengths=batch['net_input']['src_lengths'],
                prefix=None,
            ), batch['id']


def main(args):
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    models, model_args = utils.load_ensemble_for_inference(model_paths, task)

    if args.skipgram_model:
        idx2word = pickle.load(open(os.path.join(args.skipgram_data, 'idx2word.dat'), 'rb'))
        word2idx = pickle.load(open(os.path.join(args.skipgram_data, 'word2idx.dat'), 'rb'))
        vocab_size = len(idx2word)
        # TODO: use saved e_dim
        model = Word2Vec(vocab_size=vocab_size, embedding_size=300)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=1, weights=None)
        sgns.load_state_dict(torch.load(args.skipgram_model))
        if use_cuda:
            sgns = sgns.cuda()
        sgns.eval()

        model_dict = models[0].decoder.dictionary
        sgns_dict_map = []
        for i in range(len(idx2word)):
            sgns_dict_map.append(model_dict.index(idx2word[i]))

        all_neighbor_word = range(len(idx2word))

    # Load LM
    if args.lm:
        print('test time lm')
        lm_args = copy.copy(args)
        setattr(lm_args, 'task', 'language_modeling')
        # For loading vocab
        setattr(lm_args, 'data', os.path.dirname(args.lm))
        lm_task = tasks.setup_task(lm_args)
        print('| loading pretrained LM from {}'.format(args.lm))
        lm, _ = utils.load_ensemble_for_inference([args.lm], lm_task)
        lm = lm[0]
        lm_dict = lm_task.dictionary
    else:
        lm, lm_dict = None, None

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(beamable_mm_beam_size=None if args.no_beamable_mm else args.beam)
        if args.fp16:
            model.half()

    # Initialize generator
    translator = SequenceGenerator(
        models, tgt_dict, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
        unk_penalty=args.unkpen, sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
        minlen=args.min_len,
        lm=lm, lm_dict=lm_dict,
    )

    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    def make_result(src_str, hypos):
        result = Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu(),
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.alignments.append('A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment))))
        return result

    def process_batch(batch, pun_prob=None, alter_prob=None, prefix=None):
        tokens = batch.tokens
        lengths = batch.lengths
        prefix = batch.prefix

        if use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()
            if not prefix is None:
                prefix = prefix.cuda()

        translations = translator.generate(
            tokens,
            lengths,
            maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
            pun_prob=pun_prob, alter_prob=alter_prob,
            prefix_tokens=prefix,
        )

        return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]

    decoder_dict_size = len(tgt_dict)
    def get_related_prob(word):
        iword = [word2idx[word]]
        owords = all_neighbor_word
        ivectors = sgns.embedding.forward_i(iword)
        ovectors = sgns.embedding.forward_o(owords)
        scores = torch.matmul(ovectors, ivectors.t())
        probs = scores.squeeze().sigmoid()
        probs[word2idx[word]] = 0
        probs[word2idx['<unk>']] = 0
        avg_prob = torch.mean(probs)
        decoder_probs = torch.ones(decoder_dict_size, dtype=torch.float32).cuda() * avg_prob
        decoder_probs[sgns_dict_map] = probs
        topk_prob, topk_id = torch.topk(decoder_probs, 10)
        print('related to', word)
        print([model_dict[id_] for id_ in topk_id])
        return decoder_probs

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')

    #for inputs in buffered_read(args.buffer_size):
    while True:
        s = input('Input pun word and alternative word: ')
        # Get related words
        #pun_word, alter_word = s.split()
        #if not (pun_word in word2idx and alter_word in word2idx):
        #    print('Unknown word to Skipgram. continue...')
        #    continue
        #pun_word_neighbor_prob = get_related_prob(pun_word)
        #alter_word_neighbor_prob = get_related_prob(alter_word)
        #inputs = [alter_word]
        pun_word_neighbor_prob = None
        alter_word_neighbor_prob = None

        if args.switch:
            pun_word, alter_word = s.split()
            inputs = [pun_word]
            for batch, batch_indices in make_batches(inputs, args, src_dict, models[0].max_positions()):
                results = process_batch(batch, pun_word_neighbor_prob, None)
                hypo = results[0].hypos[0]
                print(hypo)
            inputs = [alter_word]
            tgt_str = hypo.split('\t')[-1]
            tgt_tokens = tgt_str.split()
            tokens = []
            for t in tgt_tokens:
                if t == pun_word:
                    break
                else:
                    tokens.append(t)
            tgt_str = ' '.join(tokens)
            for batch, batch_indices in make_batches(inputs, args, src_dict, models[0].max_positions(), tgt_str=tgt_str, tgt_dict=tgt_dict):
                results = process_batch(batch, alter_word_neighbor_prob, None)
                hypo = results[0].hypos[0]
                print(hypo)
        elif args.normal:
            inputs = [s]
            indices = []
            results = []
            for batch, batch_indices in make_batches(inputs, args, src_dict, models[0].max_positions()):
                indices.extend(batch_indices)
                results += process_batch(batch, pun_word_neighbor_prob, alter_word_neighbor_prob)

            for i in np.argsort(indices):
                result = results[i]
                print(result.src_str)
                for hypo, align in zip(result.hypos, result.alignments):
                    print(hypo)
                    #print(align)


if __name__ == '__main__':
    parser = options.get_generation_parser(interactive=True)
    parser.add_argument('--lm')
    parser.add_argument('--skipgram-model')
    parser.add_argument('--skipgram-data')
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--switch', action='store_true')
    args = options.parse_args_and_arch(parser)
    main(args)
