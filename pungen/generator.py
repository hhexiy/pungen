import torch
import argparse
import numpy as np
from collections import namedtuple

from fairseq.data.dictionary import Dictionary
from fairseq.data import EditDataset
from fairseq.edit_sequence_generator import SequenceGenerator as EditSequenceGenerator
from fairseq.sequence_generator import SequenceGenerator
from fairseq import options, tasks, utils, tokenizer, data

from .wordvec.model import Word2Vec, SGNS
from .wordvec.generate import SkipGram
from .pretrained_wordvec import Glove
from .utils import get_lemma

import logging
logger = logging.getLogger('pungen')

import spacy
from spacy.symbols import ORTH, LEMMA, POS, TAG
nlp = spacy.load('en_core_web_sm', disable=['ner'])
# Don't tokenize these entities
for ent in ('<org>', '<person>', '<date>', '<time>', '<gpe>', '<norp>',
        '<loc>', '<percent>', '<money>', '<ordinal>', '<quantity>', '<cardinal>',
        '<language>', '<law>', '<event>', '<product>', '<fac>'):
    special_case = [{ORTH: ent, LEMMA: ent, POS: 'NOUN'}]
    nlp.tokenizer.add_special_case(ent, special_case)

Batch = namedtuple('Batch', 'srcs tokens lengths')

class RetrieveSwapGenerator(object):
    def __init__(self, retriever, scorer):
        self.retriever = retriever
        self.scorer = scorer

    def generate(self, alter_word, pun_word, k=20, ncands=500, ntemps=10):
        """
        Args:
            k (int): number of topic words returned by skipgram (before filtering)
            ncands (int): number of sentences returned by retriever (before filtering)
            ntemps (int): number of templates returned by retriever (after filtering)
        """
        templates = self.retriever.retrieve_pun_template(alter_word, num_cands=ncands, num_templates=ntemps)
        results = []
        for template in templates:
            pun_sent = template.replace_keyword(pun_word)
            pun_word_id = template.keyword_id
            score = self.scorer.score(pun_sent, pun_word_id, alter_word)
            r = {'output': pun_sent, 'score': score, 'template-id': template.id}
            results.append(r)
        return results

class RetrieveGenerator(object):
    def __init__(self, retriever, scorer):
        self.retriever = retriever
        self.scorer = scorer

    def generate(self, alter_word, pun_word, k=20, ncands=500, ntemps=10):
        """
        Args:
            k (int): number of topic words returned by skipgram (before filtering)
            ncands (int): number of sentences returned by retriever (before filtering)
            ntemps (int): number of templates returned by retriever (after filtering)
        """
        templates = self.retriever.retrieve_pun_template(pun_word, num_cands=ncands, num_templates=ntemps)
        results = []
        for template in templates:
            pun_sent = template.tokens
            pun_word_id = template.keyword_id
            score = self.scorer.score(pun_sent, pun_word_id, alter_word)
            r = {'output': pun_sent, 'score': score, 'template-id': template.id}
            results.append(r)
        return results

class RulebasedGenerator(object):
    def __init__(self, retriever, neighbor_predictor, type_recognizer, scorer, dist_to_pun=5):
        self.retriever = retriever
        self.neighbor_predictor = neighbor_predictor
        self.scorer = scorer
        self.dist_to_pun = dist_to_pun
        self.type_recognizer = type_recognizer

    def _delete_candidates(self, parsed_sent, pun_word_id):
        noun_ids = [i for i in range(max(1, pun_word_id - self.dist_to_pun))
                if parsed_sent[i].pos_ in ('NOUN', 'PROPN', 'PRON')]
                    #and (parsed_sent[i].dep_.startswith('nsubj') or
                    #parsed_sent[i].dep_ == 'ROOT')]
        return noun_ids

    def delete_words(self, templates):
        parsed_sents = nlp.pipe([' '.join(t.tokens) for t in templates])
        pun_word_ids = [t.keyword_id for t in templates]
        for parsed_sent, pun_word_id in zip(parsed_sents, pun_word_ids):
            ids = self._delete_candidates(parsed_sent, pun_word_id)
            if not ids:
                yield None, None
            else:
                del_word_id = ids[0]
                del_span = (del_word_id, del_word_id+1)
                yield del_span, del_word_id

    def get_topic_words(self, pun_word, del_word, context=None, tags=('NOUN', 'PROPN'), k=20):
        del_word = get_lemma(del_word)

        # type constraints
        types = self.type_recognizer.get_type(del_word, 'noun')
        if len(types) == 0:
            logger.debug('FAIL: deleted word "{}" has unknown type.'.format(del_word))
            return []

        words = self.neighbor_predictor.predict_neighbors(pun_word, k=k, masked_words=[del_word])

        # type constraints
        new_words = []
        for w in words:
            if self.type_recognizer.is_types(w, types, 'noun'):
                new_words.append(w)
        words = new_words
        if len(words) == 0:
            logger.debug('FAIL: no topic words has same type as {}.'.format(del_word))

        return words

    def rewrite(self, pun_sent, delete_span, insert_word, pun_word_id):
        """
        Return:
            s (list): rewritten sentence
            pun_word_id (int)
        """
        s = pun_sent[:delete_span[0]] + [insert_word] + pun_sent[delete_span[1]:]
        # pun_word_id is not changed due to rewrite
        yield s, pun_word_id

    def generate(self, alter_word, pun_word, k=20, ncands=500, ntemps=10):
        """
        Args:
            k (int): number of topic words returned by skipgram (before filtering)
            ncands (int): number of sentences returned by retriever (before filtering)
            ntemps (int): number of templates returned by retriever (after filtering)
        """
        templates = self.retriever.retrieve_pun_template(alter_word, num_cands=ncands, num_templates=ntemps)

        results = []
        for i, (template, (delete_span_ids, delete_word_id)) in enumerate(zip(templates, self.delete_words(templates))):
            #logger.debug(str(template))
            alter_sent = template.tokens
            pun_sent = template.replace_keyword(pun_word)
            pun_word_id = template.keyword_id

            r = {}
            r['template-id'] = template.id
            r['template'] = alter_sent
            r['retrieved'] = ' '.join(list(pun_sent))

            if not delete_word_id:
                #logger.debug('nothing to delete')
                results.append(r)
                continue
            r['deleted'] = alter_sent[delete_word_id]

            topic_words = self.get_topic_words(pun_word, k=k, del_word=alter_sent[delete_word_id], context=pun_sent)
            if not topic_words:
                results.append(r)
                continue
            #print(' '.join(alter_sent))
            #print(r['deleted'])
            #print(topic_words)
            #print()
            #continue

            for w in topic_words:
                for s, new_pun_word_id in self.rewrite(pun_sent, delete_span_ids, w, pun_word_id):
                    if s is None:
                        continue
                    alter_word = alter_sent[pun_word_id]
                    score = self.scorer.score(s, new_pun_word_id, alter_word)
                    r = dict(r)
                    r.update({'inserted': w, 'output': s, 'score': score})
                    results.append(r)

        return results

class KeywordsGenerator(object):
    def __init__(self, retriever, neighbor_predictor):
        self.retriever = retriever
        self.neighbor_predictor = neighbor_predictor

    def _get_local_context(self, template, word, size=2):
        start = max(0, template.keyword_id - size)
        end = min(len(template), template.keyword_id + size + 1)
        tokens = template.replace_keyword(word)
        return tokens[start:end]

    def generate(self, alter_word, pun_word, k=20, ncands=500, ntemps=10):
        templates = self.retriever.retrieve_pun_template(alter_word, num_cands=ncands, num_templates=ntemps)
        local_contexts = [self._get_local_context(t, pun_word) for t in templates]
        words = self.neighbor_predictor.predict_neighbors(pun_word, k=k)
        words = [w for w in words if w.isalnum()]
        r = {
                'local_contexts': local_contexts,
                'topic_words': words,
                }
        results = [r]
        return results

class NeuralSLGenerator(object):
    def __init__(self, args):
        task, model, model_args = self.load_model(args)
        use_cuda = torch.cuda.is_available() and not args.cpu
        tgt_dict = task.target_dictionary

        generator = SequenceGenerator(
            [model], tgt_dict, beam_size=args.beam, stop_early=(not args.no_early_stop),
            normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
            unk_penalty=args.unkpen, sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
            minlen=args.min_len,
        )

        if use_cuda:
            generator.cuda()

        self.generator = generator
        self.task = task
        self.model = model
        self.use_cuda = use_cuda
        self.args = args
        self.model_args = model_args

    def load_model(self, args):
        #args = argparse.Namespace(data=data_path, path=model_path, cpu=cpu, task='edit')
        use_cuda = torch.cuda.is_available() and not args.cpu
        task = tasks.setup_task(args)
        logger.info('loading model from {}'.format(args.path))
        overrides = {'encoder_embed_path': None, 'decoder_embed_path': None}
        models, model_args = utils.load_ensemble_for_inference(args.path.split(':'), task, overrides)
        return task, models[0], model_args

    def make_batches(self, lines, args, src_dict, max_positions):
        tokens = [
            tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
            for src_str in lines
        ]
        lengths = np.array([t.numel() for t in tokens])
        itr = data.EpochBatchIterator(
            dataset=data.LanguagePairDataset(tokens, lengths, src_dict),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=max_positions,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            yield Batch(
                srcs=[lines[i] for i in batch['id']],
                tokens=batch['net_input']['src_tokens'],
                lengths=batch['net_input']['src_lengths'],
            ), batch['id']

    def generate(self, alter_word, pun_word):
        sents = self._generate(alter_word, pun_word)
        results = []
        for s in sents:
            r = {'output': s}
            results.append(r)
        return results

    def _generate(self, alter_word, pun_word):
        src_dict = self.task.source_dictionary
        max_positions = self.model.max_positions()
        lines = ['{} {}'.format(pun_word, alter_word)]
        for batch, batch_indices in self.make_batches(lines, self.args, src_dict, max_positions):
            tokens = batch.tokens
            lengths = batch.lengths

            if self.use_cuda:
                tokens = tokens.cuda()
                lengths = lengths.cuda()

            outputs = self.generator.generate(tokens, lengths, maxlen=int(self.args.max_len_a * tokens.size(1) + self.args.max_len_b))
            for hypos in outputs:
                return self.make_results(hypos, self.args)

    def make_results(self, hypos, args):
        results = []
        tgt_dict = self.task.target_dictionary
        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=None,
                alignment=None,
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            #results.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            #results.append('{}'.format(hypo_str))
            results.append(hypo_str.split())
        return results


class NeuralCombinerGenerator(RulebasedGenerator):
    def __init__(self, retriever, neighbor_predictor, type_recognizer, scorer, dist_to_pun, args):
        super().__init__(retriever, neighbor_predictor, type_recognizer, scorer, dist_to_pun)

        task, model, model_args = self.load_model(args)

        use_cuda = torch.cuda.is_available() and not args.cpu
        tgt_dict = task.target_dictionary
        Generator = EditSequenceGenerator if model_args.insert != 'none' and model_args.combine == 'embedding' else SequenceGenerator
        generator = Generator(
            [model], tgt_dict, beam_size=args.beam, stop_early=(not args.no_early_stop),
            normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
            unk_penalty=args.unkpen, sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
            minlen=args.min_len,
        )

        if use_cuda:
            generator.cuda()

        self.generator = generator
        self.task = task
        self.model = model
        self.use_cuda = use_cuda
        self.args = args
        self.model_args = model_args

    def load_model(self, args):
        use_cuda = torch.cuda.is_available() and not args.cpu
        task = tasks.setup_task(args)
        logger.info('loading edit model from {}'.format(args.path))
        models, model_args = utils.load_ensemble_for_inference(args.path.split(':'), task)
        return task, models[0], model_args

    def make_batches(self, templates, deleted_words, src_dict, max_positions):
        temps = [
            tokenizer.Tokenizer.tokenize(temp, src_dict, add_if_not_exist=False, tokenize=lambda x: x).long()
            for temp in templates
        ]
        deleted = [
            tokenizer.Tokenizer.tokenize(word, src_dict, add_if_not_exist=False, tokenize=lambda x: x).long()
            for word in deleted_words
        ]
        inputs = [
                {'template': temp, 'deleted': dw} for
                temp, dw in zip(temps, deleted)
                ]
        lengths = np.array([t['template'].numel() for t in inputs])
        dataset = EditDataset(inputs, lengths, src_dict, insert=self.model_args.insert, combine=self.model_args.combine)
        itr = self.task.get_batch_iterator(
                dataset=dataset,
                max_tokens=100,
                max_sentences=5,
                max_positions=max_positions,
            ).next_epoch_itr(shuffle=False)
        return itr

    def make_results(self, hypos, args):
        results = []
        tgt_dict = self.task.target_dictionary
        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=None,
                alignment=None,
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            results.append(hypo_str.split())
        return results

    def delete_words(self, templates):
        for template, (del_span, del_word_id) in zip(templates, super().delete_words(templates)):
            if del_span is None:
                yield None, None
            else:
                # TODO: don't delete content words
                start = max(0, del_span[0] - 1)
                end = min(len(template), del_span[0] + 2)
                yield (start, end), del_word_id

    def get_topic_words(self, pun_word, del_word=None, tags=('NOUN', 'PROPN'), k=20, context=None):
        return super().get_topic_words(pun_word, del_word=del_word, tags=tags, k=k, context=context)

    def rewrite(self, pun_sent, delete_span, insert_word, pun_word_id):
        start, end = delete_span
        template = pun_sent[:start] + ['<placeholder>'] + pun_sent[end:]
        pun_word = pun_sent[pun_word_id]
        deleted = [insert_word]
        if self.model_args.insert == 'deleted' and not insert_word in self.task.source_dictionary.indices:
            logger.debug('Inserted word {} is OOV'.format(insert_word))
            yield None, None
        logger.debug('template: {}'.format(' '.join(template)))
        logger.debug('deleted: {}'.format(' '.join(pun_sent[start:end])))
        logger.debug('insert: {}'.format(' '.join(deleted)))
        results = self._generate([template], [deleted])
        for s in results:
            logger.debug('generated: {}'.format(' '.join(s)))
            r = pun_sent[:start] + s + pun_sent[end:]
            pun_id = pun_word_id + (len(s) - (end - start))
            yield r, pun_id

    def _generate(self, templates, deleted_words):
        src_dict = self.task.source_dictionary
        max_positions = self.model.max_positions()
        insert = self.model_args.insert
        for batch in self.make_batches(templates, deleted_words, src_dict, max_positions):
            src_tokens = batch['net_input']['src_tokens']
            src_lengths = batch['net_input']['src_lengths']
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
            encoder_input = {'src_tokens': src_tokens, 'src_lengths': src_lengths}
            outputs = self.generator.generate(encoder_input, maxlen=int(self.args.max_len_a * src_tokens.size(1) + self.args.max_len_b))
            # TODO: batches
            for hypos in outputs:
                return self.make_results(hypos, self.args)
                #for r in self.make_results(hypos, self.args):
                #    print(r)

    def test_generate(self):
        templates = ['<placeholder> going to die'.split()]
        deleted_words = [['painter']]
        related_words = [['die']]
        outputs = self._generate(templates, deleted_words, related_words, self.args.insert)
        for s in outputs:
            print(s)


if __name__ == '__main__':
    from .utils import logging_config
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    logging_config()

    generator = NeuralCombinerGenerator(None, None, None, args)
    generator.test_generate()
