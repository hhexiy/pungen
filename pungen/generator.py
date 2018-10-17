import torch
import argparse
import numpy as np

from fairseq.data.dictionary import Dictionary
from fairseq.data import EditDataset
from fairseq.edit_sequence_generator import SequenceGenerator as EditSequenceGenerator
from fairseq.sequence_generator import SequenceGenerator
from fairseq import options, tasks, utils, tokenizer, data

from .wordvec.model import Word2Vec, SGNS
from .wordvec.generate import SkipGram
from .pretrained_wordvec import Glove

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

import spacy
from spacy.symbols import ORTH, LEMMA, POS, TAG
nlp = spacy.load('en_core_web_sm', disable=['ner'])
# Don't tokenize these entities
for ent in ('<org>', '<person>', '<date>', '<time>', '<gpe>', '<norp>',
        '<loc>', '<percent>', '<money>', '<ordinal>', '<quantity>', '<cardinal>',
        '<language>', '<law>', '<event>', '<product>', '<fac>'):
    special_case = [{ORTH: ent, LEMMA: ent, POS: 'NOUN'}]
    nlp.tokenizer.add_special_case(ent, special_case)


class RulebasedGenerator(object):
    def __init__(self, retriever, neighbor_predictor, typer_recognizer, scorer, beginning_portion=0.3):
        self.retriever = retriever
        self.neighbor_predictor = neighbor_predictor
        self.scorer = scorer
        self.beginning_portion = beginning_portion
        self.typer_recognizer = typer_recognizer

    # TODO: update neural version
    def delete_words(self, sents, pun_word_ids):
        parsed_sents = nlp.pipe([' '.join(s) for s in sents])
        for parsed_sent, pun_word_id in zip(parsed_sents, pun_word_ids):
            n = max(0, int(self.beginning_portion * len(parsed_sent)))
            noun_ids = [i for i in range(min(n+1, pun_word_id))
                    if parsed_sent[i].pos_ in ('NOUN', 'PROPN', 'PRON') and
                        (parsed_sent[i].dep_.startswith('nsubj') or
                        parsed_sent[i].dep_ == 'ROOT')]
            if not noun_ids:
                yield None, None
            else:
                del_word = noun_ids[0]
                del_span = [del_word]
                yield del_span, del_word


    def _delete_words(self, alter_sent, pun_word_id):
        N = len(alter_sent)
        # TODO: make this an argument (saved in model config)
        n = max(0, int(0.3 * N))
        parsed_alter_sent = nlp(' '.join(alter_sent))
        noun_ids = [i for i in range(min(n+1, pun_word_id))
                if parsed_alter_sent[i].pos_ in ('NOUN', 'PROPN', 'PRON') and
                True]
                #(parsed_alter_sent[i].dep_.startswith('nsubj') or
                #    parsed_alter_sent[i].dep_ == 'ROOT')]
        if not noun_ids:
            return None, None
        return [noun_ids[0]], noun_ids[0]

    def get_topic_words(self, pun_word, del_word=None, context=None, tags=('NOUN', 'PROPN'), k=20):
        # Get sentences in similar context
        #ids = self.retriever.query(' '.join(context), k=500)
        #sim_sents = [self.retriever.docs[id_].split() for id_ in ids]
        #cands = set()
        #for s in sim_sents:
        #    cands.update(s)
        cands = None
        words = self.neighbor_predictor.predict_neighbors(pun_word, k=k, sim_words=None, cands=cands)

        if not words:
            return []

        if del_word is not None:
            del_word = nlp(del_word)[0].lemma_

        # POS constraints
        new_words = []
        parsed_words = nlp.pipe(words)
        for w in parsed_words:
            w_ = w[0]
            if not w_.lemma_ in (pun_word, del_word) and w_.pos_ in tags:
                new_words.append(w_.lemma_)
        words = new_words

        # type constraints
        new_words = []
        types = self.type_recognizer.get_type(del_word)
        for w in words
            for type_ in types:
                if self.type_recognizer.is_type(w, type_):
                    new_words.append(w)
        words = new_words

        return words

    def rewrite(self, pun_sent, delete_span_ids, insert_word, pun_word_id):
        """
        Return:
            s (list): rewritten sentence
            pun_word_id (int)
        """
        s = list(pun_sent)
        delete_id = delete_span_ids[0]
        s[delete_id] = insert_word
        yield s, pun_word_id

    def generate(self, alter_word, pun_word, k=20, ncands=500):
        alter_sents, pun_sents, pun_word_ids, alter_ori_sents = self.retriever.retrieve_pun_template(pun_word, alter_word, num_cands=ncands, num_templates=10)
        results = []
        for i, (alter_sent, pun_sent, pun_word_id, alter_ori_sent, (delete_span_ids, delete_word_id)) in enumerate(zip(alter_sents, pun_sents, pun_word_ids, alter_ori_sents, self.delete_words(alter_sents, pun_word_ids))):
            r = {}
            r['template-id'] = i
            r['retrieved'] = alter_sent
            r['retrieved-ori'] = alter_ori_sent
            #delete_span_ids, delete_word_id  = self.delete_words(alter_sent, pun_word_id)
            if not delete_word_id:
                r['deleted'] = None
                results.append(r)
                continue
            r['deleted'] = alter_sent[delete_word_id]
            topic_words = self.get_topic_words(pun_word, k=k, del_word=alter_sent[delete_word_id], context=pun_sent)
            if not topic_words:
                r['topic_words'] = None
                results.append(r)
                continue
            #print(' '.join(alter_sent))
            #print(r['deleted'])
            #print(topic_words)
            #print()
            #continue
            for w in topic_words:
                for s, new_pun_word_id in self.rewrite(pun_sent, delete_span_ids, w, pun_word_id):
                    score = self.scorer.score(alter_sent, pun_word_id, s, new_pun_word_id, 2)
                    #score = 1.
                    r = dict(r)
                    r.update({'inserted': w, 'output': s, 'score': score})
                    results.append(r)
        return results

class NeuralGenerator(RulebasedGenerator):
    def __init__(self, retriever, neighbor_predictor, scorer, args):
        super().__init__(retriever, neighbor_predictor, scorer)

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
        #args = argparse.Namespace(data=data_path, path=model_path, cpu=cpu, task='edit')
        use_cuda = torch.cuda.is_available() and not args.cpu
        task = tasks.setup_task(args)
        print('| loading edit model from {}'.format(args.path))
        models, model_args = utils.load_ensemble_for_inference(args.path.split(':'), task)
        return task, models[0], model_args

    def make_batches(self, templates, deleted_words, related_words, src_dict, max_positions):
        temps = [
            tokenizer.Tokenizer.tokenize(temp, src_dict, add_if_not_exist=False, tokenize=lambda x: x).long()
            for temp in templates
        ]
        deleted = [
            tokenizer.Tokenizer.tokenize(word, src_dict, add_if_not_exist=False, tokenize=lambda x: x, append_eos=False).long()
            for word in deleted_words
        ]
        related = [
            tokenizer.Tokenizer.tokenize(word, src_dict, add_if_not_exist=False, tokenize=lambda x: x, append_eos=False).long()
            for word in related_words
        ]
        inputs = [
                {'template': temp, 'deleted': dw, 'related': rw} for
                temp, dw, rw in zip(temps, deleted, related)
                ]
        lengths = np.array([t['template'].numel() for t in inputs])
        dataset = EditDataset(inputs, lengths, src_dict, insert=self.model_args.insert, combine=self.model_args.combine)
        itr = data.EpochBatchIterator(
                dataset=dataset,
                max_tokens=6000,
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
            #results.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            #results.append('{}'.format(hypo_str))
            results.append(hypo_str.split())
        return results

    def delete_words(self, alter_sent, pun_word_id):
        parsed_alter_sent = nlp(' '.join(alter_sent))
        noun_ids = [i for i in range(pun_word_id)
                if parsed_alter_sent[i].pos_ in ('NOUN', 'PROPN', 'PRON') and
                True]
                #(parsed_alter_sent[i].dep_.startswith('nsubj') or
                #    parsed_alter_sent[i].dep_ == 'ROOT')]
        if not noun_ids:
            return None, None

        id_ = noun_ids[0]
        deleted = [id_]
        if id_ - 1 >= 0: #and not parsed_alter_sent[id_ - 1].pos_ in ('NOUN', 'VERB'):
            deleted.insert(0, id_ - 1)
        if id_ + 1 < len(alter_sent): #and not parsed_alter_sent[id_ + 1].pos_ in ('NOUN', 'VERB') :
            deleted.append(id_ + 1)
        return deleted, id_

    def get_topic_words(self, pun_word, del_word=None, tags=('NOUN', 'PROPN'), k=20, context=None):
        if self.model_args.insert == 'related':
            return ['dummy']
        else:
            return super().get_topic_words(pun_word, del_word=del_word, tags=tags, k=k, context=context)

    def rewrite(self, pun_sent, delete_ids, insert_word, pun_word_id):
        template = pun_sent[:delete_ids[0]] + ['<placeholder>'] + pun_sent[delete_ids[-1]+1:]
        pun_word = pun_sent[pun_word_id]
        related = [pun_word]
        deleted = [insert_word]
        if self.model_args.insert == 'deleted' and not insert_word in self.task.source_dictionary.indices:
            logging.info('Inserted word {} is OOV'.format(insert_word))
            return
        results = self._generate([template], [deleted], [related])
        for r in results:
            pun_ids = [i for i, w in enumerate(r) if w == pun_word]
            if not pun_ids:
                logging.debug('changed pun. continue.')
                continue
            yield r, pun_ids[0]

    def _generate(self, templates, deleted_words, related_words):
        src_dict = self.task.source_dictionary
        max_positions = self.model.max_positions()
        insert = self.model_args.insert if self.model_args.combine != 'token' else 'none'
        for batch in self.make_batches(templates, deleted_words, related_words, src_dict, max_positions):
            src_tokens = batch['net_input']['src_tokens']
            src_lengths = batch['net_input']['src_lengths']
            if insert != 'none':
                src_insert = batch['net_input']['src_insert']
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if insert != 'none':
                    src_insert = src_insert.cuda()
            if insert != 'none':
                outputs = self.generator.generate(src_tokens, src_lengths, src_insert, maxlen=int(self.args.max_len_a * src_tokens.size(1) + self.args.max_len_b))
            else:
                outputs = self.generator.generate(src_tokens, src_lengths, maxlen=int(self.args.max_len_a * src_tokens.size(1) + self.args.max_len_b))
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
    parser = options.get_generation_parser(interactive=True)
    # TODO: read from saved model
    parser.add_argument('--insert')
    args = options.parse_args_and_arch(parser)
    generator = NeuralGenerator(None, None, None, args)
    generator.test_generate()
