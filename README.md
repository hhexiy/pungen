# Pun Generation with Surprise
This repo contains code and data for the paper
[Pun Generation with Surprise](https://arxiv.org/abs/1904.06828).

## Requirements
- Python 3.6
- Pytorch 0.4
```
conda install pytorch=0.4.0 torchvision -c pytorch
```
- Fairseq(-py)
```
git clone -b pungen https://github.com/hhexiy/fairseq.git
cd fairseq
pip install -r requirements.txt
python setup.py build develop
```
- Pretrained WikiText-103 model from Fairseq
```
curl --create-dirs --output models/wikitext/model https://dl.fbaipublicfiles.com/fairseq/models/wiki103_fconv_lm.tar.bz2
tar xjf models/wikitext/model -C models/wikitext
rm models/wikitext/model
```

## Training
### Word relatedness model
We approximate relatedness between a pair of words with a long-distance skip-gram model trained on BookCorpus sentences.
The original BookCorpus data is parsed by `scripts/preprocess_raw_text.py`
and you can see the sample file in `sample_data/bookcorpus/raw/train.txt`.

Preprocess bookcorpus data:
```
python -m pungen.wordvec.preprocess --data-dir data/bookcorpus/skipgram \
	--corpus data/bookcorpus/raw/train.txt \
	--min-dist 5 --max-dist 10 --threshold 80 \
	--vocab data/bookcorpus/skipgram/dict.txt
```

Train skip-gram model:
```
python -m pungen.wordvec.train --weights --cuda --data data/bookcorpus/skipgram/train.bin \
    --save_dir models/bookcorpus/skipgram \
    --mb 3500 --epoch 15 \
    --vocab data/bookcorpus/skipgram/dict.txt
```

### Edit model
The edit model takes a word and a template (masked sentence) and combine the two coherently.

Preprocess data:
```
for split in train valid; do \
	PYTHONPATH=. python scripts/make_src_tgt_files.py -i data/bookcorpus/raw/$split.txt \
        -o data/bookcorpus/edit/$split --delete-frac 0.5 --window-size 2 --random-window-size; \
done

python -m pungen.preprocess --source-lang src --target-lang tgt \
	--destdir data/bookcorpus/edit/bin/data --thresholdtgt 80 --thresholdsrc 80 \
	--validpref data/bookcorpus/edit/valid \
	--trainpref data/bookcorpus/edit/train \
	--workers 8
```

Training:
```
python -m pungen.train data/bookcorpus/edit/bin/data -a lstm \
    --source-lang src --target-lang tgt \
    --task edit --insert deleted --combine token \
    --criterion cross_entropy \
    --encoder lstm --decoder-attention True \
    --optimizer adagrad --lr 0.01 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --clip-norm 5 --max-epoch 50 --max-tokens 7000 --no-epoch-checkpoints \
    --save-dir models/bookcorpus/edit/deleted --no-progress-bar --log-interval 5000
```

### Retriever
Build a sentence retriever based on Bookcorpus.
The input should have a tokenized sentence per line.
```
python -m pungen.retriever --doc-file data/bookcorpus/raw/sent.tokenized.txt \
    --path models/bookcorpus/retriever.pkl --overwrite
```

## Analyze what makes a pun funny
Compute correlation between local-global suprise scores and human funniness ratings.
We provide our annotated dataset in `data/funniness_annotation`:
- `analysis_pun_scores.txt`: sentences annotated with funniness scores from 1 to 5.
- `analysis_zscored_pun_scores.txt`: the same data where scores are standardized for each annotator.
```
python eval_scoring_func.py --human-eval data/funniness_annotation/analysis_zscored_pun_scores.txt \
	--lm-path models/wikitext/wiki103.pt --word-counts-path models/wikitext/dict.txt \
    --skipgram-model data/bookcorpus/skipgram/dict.txt \
                     models/bookcorpus/skipgram/sgns-e15.pt \
    --outdir results/pun-analysis/analysis_zscored \
    --features grammar ratio --analysis --ignore-cache  
```

## Generate puns
We generate puns given a pair of pun word and alternative word.
We support pun generation with the following methods specified by the `system` argument.
- `rule`: the SURGEN method described in the paper 
- `rule+neural`: in the last step of SURGEN, use a neural combiner to edit the topic words
- `retrieve`: retrieve a sentence containing the pun word
- `retrieve+swap`: retrieve a sentence containing the alternative word and replace it with the pun word
For arguments controlling the neural generator (e.g., `--beam`, `--nbest`), see `fairseq.options`.
All results and logs are saved in `outdir`.
```
python generate_pun.py data/bookcorpus/edit/bin/data \
	--path models/bookcorpus/edit/delete/checkpoint_best.pt \
	--beam 20 --nbest 1 --unkpen 100 \
	--system rule --task edit \
	--retriever-model models/bookcorpus/retriever.pkl --doc-file data/bookcorpus/raw/sent.tokenized.txt \
	--lm-path models/wikitext/wiki103.pt --word-counts-path models/wikitext/dict.txt \
	--skipgram-model data/bookcorpus/skipgram/dict.txt models/bookcorpus/skipgram/sgns-e15.pt \
	--num-candidates 500 --num-templates 100 \
	--num-topic-word 100 --type-consistency-threshold 0.3 \
	--pun-words data/semeval/hetero/dev.json \
	--outdir results/semeval/hetero/dev/rule \
	--scorer random \
	--max-num-examples 100
```

## Reference
If you use the annotated SemEval pun dataset, please cite our paper:
```
@inproceedings{he2019pun,
    title={Pun Generation with Surprise},
    author={He He and Nanyun Peng and Percy Liang},
    booktitle={North American Association for Computational Linguistics (NAACL)},
    year={2019}
}
```
