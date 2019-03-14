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
We approximate relatedness between a pair of words with a long-distance skip-gram model trained on bookcorpus sentences.

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
    --clip-norm 5 --max-epoch 50 --max-tokens 7000 \
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
```
python eval_scoring_func.py --human-eval data/funniness_annotation/analysis_zscored_pun_scores.txt \
	--lm-path models/wikitext/wiki103.pt --word-counts-path models/wikitext/dict.txt \
    --skipgram-model data/bookcorpus/skipgram/dict.txt \
                     models/bookcorpus/skipgram/sgns-e15.pt \
    --outdir results/pun-analysis/analysis_zscored \
    --features grammar ratio --analysis --ignore-cache  
```

## Deprecated below

## Training
Training commands are in `Makefile`.

- Put raw data in `data/roc/kw-story/{train,valid,test}.{src,tgt}`
- `make preprocess-fairseq data=roc/kw-story` will prepare processed data in `data/roc/kw-story/bin`
- `make train data=roc/kw-story model=lstm ckpt=lstm`

## Retrieve
The retriever vectorize sentences by TFIDF scores (fitted on a training corpus).
Given a query sentence, the retriever returns the top K sentences in the training corpus with the highest similarity scores,
where similarity is computed by the dot product between TFIDF vectors.

```
python -m pungen.retriever --doc-file <train_data> --path <output> --overwrite 
```
- train_data: tokenized sentences, one sentence per line. Tokenized 1B sentences are here: `https://worksheets.codalab.org/bundles/0x364840a62d6b495794354b2f9e849472/`.
- lm_model: pretrained pytorch model, `https://github.com/pytorch/fairseq/tree/master/examples/language_model`.
You can download it here: `https://worksheets.codalab.org/bundles/0x08e710512cdf471c8f66f19e70f910ef/`.
- output: output path to save the retriever. Next time we will directly load from it.
