## Requirements
- Pytorch 0.4
- Python 3.6
- Install `fairseq` from `https://github.com/hhexiy/fairseq/tree/me` (the `me` branch)

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
python src/retriever.py --doc-file <train_data> --lm-path <lm_model> --path <output> --interactive
```
- train_data: tokenized sentences, one sentence per line.
- lm_model: pretrained pytorch model, `https://github.com/pytorch/fairseq/tree/master/examples/language_model`.
- output: output path to save the retriever. Next time we will directly load from it.
