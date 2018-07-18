## Requirements
- Pytorch 0.4
- Python 3.6
- Install `fairseq` from `https://github.com/hhexiy/fairseq/tree/me` (the `me` branch)

## Training
Training commands are in `Makefile`.

- Put raw data in `data/roc/kw-story/{train,valid,test}.{src,tgt}`
- `make preprocess-fairseq data=roc/kw-story` will prepare processed data in `data/roc/kw-story/bin`
- `make train data=roc/kw-story model=lstm ckpt=lstm`
