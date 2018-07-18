#!/bin/bash

data=roc/kw-story
model=lstm

# vanilla seq2seq
ckpt=lstm
output=logs/$data/$ckpt.train.log
echo "data=$data model=$model ckpt=$ckpt
output saved at $output"
sbatch --job-name $ckpt `readlink -f scripts/run.sbatch` \
    $data $model foo $ckpt $output

exit 0


# interpolate with pretrained wikitext LM
for fusion in prob input output; do
    ckpt=lstm-wiki-$fusion
    output=logs/$data/$ckpt.train.log
    echo "data=$data model=$model fusion=$fusion ckpt=$ckpt
    output saved at $output"
    sbatch --job-name $ckpt `readlink -f scripts/run.sbatch` \
        $data $model $fusion $ckpt $output
    echo ""
done
