#!/bin/bash

data=onebillion/edit
combine=embedding

for insert in deleted related; do
    if [ $insert == "none" ] || [ $combine == "token" ]; then
        model=lstm
    else
        model=edit-lstm
    fi
    ckpt=$insert-$combine
    output=logs/$data/$ckpt/train.log
    if [ ! -d logs/$data/$ckpt ]; then
        mkdir -p logs/$data/$ckpt
    fi
    echo "data=$data model=$model insert=$insert combine=$combine ckpt=$ckpt
    output saved at $output"
    sbatch --job-name $ckpt `readlink -f scripts/run.sbatch` \
        $data $model $insert $combine $ckpt $output
done

# vanilla seq2seq
#ckpt=lstm
#output=logs/$data/$ckpt.train.log
#echo "data=$data model=$model ckpt=$ckpt
#output saved at $output"
#sbatch --job-name $ckpt `readlink -f scripts/run.sbatch` \
#    $data $model foo $ckpt $output
#
#exit 0

# interpolate with pretrained wikitext LM
#for fusion in prob input output; do
#    ckpt=lstm-wiki-$fusion
#    output=logs/$data/$ckpt.train.log
#    echo "data=$data model=$model fusion=$fusion ckpt=$ckpt
#    output saved at $output"
#    sbatch --job-name $ckpt `readlink -f scripts/run.sbatch` \
#        $data $model $fusion $ckpt $output
#    echo ""
#done
