#!/bin/bash

#SBATCH --partition=jag-lo
#SBATCH --mem=24g
#SBATCH --gres=gpu:1
#SBATCH --nodelist=jagupard11

HOME=/u/scr/hehe
data=$1
model=$2
insert=$3
combine=$4
ckpt=$5
output=$6

cd $HOME/story
source $HOME/anaconda/bin/activate fairseq
make train data=$data model=$model insert=$insert combine=$combine ckpt=$ckpt &> $output
