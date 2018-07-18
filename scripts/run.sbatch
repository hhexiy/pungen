#!/bin/bash

#SBATCH --partition=jag-hi
#SBATCH --mem=12g
#SBATCH --gres=gpu:1

HOME=/u/scr/hehe
data=$1
model=$2
fusion=$3
ckpt=$4
output=$5

cd $HOME/story
source $HOME/anaconda/bin/activate fairseq
make train data=$data model=$model fusion=$fusion ckpt=$ckpt &> $output
