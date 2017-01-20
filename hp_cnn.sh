#!/usr/bin/env bash
echo 'exp_name: ' $1
echo 'gpu: ' $2

# setup GPU to run
th_flag="THEANO_FLAGS=\"device=gpu"
th_flag+=$2
th_flag+="\""
echo $th_flag
eval $th_flag

KERAS_BACKEND=theano python app/tweak_params.py \
 --data_dir=csv/ \
 --exp_name=${1} \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --nb_words=5300 \
 --max_sequence_len=40 \
 --num_epochs=30 \
 --max_evals=100
