#!/usr/bin/env bash
echo 'exp_name' $1
KERAS_BACKEND=theano python app/runexp_cv.py \
 --data_dir=csv/ \
 --exp_name=${1} \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --nb_words=5300 \
 --max_sequence_len=40 \
 --num_epochs=30
