#!/usr/bin/env bash
echo 'GPU:' $1
KERAS_BACKEND=theano THEANO_FLAGS='device=${1}' python app/tweak_params.py \
 --data_dir=csv/ \
 --exp_name=cnn \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --nb_words=5300 \
 --max_sequence_len=40 \
 --num_epochs=30 \
 --max_evals=100
