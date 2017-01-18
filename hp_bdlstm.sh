#!/usr/bin/env bash
KERAS_BACKEND=theano python app/tweak_params.py \
 --data_dir=csv/ \
 --exp_name=lstm \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --nb_words=5300 \
 --max_sequence_len=40 \
 --num_epochs=30
