#!/usr/bin/env bash
echo $1
KERAS_BACKEND=theano python app/train_pair.py \
 --exp_name=lstm\
 --data_dir=csv/ \
 --embedding_file_path=embd/Google_w2v_300d.txt \
 --embedding_dim=300 \
 --nb_words=5100 \
 --batch_size=16 \
 --max_sequence_len=40 \
 --lstm_hs=32 \
 --num_epochs=100 \
 --model_name=$1