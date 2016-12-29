#!/usr/bin/env bash
KERAS_BACKEND=theano python app/train_pair_uttlabel.py \
 --exp_name=cnn\
 --data_dir=csv/ \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --nb_words=5100 \
 --batch_size=16 \
 --max_sequence_len=40 \
 --num_epochs=20 \
 --model_name=cnn-static
