from __future__ import print_function

import numpy as np
from keras.callbacks import EarlyStopping

from model.model_more import model_selector
from reader.filereader import read_glove_vectors
from reader.csvreader import read_input_csv, _reshape_input

from utils import argumentparser
from scipy.stats import pearsonr
import pandas as pd

np.random.seed(42)


def main():
    args = argumentparser.ArgumentParser()
    ta_csv = args.data_dir + "param_train.csv"
    ts_csv = args.data_dir + "param_dev.csv"
    train_pair(args, ta_csv, ts_csv)


def train_pair(args, train_csv, test_csv):
    print('Reading word vectors.')
    embeddings_index = read_glove_vectors(args.embedding_file_path)
    print('Found {} word vectors.'.format(len(embeddings_index)))

    print('Processing input data')
    x_train, y_train, x_test, y_test, word_index, = read_input_csv(train_csv,
                                                                   test_csv,
                                                                   args.nb_words,
                                                                   args.max_sequence_len)

    print('Preparing embedding matrix.')
    # initiate embedding matrix with zero vectors.
    nb_words = min(args.nb_words, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, args.embedding_dim))
    for word, i in word_index.items():
        if i > nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    args.nb_words = nb_words
    args.len_labels_index = 1  # fixed for regression.

    model = model_selector(args, embedding_matrix)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    callbacks_list = [earlystop]

    # During training, convert wide x_train to long shape
    x_train_onecol, y_train_onecol = _reshape_input(x_train, y_train)

    model.fit(x_train_onecol, y_train_onecol,
              validation_split=0.1,
              nb_epoch=args.num_epochs,
              batch_size=args.batch_size,
              callbacks=callbacks_list)

    # During testing, iterate each x_test column
    x_test_onecol, y_test_onecol = _reshape_input(x_test, y_test)
    pred = earlystop.model.predict(x_test_onecol, batch_size=args.batch_size)
    pred = pred.flatten() # to 1-D arrary

    print('y_test_col shape {}'.format(y_test_onecol.shape))
    print('prediction shape {}'.format(pred.shape))

    # out to result csv
    df = pd.DataFrame({'pred': pred, 'actual': y_test_onecol})
    df.to_csv('result_uttlabel.csv')

    corr_r = pearsonr(y_test_onecol, pred)
    print('prediciton.{}'.format(pred))
    print('Test Pearson corr: {}.'.format(corr_r))


if __name__ == '__main__':
    main()
