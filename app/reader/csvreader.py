from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def read_input_csv(train_csv, test_csv, nb_words, maxlen):
    """
    Method to read csv file pairs

    :returns
    X_train (, maxlen)
    Y_train (, 1)
    X_test
    Y_Test
    word_index: will be used later for initing embedding matrix

    """
    train_df = pd.read_csv(train_csv)
    train_df = train_df.sample(frac=1).reset_index(drop=True) # shuffle train
    test_df = pd.read_csv(test_csv)
    print(train_df.head())


    # obtain Tokenizer
    texts = []
    for ci in range(1, 10):
        texts = texts + train_df.ix[:, ci].values.tolist()
        texts = texts + test_df.ix[:, ci].values.tolist()

    textraw = [line.encode('utf-8') for line in texts]  # keras needs str
    token = Tokenizer(nb_words=nb_words)
    token.fit_on_texts(textraw)
    word_index = token.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    train_X = []
    test_X = []
    for ci in range(1,11):
        ta_X = train_df.ix[:, ci].values.tolist()
        ts_X = test_df.ix[:, ci].values.tolist()

        n_ta = len(ta_X)
        n_ts = len(ts_X)
        print('col: {}'.format(ci))
        textseq = token.texts_to_sequences(ta_X + ts_X)
        lens = [len(line) for line in textseq]
        print('sentence lens: {}'.format([np.min(lens), np.max(lens), np.percentile(lens, 90)]))

        ta_X = pad_sequences(textseq[0:n_ta], maxlen,  padding='post', truncating='post')
        ts_X = pad_sequences(textseq[n_ta:], maxlen,  padding='post', truncating='post')
        train_X.append(ta_X)
        test_X.append(ts_X)

    train_y = train_df.BARS.values
    test_y = test_df.BARS.values
    # for regression task, directly using float number

    return train_X, train_y, test_X, test_y, word_index


def test_data():
    """
    test code
    :return:
    """
    dir = "../../csv/"
    ta_csv = dir + "param_train.csv"
    ts_csv = dir + "param_dev.csv"
    tps = read_input_csv(ta_csv, ts_csv, nb_words=20000, maxlen=30)
    print(tps[1])

if __name__ == '__main__':
    test_data()
