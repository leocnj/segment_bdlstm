from __future__ import print_function

import numpy as np
from keras.callbacks import EarlyStopping

from model.model_newdesign import model_selector
from reader.filereader import read_glove_vectors
from reader.csvreader import read_input_csv, _reshape_input, _reshape_pred

from utils import argumentparser
from scipy.stats import pearsonr

np.random.seed(42)


def one_cv_exp(args, params):
    '''
    Data providing function:

    :return:
    '''
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 35, 36, 37,
             38]

    trains = [args.data_dir + 'cv_ta_' + str(fold) for fold in folds]
    tests = [args.data_dir + 'cv_ts_' + str(fold) for fold in folds]
    pairs = zip(trains, tests)

    all_pred, all_act = train_cv(args, params, pairs)
    corr_r = pearsonr(all_pred, all_act)
    print('CV Pearson corr: {}.'.format(corr_r))


def _gen_embd_matrix(args, embeddings_index, word_index):
    embedding_matrix = np.zeros((args.nb_words + 1, args.embedding_dim))
    for word, i in word_index.items():
        if i > args.nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def run_a_model_uttlabel(args, params, embedding_matrix, x_train, y_train, x_test, y_test):
    '''

    :param params:
    :return:
    '''
    nb_epoch = args.num_epochs
    batch_size = params['batch_size']

    model = model_selector(params, args, embedding_matrix)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    callbacks_list = [earlystop]

    # for each segment, use uttlabel to train
    x_train_onecol, y_train_onecol = _reshape_input(x_train, y_train)
    x_test_onecol, y_test_onecol = _reshape_input(x_test, y_test)
    model.fit(x_train_onecol, y_train_onecol,
              validation_split=0.1,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              callbacks=callbacks_list,
              verbose=0)

    pred = earlystop.model.predict(x_test_onecol, batch_size=batch_size)
    pred = pred.flatten()  # to 1D array
    # mean every 10 rows
    # http://bit.ly/2hRcM1r
    pred = _reshape_pred(pred)
    return pred


def run_a_model(args, params, embedding_matrix, x_train, y_train, x_test, y_test):
    '''

    :param params:
    :return:
    '''
    nb_epoch = args.num_epochs
    batch_size = params['batch_size']

    model = model_selector(params, args, embedding_matrix)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    callbacks_list = [earlystop]

    model.fit(x_train, y_train,
              validation_split=0.1,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              callbacks=callbacks_list,
              verbose=0)

    pred = earlystop.model.predict(x_test, batch_size=batch_size)
    pred = pred.flatten()  # to 1D array
    return pred


def train_cv(args, params, pairs):
    embeddings_index = read_glove_vectors(args.embedding_file_path)

    all_pred = np.zeros(0)
    all_act = np.zeros(0)
    for (train, test) in pairs:
        print(train + '=>' + test + '...')
        x_train, y_train, x_test, y_test, word_index = read_input_csv(train,
                                                                      test,
                                                                      args.nb_words,
                                                                      args.max_sequence_len)
        embedding_matrix = _gen_embd_matrix(args, embeddings_index, word_index)

        if args.exp_name.lower() == 'cnn':
            y_pred = run_a_model(args, params, embedding_matrix, x_train, y_train, x_test, y_test)
        elif args.exp_name.lower() == 'uttlabel_cnn':
            y_pred = run_a_model_uttlabel(args, params, embedding_matrix, x_train, y_train, x_test, y_test)
        else:
            pass

        corr_r = np.nan_to_num(pearsonr(y_pred, y_test)[0])  # force no NAN
        print('Pearson corr: {}.'.format(corr_r))
        all_pred = np.concatenate([all_pred, y_pred])
        all_act = np.concatenate([all_act, y_test])
    return (all_pred, all_act)


if __name__ == '__main__':
    args = argumentparser.ArgumentParser()

    if args.exp_name.lower() == 'cnn':
        space = {'optimizer': 'adadelta',
                 'batch_size': 64,
                 'filter_size': 4,
                 'nb_filter': 100,
                 'dropout1': 0.6366,
                 'dropout2': 0.5996,
                 'embeddings_trainable': False}
    elif args.exp_name.lower() == 'uttlabel_cnn':
        space = {'optimizer': 'adadelta',
                 'batch_size': 32,
                 'filter_size': 5,
                 'nb_filter': 100,
                 'dropout1': 0.3588,
                 'dropout2': 0.2597,
                 'embeddings_trainable': False}
    elif args.exp_name.lower() == 'lstm':
        space = {'optimizer': 'adadelta',
                 'batch_size': 64,
                 'filter_size': 4,
                 'nb_filter': 100,
                 'dropout1': 0.6366,
                 'dropout2': 0.5996,
                 'embeddings_trainable': False}
    one_cv_exp(args, space)
