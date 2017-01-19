from __future__ import print_function

import numpy as np
from keras.callbacks import EarlyStopping

from model.model_newdesign import model_selector
from reader.filereader import read_glove_vectors
from reader.csvreader import read_input_csv

from utils import argumentparser
from scipy.stats import pearsonr

from hyperopt import fmin, hp, Trials, STATUS_OK, tpe

np.random.seed(42)


def prep_data(args):
    '''
    Data providing function:

    :return:
    '''
    train = args.data_dir + '/param_train.csv'
    test = args.data_dir + '/param_dev.csv'
    x_train, y_train, x_test, y_test, word_index = read_input_csv(train,
                                                                  test,
                                                                  args.nb_words,
                                                                  args.max_sequence_len)
    embedding_matrix = _gen_embd_matrix(args, word_index)
    return x_train, y_train, x_test, y_test, embedding_matrix


def _gen_embd_matrix(args, word_index):
    embeddings_index = read_glove_vectors(args.embedding_file_path)
    embedding_matrix = np.zeros((args.nb_words + 1, args.embedding_dim))
    for word, i in word_index.items():
        if i > args.nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def trials2csv(trials, csvfile):
    lst = [(t['misc']['vals'], -t['result']['loss']) for t in trials.trials]
    new = []
    for dict, val in lst:
        dict['val'] = val
        new.append(dict)

    keys = new[0].keys()
    with open(csvfile, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(new)


def model_to_tweak(params):
    '''

    :param params:
    :return:
    '''
    #global x_train, y_train, x_test, y_test
    global args
    global embedding_matrix

    nb_epoch = args.num_epochs
    batch_size = params['batch_size']

    model = model_selector(params, args, embedding_matrix)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    callbacks_list = [earlystop]

    model.fit(x_train, y_train,
              validation_split=0.1,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              callbacks=callbacks_list)

    pred = earlystop.model.predict(x_test, batch_size=batch_size)
    pred = pred.flatten()  # to 1D array

    # corr_r = pearsonr(y_test, pred)
    corr_r = np.nan_to_num(pearsonr(y_test, pred)[0])  # force no NAN
    print('Test Pearson corr: {}.'.format(corr_r))
    return {'loss': -1.0*float(corr_r), 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    args = argumentparser.ArgumentParser()
    print('Processing input data')
    x_train, y_train, x_test, y_test, embedding_matrix = prep_data(args)

    if args.exp_name.lower() == 'cnn':
        space = {'optimizer': hp.choice('optimizer', ['adadelta', 'rmsprop']),
                 'batch_size': hp.choice('batch_size', [32, 64]),
                 'filter_size': hp.choice('filter_size', [3, 4, 5]),
                 'nb_filter': hp.choice('nb_filter', [75, 100]),
                 'dropout1': hp.uniform('dropout1', 0.25, 0.75),
                 'dropout2': hp.uniform('dropout2', 0.25, 0.75),
                 'embeddings_trainable': False}
        trials = Trials()
        best = fmin(model_to_tweak, space, algo=tpe.suggest, max_evals=args.max_evals, trials=trials)
        print(best)
        trials2csv(trials, 'segment_cnn_hp.csv')
    elif args.exp_name.lower() == 'lstm':
        space = {'optimizer': hp.choice('optimizer', ['adadelta', 'rmsprop']),
                 'batch_size': hp.choice('batch_size', [32, 64]),
                 'dropout1': hp.uniform('dropout1', 0.25, 0.75),
                 'dropout2': hp.uniform('dropout2', 0.25, 0.75),
                 'lstm_hs': hp.choice('lstm_hs', [32, 48, 64]),
                 'embeddings_trainable': False}
    trials = Trials()
    best = fmin(model_to_tweak, space, algo=tpe.suggest, max_evals=args.max_evals, trials=trials)
    print(best)
    trials2csv(trials, 'segment_bdlstm_hp.csv')
