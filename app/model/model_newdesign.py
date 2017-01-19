from __future__ import print_function

from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Flatten, Dropout, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import Adadelta, RMSprop
from keras.engine import Input, Merge, merge
import sys, os

def model_selector(params, args, embedding_matrix):
    '''Method to select the model to be used for classification'''
    if (args.exp_name.lower() == 'cnn'):
        return _segment_cnn_model(params, args, embedding_matrix)
    elif (args.exp_name.lower() == 'uttlabel_cnn'):
        return _uttlabel_cnn_model(params, args, embedding_matrix)
    elif (args.exp_name.lower() == 'lstm'):
        return _segment_bdlstm_model(params, args, embedding_matrix)
    else:
        print('wrong exp_name')
        sys.exit()


def _segment_bdlstm_model(params, args, embedding_matrix):
    """
    :param params: params will allow the model's HPs be tuned by using hyperopt
    :param args:
    :param embedding_matrix:
    :return:
    """
    lstm_hs = params['lstm_hs']
    dropout_list = [params['dropout1'], params['dropout2']]
    optimizer = params['optimizer']
    embeddings_trainable = params['embeddings_trainable']

    print('Defining segment BDLSTM model using neural-reader style.')

    ########## MODEL ############
    segs = range(0, 10)
    ins = []
    for seg in segs:
        name = 'seg_' + str(seg)
        input = Input(shape=(args.max_sequence_len,), dtype='int32', name=name)
        ins.append(input)

    in_x = Input(shape=(args.max_sequence_len,), dtype='int32')
    x = Embedding(input_dim=args.nb_words+1,
                  output_dim=args.embedding_dim,
                  mask_zero=True,
                  weights=[embedding_matrix])(in_x)
    x = Dropout(dropout_list[0])(x)
    x = Bidirectional(LSTM(output_dim=lstm_hs, dropout_U=0.2, dropout_W=0.2))(x)
    out_x = Dense(1, activation='relu')(x)
    shared_bdlstm = Model(input=in_x, output=out_x)

    lstm_outs = []
    for seg in ins:
        x = shared_bdlstm(seg)
        lstm_outs.append(x)

    x = merge(lstm_outs, mode='concat')
    result = Dense(1, init='normal')(x) # regression won't use any non-linear activation.

    model = Model(input=ins, output=result)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    print(model.summary())
    return model


def _segment_cnn_model(params, args, embedding_matrix):
    """
    :param args:
    :param embedding_matrix:
    :return:
    """
    filtersize = params['filter_size']
    #filtersize_list = [filtersize-1, filtersize, filtersize+1]
    nb_filter = params['nb_filter']
    dropout_list = [params['dropout1'], params['dropout2']]
    optimizer = params['optimizer']
    embeddings_trainable = params['embeddings_trainable']
    print('Defining segment CNN model using neural-reader style.')

    # call neural-reader's implementations
    ########## PARAM #############
    vocab_size = args.nb_words
    word_dim = args.embedding_dim
    story_maxlen = args.max_sequence_len
    embed_weights = embedding_matrix

    ########## MODEL ############
    segs = range(0, 10)
    ins = []
    for seg in segs:
        name = 'seg_' + str(seg)
        input = Input(shape=(story_maxlen,), dtype='int32', name=name)
        ins.append(input)

    # A shared model (sm) across all segments.
    pool_length = args.max_sequence_len - filtersize + 1

    in_x = Input(shape=(story_maxlen,), dtype='int32')
    x = Embedding(input_dim=vocab_size+1,
                  output_dim=word_dim,
                  input_length=story_maxlen,
                  mask_zero=False,
                  weights=[embed_weights],
                  trainable=embeddings_trainable)(in_x)
    x = Dropout(dropout_list[0])(x)
    x = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')(x)
    x = MaxPooling1D(pool_length=pool_length)(x)
    x = Flatten()(x)
    x = Dense(1, activation='relu')(x)
    out_x = Dropout(dropout_list[1])(x)

    shared_cnn = Model(in_x, out_x)

    sm_outs = []
    for seg in ins:
        out = shared_cnn(seg)
        sm_outs.append(out)

    x = merge(sm_outs, mode='concat')
    result = Dense(1, init='normal')(x) # regression won't use any non-linear activation.

    model = Model(input=ins, output=result)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    print(model.summary())
    return model


def _uttlabel_cnn_model(params, args, embedding_matrix):
    """
    Use utterance level label to be segment level label.
    """
    filtersize = params['filter_size']
    nb_filter = params['nb_filter']
    dropout_list = [params['dropout1'], params['dropout2']]
    optimizer = params['optimizer']
    embeddings_trainable = params['embeddings_trainable']
    print('Defining uttlabel CNN model')

    # call neural-reader's implementations
    ########## PARAM #############
    vocab_size = args.nb_words
    word_dim = args.embedding_dim
    story_maxlen = args.max_sequence_len
    embed_weights = embedding_matrix

    pool_length = story_maxlen - filtersize + 1
    ########## MODEL ############
    input = Input(shape=(story_maxlen,), dtype='int32', name='input')
    x = Embedding(input_dim=vocab_size+1,
                  output_dim=word_dim,
                  input_length=story_maxlen,
                  mask_zero=False,
                  weights=[embed_weights],
                  trainable=embeddings_trainable)(input)
    x = Dropout(dropout_list[0])(x)
    x = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')(x)
    x = MaxPooling1D(pool_length=pool_length)(x)
    x = Flatten()(x)
    x = Dropout(dropout_list[1])(x)
    result = Dense(1, init='normal')(x)

    model = Model(input=input, output=result)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    print(model.summary())
    return model

