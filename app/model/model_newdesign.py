from __future__ import print_function

from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Flatten, Dropout, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import Adadelta, RMSprop
from keras.engine import Input, Merge, merge
import sys, os

def model_selector(args, embedding_matrix):
    '''Method to select the model to be used for classification'''
    if (args.exp_name.lower() == 'cnn'):
        return _segment_cnn_model(args, embedding_matrix)
    elif (args.exp_name.lower() == 'uttlabel_cnn'):
        return _uttlabel_cnn_model(args, embedding_matrix)
    elif (args.exp_name.lower() == 'lstm'):
        return _segment_bdlstm_model(args, embedding_matrix)
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
    (lstm_hs,
    dropout_list, optimizer, use_embeddings, embeddings_trainable) \
        = _param_selector_lstm(args)

    print('Defining segment BDLSTM model using neural-reader style.')

    # call neural-reader's implementations
    ########## PARAM #############
    vocab_size = args.nb_words
    word_dim = args.embedding_dim
    story_maxlen = args.max_sequence_len
    embed_weights = embedding_matrix
    lstm_hs = args.lstm_hs

    ########## MODEL ############
    segs = range(0, 10)
    ins = []
    embds = []
    for seg in segs:
        name = 'seg_' + str(seg)
        input = Input(shape=(story_maxlen,), dtype='int32', name=name)
        embd = Embedding(input_dim=vocab_size+1,
                      output_dim=word_dim,
                      input_length=story_maxlen,
                      mask_zero=True,
                      weights=[embed_weights],
                      trainable=embeddings_trainable)(input)
        ins.append(input)
        embds.append(embd)

    # A shared BDLSTM across all segments.
    shared_bdlstm = Bidirectional(LSTM(lstm_hs, dropout_W=0.2, dropout_U=0.2))
    shared_dense = Dense(1, activation='relu')

    lstm_outs = []
    for seg in segs:
        x = shared_bdlstm(embds[seg])
        out = shared_dense(x)
        lstm_outs.append(out)

    merged = merge(lstm_outs, mode='concat')
    result = Dense(1, init='normal')(merged) # regression won't use any non-linear activation.

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
    filtersize = params['filtersize']
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


def _uttlabel_cnn_model(args, embedding_matrix):
    """
    Use utterance level label to be segment level label.
    """
    (filtersize_list, number_of_filters_per_filtersize,
     dropout_list, optimizer, use_embeddings, embeddings_trainable) \
        = _param_selector(args)
    print('Defining uttlabel CNN model')

    # call neural-reader's implementations
    ########## PARAM #############
    vocab_size = args.nb_words
    word_dim = args.embedding_dim
    story_maxlen = args.max_sequence_len
    embed_weights = embedding_matrix

    ########## MODEL ############
    input = Input(shape=(story_maxlen,), dtype='int32', name='input')
    embd = Embedding(input_dim=vocab_size+1,
                  output_dim=word_dim,
                  input_length=story_maxlen,
                  mask_zero=False,
                  weights=[embed_weights],
                  trainable=embeddings_trainable)(input)

    nb_filter = number_of_filters_per_filtersize[0]
    filtersize = filtersize_list[0]
    pool_length = args.max_sequence_len - filtersize + 1

    shared_cnn = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')
    shared_dense = Dense(1, init='normal')

    x = shared_cnn(embd)
    x = MaxPooling1D(pool_length=pool_length)(x)
    x = Flatten()(x)
    result = shared_dense(x)

    model = Model(input=input, output=result)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    print(model.summary())
    return model


def _kim_cnn_model(args, embedding_matrix):
    '''
    fully functional API style so that we can see all model details.
    :param args:
    :param embedding_matrix:
    :return:
    '''
    if args.run_mode == 'paramtweak':
         (filtersize_list, number_of_filters_per_filtersize,
         dropout_list, optimizer, use_embeddings, embeddings_trainable) \
            = _param_for_tweak(args)
    else:
        (filtersize_list, number_of_filters_per_filtersize,
         dropout_list, optimizer, use_embeddings, embeddings_trainable) \
            = _param_selector(args)

    input = Input(shape=(args.max_sequence_len,), dtype='int32', name="input")
    if (use_embeddings):
        embedding_layer = Embedding(args.nb_words + 1,
                                    args.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=args.max_sequence_len,
                                    trainable=embeddings_trainable)(input)
    else:
        embedding_layer = Embedding(args.nb_words + 1,
                                    args.embedding_dim,
                                    weights=None,
                                    input_length=args.max_sequence_len,
                                    trainable=embeddings_trainable)(input)
    embedding_layer = Dropout(dropout_list[0])(embedding_layer)
    print('Defining Yoon Kim CNN model.')

    conv_list = []
    for index, filtersize in enumerate(filtersize_list):
        nb_filter = number_of_filters_per_filtersize[index]
        pool_length = args.max_sequence_len - filtersize + 1
        conv = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')(embedding_layer)
        pool = MaxPooling1D(pool_length=pool_length)(conv)
        flatten = Flatten()(pool)
        conv_list.append(flatten)

    if (len(filtersize_list) > 1):
        conv_out = Merge(mode='concat', concat_axis=1)(conv_list)
    else:
        conv_out = conv_list[0]

    dp_out = Dropout(dropout_list[1])(conv_out)
    result = Dense(args.len_labels_index, activation='softmax')(dp_out)

    model = Model(input=input, output=result)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer)
    #             metrics=['acc'])
    print(model.summary())
    return model



