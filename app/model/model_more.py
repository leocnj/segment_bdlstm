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

def _segment_bdlstm_model(args, embedding_matrix):
    """
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


def _param_selector_lstm(args):
    '''
    Setup param for LSTM
    :param args:
    :return:
    '''
    lstm_hs = args.lstm_hs
    dropout_list = [0.5, 0.5]
    optimizer = Adadelta(clipvalue=3)
    use_embeddings = True
    embeddings_trainable = False

    if (args.model_name.lower() == 'lstm-rand'):
        use_embeddings = False
        embeddings_trainable = True
    elif (args.model_name.lower() == 'lstm-static'):
        pass
    elif (args.model_name.lower() == 'lstm-non-static'):
        embeddings_trainable = True
    else: # best setup
        dropout_list = [0.25, 0.5]
        optimizer = RMSprop(lr=args.learning_rate, decay=args.decay_rate,
                            clipvalue=args.grad_clip)
        use_embeddings = True
        embeddings_trainable = False
    return (lstm_hs, dropout_list, optimizer, use_embeddings, embeddings_trainable)



def _segment_cnn_model(args, embedding_matrix):
    """
    :param args:
    :param embedding_matrix:
    :return:
    """
    (filtersize_list, number_of_filters_per_filtersize,
     dropout_list, optimizer, use_embeddings, embeddings_trainable) \
        = _param_selector(args)
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
    embds = []
    for seg in segs:
        name = 'seg_' + str(seg)
        input = Input(shape=(story_maxlen,), dtype='int32', name=name)
        embd = Embedding(input_dim=vocab_size+1,
                      output_dim=word_dim,
                      input_length=story_maxlen,
                      mask_zero=False,
                      weights=[embed_weights],
                      trainable=embeddings_trainable)(input)
        embd = Dropout(dropout_list[0])(embd)
        ins.append(input)
        embds.append(embd)

    # A shared CNN across all segments.
    nb_filter = number_of_filters_per_filtersize[0]
    filtersize = filtersize_list[0]
    pool_length = args.max_sequence_len - filtersize + 1

    shared_cnn = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')
    shared_dense = Dense(1, activation='relu')

    lstm_outs = []
    for seg in segs:
        x = shared_cnn(embds[seg])
        x = MaxPooling1D(pool_length=pool_length)(x)
        x = Flatten()(x)
        out = shared_dense(x)
        lstm_outs.append(out)

    x = merge(lstm_outs, mode='concat')
    x = Dropout(dropout_list[1])(x)
    result = Dense(8, activation='softmax')(x) # regression won't use any non-linear activation.

    model = Model(input=ins, output=result)
#    model.compile(optimizer=optimizer,
#                  loss='mean_squared_error')
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
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
    embd = Dropout(dropout_list[0])(embd)

    nb_filter = number_of_filters_per_filtersize[0]
    filtersize = filtersize_list[0]
    pool_length = args.max_sequence_len - filtersize + 1

    shared_cnn = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')
    shared_dense = Dense(8, activation='softmax')

    x = shared_cnn(embd)
    x = MaxPooling1D(pool_length=pool_length)(x)
    x = Flatten()(x)
    x = Dropout(dropout_list[1])(x)
    result = shared_dense(x)

    model = Model(input=input, output=result)
#   model.compile(optimizer=optimizer,
#                 loss='mean_squared_error')
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
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

def _param_selector(args):
    '''Method to select parameters for models defined in Convolutional Neural Networks for
        Sentence Classification paper by Yoon Kim'''
    filtersize_list = [3, 4, 5]
    number_of_filters_per_filtersize = [100, 100, 100]
    dropout_list = [0.5, 0.5]
    optimizer = Adadelta(clipvalue=3)
    use_embeddings = True
    embeddings_trainable = False

    if (args.model_name.lower() == 'cnn-rand'):
        use_embeddings = False
        embeddings_trainable = True
    elif (args.model_name.lower() == 'cnn-static'):
        pass
    elif (args.model_name.lower() == 'cnn-non-static'):
        embeddings_trainable = True
    else:  # other case follows dl setup.
        filtersize_list = [2, 5, 8]
        number_of_filters_per_filtersize = [100, 100, 100]
        dropout_list = [0.25, 0.5]
        optimizer = RMSprop(lr=args.learning_rate, decay=args.decay_rate,
                            clipvalue=args.grad_clip)
        use_embeddings = True
        embeddings_trainable = False
    return (filtersize_list, number_of_filters_per_filtersize,
            dropout_list, optimizer, use_embeddings, embeddings_trainable)


######################################################
#  Code for tweak_params
#
def _param_for_tweak(args):
    """
    Setup params for tweak
    :param args:
    :return:
    """
    filtersize_list = args.filter_size
    number_of_filters_per_filtersize = args.filter_num
    dropout_list = [0.5, 0.5]
    optimizer = Adadelta(clipvalue=3)
    use_embeddings = True
    embeddings_trainable = False

    if (args.model_name.lower() == 'cnn-rand'):
        use_embeddings = False
        embeddings_trainable = True
    elif (args.model_name.lower() == 'cnn-static'):
        pass
    elif (args.model_name.lower() == 'cnn-non-static'):
        embeddings_trainable = True
    else:
        pass
    return (filtersize_list, number_of_filters_per_filtersize,
            dropout_list, optimizer, use_embeddings, embeddings_trainable)
