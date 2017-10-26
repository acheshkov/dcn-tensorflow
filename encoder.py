import numpy as np
import tensorflow as tf
import utils



def encoder(document_ph, question_ph, document_size, question_size, lstm, lstm_cenc_fw, lstm_cenc_bw, sentinel_d, sentinel_q, FLAGS):
    
    lstm_size = FLAGS.lstm_size
    max_sequence_length = FLAGS.max_sequence_length
    max_question_length = FLAGS.max_question_length
    
    '''
    transform tensor of shape [1, question_size, word_vector_size] to list of tensors of shape [1, word_vector_size]
    of length question_size. first dimenstion is batch size = 1
    '''
    # we use the same LSTM for both encodings to share weights
    with tf.name_scope('ENCODER'):
        with tf.name_scope('Q_ENC'):
            outputs_q, state_q = tf.nn.dynamic_rnn(
                lstm, inputs = question_ph, 
                sequence_length = [question_size], dtype=tf.float32)
        with tf.name_scope('D_ENC'):
            outputs_d, state_d = tf.nn.dynamic_rnn(
                lstm, 
                inputs = document_ph,
                sequence_length = [document_size], dtype=tf.float32)


    #document_size = doc_len_ph
    #question_size = que_len_ph
    doc_padding = tf.subtract([0, max_sequence_length], [0, document_size])
    que_padding = tf.subtract([0, max_question_length], [0, question_size])


    # "squeeze" transforms list of tensors of shape [1, lstm_size] of length L to tensor of shape [L, lstm_size]
    que_enc = tf.transpose(tf.squeeze(outputs_q))
    que_enc = tf.slice(que_enc, [0,0], [lstm_size, question_size])
    que_enc_sentinel = tf.concat([que_enc, sentinel_q], axis = 1)
    que_enc_sentinel = tf.pad(que_enc_sentinel, [[0,0], que_padding])
    que_enc_sentinel.set_shape([lstm_size, max_question_length + 1])
    que_enc_sentinel = utils.non_linear_projection(que_enc_sentinel)
    que_enc_sentinel = tf.slice(que_enc_sentinel, [0,0], [lstm_size, question_size + 1])

    doc_enc = tf.transpose(tf.squeeze(outputs_d))
    doc_enc = tf.slice(doc_enc, [0,0], [lstm_size, document_size])

    #tf.summary.histogram('QUE_enc', que_enc)
    #tf.summary.histogram('DOC_enc', doc_enc)
    #tf.summary.histogram('DOC_enc_max', tf.reduce_max(doc_enc))
    #tf.summary.histogram('QUE_enc_max', tf.reduce_max(que_enc))
    #tf.summary.histogram('Document_size', document_size)
    #tf.summary.histogram('Question_size', question_size)


    # append sentinel vector for both encodings 
    doc_enc_sentinel = tf.concat([doc_enc, sentinel_d], axis = 1)
    #que_enc_sentinel = utils.non_linear_projection(tf.concat([que_enc, sentinel_q], axis = 1))
    print(que_enc_sentinel)
    #que_enc_sentinel = tf.slice(que_enc_sentinel, [0,0], [lstm_size, question_size + 1])

    # ===================  COATTENTION ENCODER ===================
    with tf.name_scope('COATTENTION_ENCODER'):
        # L \in R(doc_size + 1) x (que_size + 1)
        L = tf.matmul(doc_enc_sentinel, que_enc_sentinel, transpose_a = True)
        A_Q = tf.nn.softmax(L, 1)
        A_D = tf.nn.softmax(tf.transpose(L), 1)
        C_Q = tf.matmul(doc_enc_sentinel, A_Q)
        # C_D \in R_2*lstm_size x (doc_size + 1)
        C_D = tf.matmul(tf.concat([que_enc_sentinel, C_Q], axis = 0), A_D)

        # TODO Q: would we use single cell of two different
        bi_lstm_input = tf.concat([doc_enc_sentinel, C_D], axis = 0)
        bi_lstm_input = tf.transpose(bi_lstm_input)
        bi_lstm_input = tf.reshape(bi_lstm_input, [1, document_size + 1, 3*lstm_size])

        #tf.summary.histogram('bi_lstm_input', bi_lstm_input)

        outputs_bi, output_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = lstm_cenc_fw, 
            cell_bw = lstm_cenc_bw,
          #  cell_bw = lstm_cenc_bw,
            inputs = bi_lstm_input,
           # sequence_length = [document_size[0] + 1],
            dtype=tf.float32
        )

        # we take first because of we feed to bi-RNN only one sentence
        outputs_bi = tf.concat(outputs_bi, axis=2)[0]
        print(outputs_bi)
        U = tf.slice(outputs_bi, [0,0], [document_size, 2*lstm_size])
        U = tf.transpose(U)
    #print(U)
    #tf.summary.histogram('U', U)
    #tf.summary.histogram('U_max', tf.reduce_max(U))
    return U