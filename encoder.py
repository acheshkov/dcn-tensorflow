import numpy as np
import tensorflow as tf
import utils

'''
document_ph (batch, max_sequence_length, word_vector_size)
question_ph (batch, max_sequence_length, word_vector_size)
document_size [batch_size]
max_sequence_length scalar
'''
def encoderBatch(document_ph, question_ph, document_size, question_size, lstm, lstm_cenc_fw, lstm_cenc_bw, sentinel_d, sentinel_q, batch_size, FLAGS):
    
    lstm_size = FLAGS.lstm_size
    max_sequence_length = FLAGS.max_sequence_length
    max_question_length = FLAGS.max_question_length
    

    # we use the same LSTM for both encodings to share weights
    with tf.name_scope('ENCODER'):
        # output has shape [batch, max_time, L]
        with tf.name_scope('Q_ENC'):
            outputs_q, state_q = tf.nn.dynamic_rnn(
                lstm, inputs = question_ph, 
                sequence_length = question_size, dtype=tf.float32)
        with tf.name_scope('D_ENC'):
            outputs_d, state_d = tf.nn.dynamic_rnn(
                lstm, 
                inputs = document_ph,
                sequence_length = document_size, dtype=tf.float32)

    # sentinel_vector = (1, L)
    def sentinelAddition(pos, sentinel_vector, size):
        return tf.pad(sentinel_vector, [[pos - 1, size - pos], [0, 0]])
        
    sentinels_q = tf.map_fn(
        lambda pos: sentinelAddition(pos + 1, sentinel_q, max_question_length + 1), question_size, dtype=tf.float32)
    # (batch, Q + 1, L)
    sentinels_q = tf.stack(sentinels_q)
    sentinels_d = tf.map_fn(
        lambda pos: sentinelAddition(pos + 1, sentinel_d, max_sequence_length + 1), document_size, dtype=tf.float32)
    # (batch, D + 1, L)
    sentinels_d = tf.stack(sentinels_d)
    
    outputs_q = tf.pad(outputs_q, [[0,0],[0,1],[0,0]])
    que_enc_sentinel = tf.add(outputs_q, sentinels_q)
    que_enc_sentinel = tf.transpose(que_enc_sentinel, perm=[0,2,1])
    # (batch, L, Q + 1)
    que_enc_sentinel = utils.non_linear_projection_batch(
        que_enc_sentinel, [lstm_size, lstm_size], [lstm_size, max_question_length + 1], batch_size
    )
    
    
    outputs_d = tf.pad(outputs_d, [[0,0],[0,1],[0,0]])
    doc_enc_sentinel = tf.add(outputs_d, sentinels_d)
    doc_enc_sentinel = tf.transpose(doc_enc_sentinel, perm=[0,2,1])
    
    
    # ===================  COATTENTION ENCODER ===================
    # (B, D+1, L)
    doc_enc_sentinel_transposed = tf.transpose(doc_enc_sentinel, perm=[0,2,1])
    
    with tf.name_scope('COATTENTION_ENCODER'):
        # L \in batch x (doc_size + 1) x (que_size + 1)
        L = tf.matmul(doc_enc_sentinel_transposed, que_enc_sentinel)
        A_Q = tf.nn.softmax(L, 2)
        A_D = tf.nn.softmax(tf.transpose(L, perm=[0,2,1]), 2)
        # (B,L,D+1) * (b*d+1*q+1) = (b, L, q+1)
        C_Q = tf.matmul(doc_enc_sentinel, A_Q)
        # C_D \in Batch,  2*lstm_size , (doc_size + 1)
        C_D = tf.matmul(tf.concat([que_enc_sentinel, C_Q], axis = 1), A_D)

        bi_lstm_input = tf.concat([doc_enc_sentinel, C_D], axis = 1)
        bi_lstm_input = tf.transpose(bi_lstm_input, perm=[0,2,1])

        #tf.summary.histogram('bi_lstm_input', bi_lstm_input)
        outputs_bi, output_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = lstm_cenc_fw, 
            cell_bw = lstm_cenc_bw,
            inputs = bi_lstm_input,
            sequence_length = document_size + 1,
            dtype=tf.float32
        )

        # outputs_bi is (batch, D + 1, 2L)
        outputs_bi = tf.concat(outputs_bi, axis=2)
        U = tf.slice(outputs_bi, [0,0,0], [batch_size, max_sequence_length, 2*lstm_size])


    return U

