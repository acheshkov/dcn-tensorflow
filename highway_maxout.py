import numpy as np
import tensorflow as tf
#import maxout
import utils as ut

# Highway Maxout Network
'''
Simple version of maxout. Reduces one dimension to 1
'''
def maxout(input, axis):
    return tf.reduce_max(input, axis=axis)


'''
Args:
    U   is shape of (B, D, 2L)
    h_i is shape of (B, L, 1)
    u_s_i, u_e_i are shape of (B, 2L, 1)

Returns:
    tensor of shape (batch, document_size) of scores
'''
def HMN_Batch(U, h_i, u_s_i, u_e_i, scope = None, FLAGS = None, dropout_rate = 1, iter_number=-1):
    #U = tf.transpose(U_T)
    maxout_pooling_size = FLAGS.maxout_pooling_size
    batch_size = FLAGS.train_batch_size
    max_sequence_length = FLAGS.max_sequence_length
    lstm_size = FLAGS.lstm_size
    scope = scope or tf.get_variable_scope()
    
    b_1 = ut.get_scope_variable(scope, 'hmn_b_1', [1, 1, lstm_size, maxout_pooling_size])
    B_1 = tf.tile(b_1, [batch_size,  max_sequence_length, 1, 1])
    w_1 = ut.get_scope_variable(scope, 'hmn_w_1', [maxout_pooling_size, lstm_size, 3 * lstm_size])
    w_d = ut.get_scope_variable(scope, 'hmn_w_d', [5*lstm_size, lstm_size])
    b_2 = ut.get_scope_variable(scope, 'hmn_b_2', [1, 1, maxout_pooling_size, lstm_size])
    B_2 = tf.tile(b_2, [batch_size,  max_sequence_length, 1, 1])
    w_2 = ut.get_scope_variable(scope, 'hmn_w_2', [maxout_pooling_size, lstm_size, lstm_size])
    b_3 = ut.get_scope_variable(scope, 'hmn_b_3', [1, 1, maxout_pooling_size])
    B_3 = tf.tile(b_3, [batch_size,  max_sequence_length, 1])
    w_3 = ut.get_scope_variable(scope, 'hmn_w_3', [maxout_pooling_size, 2 * lstm_size])
    
    if iter_number == 4:
        tf.summary.histogram(scope + '/hmn_b_1', b_1)
        tf.summary.histogram(scope + '/hmn_w_1', w_1)
        tf.summary.histogram(scope + '/hmn_w_d', w_d)
        tf.summary.histogram(scope + '/hmn_w_3', w_3)
    
    # r is shape of (B, L)
    r = tf.tanh(tf.matmul(tf.concat([h_i, u_s_i, u_e_i], axis = 1), w_d));
    r = tf.reshape(r, [batch_size, 1, lstm_size])
    # R is shape (B, D, L)
    R = tf.tile(r, [1, max_sequence_length, 1])
    # m_1_1 is shape of (B, D, L, p)
    dropout_input_m_1 = tf.nn.dropout(tf.concat([U, R], axis = 2), dropout_rate)
    m_1_1 = tf.tensordot(dropout_input_m_1, tf.transpose(w_1), axes=[[2], [0]])
    # m_1_2 is shape of (B, D, L, p)
    m_1_2 = tf.add(m_1_1, B_1)
    # m_1 is shape of (B, D, L)
    m_1 = maxout(m_1_2, axis = 3 )
    
    # m_2_1 is shape of (B, D, p, L)
    dropout_input_m_2 = tf.nn.dropout(m_1, dropout_rate)
    m_2_1 = tf.tensordot(dropout_input_m_2, w_2, axes=[[2], [2]])
    m_2_2 = tf.add(m_2_1, B_2)
    # m_2 is shape of (B, D, L)
    m_2 = maxout(m_2_2, axis = 2 )
    
    # m_3_1 is shape of (B, D, p)
    dropout_input_m_3 = tf.nn.dropout(tf.concat([m_1, m_2], axis = 2), dropout_rate)
    m_3_1 = tf.tensordot(dropout_input_m_3, w_3, axes=[[2], [1]])
    m_3_2 = tf.add(m_3_1, B_3)
    m_3 = maxout(m_3_2, axis = 2)
    
    if iter_number == 3:
        tf.summary.histogram(scope + '/hmn_b_1', b_1)
        tf.summary.histogram(scope + '/hmn_w_1', w_1)
        tf.summary.histogram(scope + '/hmn_w_d', w_d)
        tf.summary.histogram(scope + '/hmn_w_3', w_3)
        tf.summary.histogram(scope + '/hmn_m_3', m_3)
    
    return m_3


'''
Args:
    U   is shape of (D, 2L)
    h_i is shape of (L, 1)
    u_s_i, u_e_i are shape of (2L, 1)

Returns:
    tensor of shape (1, document_size) of scores
'''
def HMN2(U_T, h_i, u_s_i, u_e_i, doc_size, scope = None, FLAGS = None, dropout_rate = 1, iter_number=-1):
    U = tf.transpose(U_T)
    maxout_pooling_size = FLAGS.maxout_pooling_size
    lstm_size = FLAGS.lstm_size
    scope = scope or tf.get_variable_scope()
    b_1 = ut.get_scope_variable(scope, 'hmn_b_1', [lstm_size, maxout_pooling_size])
    B_1 = tf.reshape(tf.tile(b_1, [doc_size, 1]), [doc_size, lstm_size, maxout_pooling_size])
    w_1 = ut.get_scope_variable(scope, 'hmn_w_1', [maxout_pooling_size, lstm_size, 3 * lstm_size])
    w_d = ut.get_scope_variable(scope, 'hmn_w_d', [lstm_size, 5*lstm_size])
    b_2 = ut.get_scope_variable(scope, 'hmn_b_2', [maxout_pooling_size, lstm_size])
    B_2 = tf.reshape(tf.tile(b_2, [doc_size, 1]), [doc_size, maxout_pooling_size, lstm_size])
    w_2 = ut.get_scope_variable(scope, 'hmn_w_2', [maxout_pooling_size, lstm_size, lstm_size])
    b_3 = ut.get_scope_variable(scope, 'hmn_b_3', [maxout_pooling_size])
    B_3 = tf.reshape(tf.tile(b_3, [doc_size]),    [doc_size, maxout_pooling_size])
    w_3 = ut.get_scope_variable(scope, 'hmn_w_3', [maxout_pooling_size, 2 * lstm_size])
    
    if iter_number == 4:
        tf.summary.histogram(scope + '/hmn_b_1', b_1)
        tf.summary.histogram(scope + '/hmn_w_1', w_1)
        tf.summary.histogram(scope + '/hmn_w_d', w_d)
        tf.summary.histogram(scope + '/hmn_w_3', w_3)
    
    # r is shape of (L, 1)
    r = tf.tanh(tf.matmul(w_d, tf.concat([h_i, u_s_i, u_e_i], axis = 0)));
    # R is shape (D, L)
    R = tf.reshape(tf.tile(r, [doc_size, 1]), [doc_size, lstm_size])
    # m_1_1 is shape of (D, L, p)
    dropout_input_m_1 = tf.nn.dropout(tf.concat([U, R], axis = 1), dropout_rate)
    m_1_1 = tf.tensordot(dropout_input_m_1, tf.transpose(w_1), axes=[[1], [0]])
    # m_1_2 is shape of (D, L, p)
    m_1_2 = tf.add(m_1_1, B_1)
    # m_1 is shape of (D, L)
    m_1 = tf.squeeze(maxout(m_1_2, axis = 2 ))
    
    # m_2_1 is shape of (D, p, L)
    dropout_input_m_2 = tf.nn.dropout(m_1, dropout_rate)
    m_2_1 = tf.tensordot(dropout_input_m_2, w_2, axes=[[1], [2]])
    m_2_2 = tf.add(m_2_1, B_2)
    # m_2 is shape of (D, L)
    m_2 = tf.squeeze(maxout(m_2_2, axis = 1 ))
    
    # m_3_1 is shape of (D, p)
    dropout_input_m_3 = tf.nn.dropout(tf.concat([m_1, m_2], axis = 1), dropout_rate)
    m_3_1 = tf.tensordot(dropout_input_m_3, w_3, axes=[[1], [1]])
    m_3_2 = tf.add(m_3_1, B_3)
    m_3 = tf.transpose(maxout(m_3_2, axis = 1))
    
    if iter_number == 3:
        tf.summary.histogram(scope + '/hmn_b_1', b_1)
        tf.summary.histogram(scope + '/hmn_w_1', w_1)
        tf.summary.histogram(scope + '/hmn_w_d', w_d)
        tf.summary.histogram(scope + '/hmn_w_3', w_3)
        tf.summary.histogram(scope + '/hmn_m_3', m_3)
    
    return m_3




if __name__ == '__main__':
    with tf.Session() as sess:
        print('Hello')