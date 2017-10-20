import numpy as np
import tensorflow as tf
#import maxout
import utils as ut

'''
Simple version of maxout. Reduces one dimension to 1
'''
def maxout(input, axis):
    return tf.reduce_max(input, axis=axis)


# Highway Maxout Network
'''
Args:
    U   is shape of (2L, m)
    h_i is shape of (L, 1)
    u_s_i, u_e_i are shape of (2L, 1)

Returns:
    tensor of shape (1, document_size) of scores
'''
def HMN(U, h_i, u_s_i, u_e_i, doc_size, scope = None, FLAGS = None):
    maxout_pooling_size = FLAGS.maxout_pooling_size
    lstm_size = FLAGS.lstm_size
    #doc_size = tf.shape(U)[1]
    scope = scope or tf.get_variable_scope()
    b_1 = ut.get_scope_variable(scope, 'hmn_b_1', [maxout_pooling_size, lstm_size])
    B_1 = tf.reshape(tf.tile(b_1, [doc_size, 1]), [maxout_pooling_size, lstm_size, doc_size])
    w_1 = ut.get_scope_variable(scope, 'hmn_w_1', [maxout_pooling_size, lstm_size, 3 * lstm_size])
    w_d = ut.get_scope_variable(scope, 'hmn_w_d', [lstm_size, 5*lstm_size])
    b_2 = ut.get_scope_variable(scope, 'hmn_b_2', [maxout_pooling_size, lstm_size])
    B_2 = tf.reshape(tf.tile(b_2, [doc_size, 1]), [maxout_pooling_size, lstm_size, doc_size])
    w_2 = ut.get_scope_variable(scope, 'hmn_w_2', [maxout_pooling_size, lstm_size, lstm_size])
    b_3 = ut.get_scope_variable(scope, 'hmn_b_3', [maxout_pooling_size])
    B_3 = tf.reshape(tf.tile(b_3, [doc_size]), [maxout_pooling_size, doc_size])
    w_3 = ut.get_scope_variable(scope, 'hmn_w_3', [maxout_pooling_size, 1, 2 * lstm_size])
    
    tf.summary.histogram(scope + '/hmn_b_1', b_1)
    tf.summary.histogram(scope + '/hmn_w_1', w_1)
    tf.summary.histogram(scope + '/hmn_w_d', w_d)
    tf.summary.histogram(scope + '/hmn_w_3', w_3)
    
    # r is shape of (l)
    r = tf.tanh(tf.matmul(w_d, tf.concat([h_i, u_s_i, u_e_i], axis = 0)));
    # R is shape (L, m)
    R = tf.reshape(tf.tile(r, [doc_size, 1]), [lstm_size, doc_size])
    # m_1_1 is shape of (p, L, m)
    m_1_1 = tf.tensordot(w_1, tf.concat([U, R], axis = 0), axes=[[2], [0]])
    # m_1_2 is shape of (p, L, m)
    m_1_2 = tf.add(m_1_1, B_1)
    # m_1 is shape of (L, m)
    m_1 = tf.squeeze(maxout(m_1_2, axis = 0 ))
    
    # m_2_1 is shape of (p, L, m)
    m_2_1 = tf.tensordot(w_2, m_1, axes=[[2], [0]])
    m_2_2 = tf.add(m_2_1, B_2)
    # m_2 is shape of (L, m)
    m_2 = tf.squeeze(maxout(m_2_2, axis = 0 ))
    
    m_3_1 = tf.tensordot(w_3, tf.concat([m_1, m_2], axis = 0), axes=[[2], [0]])
    m_3_2 = tf.add(tf.squeeze(m_3_1), B_3)
    m_3 = maxout(m_3_2, axis = 0)
    
    tf.summary.histogram(scope + '/hmn_m_3', m_3)
    return m_3


if __name__ == '__main__':
    with tf.Session() as sess:
        print('Hello')