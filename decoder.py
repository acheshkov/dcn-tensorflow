import numpy as np
import tensorflow as tf
import highway_maxout as hmn


def decoderBatch(U, lstm_dec, dropout_rate, batch_size, FLAGS):
    max_sequence_length = FLAGS.max_sequence_length
    max_decoder_iterations = FLAGS.max_decoder_iterations
    #batch_size = FLAGS.train_batch_size
    lstm_size = FLAGS.lstm_size
    
    with tf.name_scope('DYNAMIC_POINTING_DECODER'):
        lstm_dec_state = lstm_dec.zero_state(batch_size, tf.float32)
        start_pos = tf.zeros(shape=[batch_size], dtype=tf.int32)
        end_pos = tf.zeros(shape=[batch_size], dtype=tf.int32)
        sum_start_scores = tf.zeros([batch_size, max_sequence_length])
        sum_end_scores = tf.zeros([batch_size, max_sequence_length])
        
        for i_ in range(max_decoder_iterations):
            scores_start, scores_end, start_pos, end_pos, lstm_dec_state = decoderIteration(U, 
                                                                                                    lstm_dec_state, 
                                                                                                    start_pos, 
                                                                                                    end_pos,
                                                                                                    lstm_dec,
                                                                                                    dropout_rate,
                                                                                                    max_sequence_length,
                                                                                                    lstm_size,
                                                                                                    FLAGS,
                                                                                                    batch_size,
                                                                                                    i_)
            sum_start_scores = tf.add(sum_start_scores, scores_start)
            sum_end_scores   = tf.add(sum_start_scores, scores_end)
            
            
    return sum_start_scores, sum_end_scores
    
    
    

'''
U  (batch, D, 2L)
scores_start  [batch, D]
scores_end  [batch, D]
start_pos  [batch]
end_pos  [batch]
new_lstm_state [batch, L, 1]
# returns batched tuple (scores_start, scores_end, start_pos, end_pos, new_lstm_state)
'''
def decoderIteration(U, lstm_state, start_pos, end_pos, lstm_dec, dropout_rate, max_sequence_length, lstm_size, FLAGS, batch_size, iter_number):
    # returns (batch, 2L)
    def _getPos(U, start_positions, rows_size, vec_size):
        def createMask(pos, size, vector):
            return tf.pad(vector, [[pos, size - pos - 1], [0, 0]])
        
        ones = tf.ones([1, vec_size], tf.float32)
        # (batch, D, 2L)
        mask_matrix = tf.map_fn(lambda pos: createMask(pos, rows_size, ones), start_positions, dtype=tf.float32)
        positions = tf.multiply(U, mask_matrix)
        
        # (batch, 2L)
        res = tf.reduce_sum(positions, 1)
        res.set_shape([None, vec_size])
        return res;
        
    with tf.name_scope('Decoder_Iteration'):
        with tf.name_scope('Next_Start'):

            scores_start = hmn.HMN_Batch(U, 
                                lstm_state.h, 
                                _getPos(U, start_pos, max_sequence_length, lstm_size * 2),
                                _getPos(U, end_pos,   max_sequence_length, lstm_size * 2),
                                batch_size,
                                'start',
                                FLAGS,
                                dropout_rate,
                                iter_number)

            new_start_pos = tf.to_int32(tf.argmax(scores_start, 1))
        with tf.name_scope('Next_End'):
            scores_end = hmn.HMN_Batch(U, 
                             lstm_state.h,
                             _getPos(U, new_start_pos, max_sequence_length, lstm_size * 2),
                             _getPos(U, end_pos,   max_sequence_length, lstm_size * 2),
                             batch_size,
                             'end',
                             FLAGS,
                             dropout_rate,
                             iter_number)
            new_end_pos = tf.to_int32(tf.argmax(scores_end, 1))
        
        with tf.name_scope('LSTM_State_Update'):
            lstm_input = tf.concat(
                [
                    _getPos(U, new_start_pos, max_sequence_length, lstm_size * 2), 
                    _getPos(U, new_end_pos, max_sequence_length, lstm_size * 2)
                ],
                axis = 1
            )
 
            output, new_lstm_state = lstm_dec(lstm_input, lstm_state)

        return scores_start, scores_end, new_start_pos , new_end_pos, new_lstm_state

    

