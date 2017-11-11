import numpy as np
import tensorflow as tf
from collections import Counter
import re
from functools import reduce


# get or create variable in scope
def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            #print('new var created')
            v = tf.get_variable(var, shape)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v

# non-linear projection layer batched version
def non_linear_projection_batch(x, shape_w, shape_b, batch_size):
    scope = tf.get_variable_scope()
    w = get_scope_variable(scope, 'p_w', shape_w)
    b = get_scope_variable(scope, 'p_b', shape_b)
    w = tf.reshape(w, [1] + shape_w)
    w = tf.tile(w, [batch_size, 1, 1])
    b = tf.reshape(b, [1] + shape_b)
    b = tf.tile(b, [batch_size, 1, 1])
    return tf.tanh(tf.add(tf.matmul(w, x), b))


# non-linear projection layer
def non_linear_projection(x):
    scope = tf.get_variable_scope()
    w = get_scope_variable(scope, 'p_w', x.get_shape())
    b = get_scope_variable(scope, 'p_b', x.get_shape())
    return tf.tanh(tf.add(tf.multiply(x, w), b))


def make_h_param_string(lr, lstm_size, max_seq_len, maxout_pooling_size, dataset_len, batch_size):
    return ';'.join([
        'LR=' + str(lr), 
        'LSTM_S=' + str(lstm_size), 
        'MAX_SL=' + str(max_seq_len), 
        'MAXOUT=' + str(maxout_pooling_size),
        'DS_LEN=' + str(dataset_len) ,
        'BS=' + str(batch_size) 
    ])

def make_h_param_string_2(hparams):
    return ';'.join([
        'BS=' + str(hparams['batch_size']), 
        'DRP=' + str(hparams['dropout_rate']),
        'LR=' + str(hparams['learning_rate'])
    ])

def normalize_answer(text):
    """Lower text and remove punctuation and extra whitespace."""
    return ' '.join(re.findall(r"\w+", text)).lower()

# :: [byte[]] -> [string]
def b2s(bs):
    return list(map(lambda s: s.decode(), bs))

# [byte[]] -> int -> int -> string
def substr(doc, start, end):
    return ' '.join(b2s(doc[start:end + 1]))

# :: string -> string -> float
def f1_score_string(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score_int(s, e, s_true, e_true):
    truth = range(s_true, e_true + 1)
    pred = range(s, e + 1)
    num_same = len(set(pred) & set(truth))
    if num_same == 0:
        return float(0)
    precision = 1.0 * num_same / len(pred)
    recall = 1.0 * num_same / len(truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)

def f1_score_int_avg(s, e, s_true, e_true):
    l = list(map(lambda t: f1_score_int(t[0], t[1], t[2], t[3]), zip(s, e, s_true, e_true)))
    return reduce(lambda x, y: x + y, l) / len(l)

if __name__ == '__main__':
    with tf.Session() as sess:
        print('Hello')