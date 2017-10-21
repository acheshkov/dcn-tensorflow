import numpy as np
import tensorflow as tf
from collections import Counter
import re


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


# non-linear projection layer
def non_linear_projection(x):
    scope = tf.get_variable_scope()
    w = get_scope_variable(scope, 'p_w', x.get_shape())
    b = get_scope_variable(scope, 'p_b', x.get_shape())
    return tf.add(tf.multiply(x, w), b)


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

if __name__ == '__main__':
    with tf.Session() as sess:
        print('Hello')