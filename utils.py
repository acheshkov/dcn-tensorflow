import numpy as np
import tensorflow as tf


# get or create variable in scope
def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
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


if __name__ == '__main__':
    with tf.Session() as sess:
        print('Hello')