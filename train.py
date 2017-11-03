import numpy as np
import tensorflow as tf
from itertools import islice
import dataset as ds
import time
import utils


def processLine(str, max_doc_length, max_que_length):
    start_pos, end_pos, doc, que = str.split(';')
    start_pos = int(start_pos)
    end_pos = int(end_pos)
    document = doc.split(' ')
    question = que.split(' ')
    doc_v = ds.sentence2Vectors_onstring(document, max_doc_length)
    que_v = ds.sentence2Vectors_onstring(question, max_que_length)
    return start_pos, end_pos, document, question, doc_v, que_v


def processLineBatch(file, batch_size, max_sequence_length, max_question_length, 
                     question_ph, document_ph, dropout_rate_ph,
                     doc_len_ph, que_len_ph, start_true_ph, end_true_ph, batch_size_ph, dropout_rate):
    next_n_lines = list(islice(file, batch_size))
    if not next_n_lines or len(next_n_lines) != batch_size: return None
    q = []
    d = []
    s = []
    e = []
    dl = []
    ql = []
    batch_size_fact = 0;
    for line_ in next_n_lines:
        start_pos, end_pos, document, question, doc_v, que_v = processLine(line_, max_sequence_length, max_question_length)
        if len(document) > max_sequence_length or  len(question) >= max_question_length or start_pos < 0: 
            print("Wrong example. Skip", document[0])
            continue;
        q.append(que_v)
        d.append(doc_v)
        s.append(start_pos)
        e.append(end_pos)
        dl.append(len(document))
        ql.append(len(question))
        batch_size_fact += 1

    feed_dict = {
        question_ph: q, 
        document_ph: d, 
        start_true_ph: s,
        end_true_ph: e,
        doc_len_ph: dl,
        que_len_ph: ql,
        dropout_rate_ph: dropout_rate,
        batch_size_ph: batch_size_fact
        
    }
    if batch_size_fact != batch_size:  print("Batch Size Fact", batch_size_fact)
    return feed_dict

def accuracyTest(sess, params, writer, accuracy_avg, summary_op, summary_op_test, pr_start_idx, pr_end_idx, step):
    try:
        acc, stat, stat_test, s, e = sess.run(
            (accuracy_avg, summary_op, summary_op_test, pr_start_idx, pr_end_idx),
            params
        )
        writer.add_summary(stat_test,  step)
        print('AVG accuracy', acc)
    except: 
        print("Test Error", params)


def trainStep(sess, feed_dict, writer, 
              train_step, sum_loss, accuracy_avg, summary_op, 
              summary_op_train, step, profiling = False):
    
    run_options = None
    run_metadata = None
    if profiling:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    start_time = time.time()
    try:
        _,loss, _, stat, stat_train = sess.run(
            (train_step, sum_loss, accuracy_avg,  summary_op, summary_op_train),
            feed_dict = feed_dict,
            options=run_options, run_metadata=run_metadata
        )
        writer.add_summary(stat, step)
        writer.add_summary(stat_train, step)
    except:
        print("Train Error", step)
        

def loss_and_accuracy(start_true, end_true, batch_size, sum_start_scores, sum_end_scores, max_sequence_length):
    # loss and train step
    onehot_labels_start = tf.one_hot(start_true, max_sequence_length)
    onehot_labels_end   = tf.one_hot(end_true, max_sequence_length)

    with tf.name_scope('Loss'):
        loss_start = tf.nn.softmax_cross_entropy_with_logits(
            labels = onehot_labels_start,
            logits = sum_start_scores)
        loss_start = tf.reduce_mean(loss_start)
        loss_end = tf.nn.softmax_cross_entropy_with_logits(
            labels = onehot_labels_end,
            logits = sum_end_scores)
        loss_end = tf.reduce_mean(loss_end)
        sum_loss = loss_start + loss_end
        
    with tf.name_scope('Accuracy'):
        with tf.name_scope('Prediction'):
            pr_start_idx = tf.to_int32(tf.argmax(sum_start_scores, 1))
            pr_end_idx = tf.to_int32(tf.argmax(sum_end_scores, 1))

        with tf.name_scope('Accuracy'):
            accuracy_avg = tf.py_func(utils.f1_score_int_avg, [pr_start_idx, pr_end_idx, start_true, end_true], tf.float64)
    
    return sum_loss, accuracy_avg,pr_start_idx, pr_end_idx


    