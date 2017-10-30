import numpy as np
import tensorflow as tf
from itertools import islice
import dataset as ds
import time


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
                     doc_len_ph, que_len_ph, start_true_ph, end_true_ph, dropout_rate):
    next_n_lines = list(islice(file, batch_size))
    if not next_n_lines or len(next_n_lines) != batch_size: return None
    q = []
    d = []
    s = []
    e = []
    dl = []
    ql = []
    for line_ in next_n_lines:
        start_pos, end_pos, document, question, doc_v, que_v = processLine(line_, max_sequence_length, max_question_length)
        q.append(que_v)
        d.append(doc_v)
        s.append(start_pos)
        e.append(end_pos)
        dl.append(len(document))
        ql.append(len(question))

    feed_dict = {
        question_ph: q, 
        document_ph: d, 
        start_true_ph: s,
        end_true_ph: e,
        doc_len_ph: dl,
        que_len_ph: ql,
        dropout_rate_ph: dropout_rate
    }
    return feed_dict

def accuracyTest(sess, params, writer, accuracy_avg, summary_op, summary_op_test, pr_start_idx, pr_end_idx, step):
    try:
        acc, stat, stat_test, s, e = sess.run(
            (accuracy_avg, summary_op, summary_op_test, pr_start_idx, pr_end_idx),
            params
        )
        #print('Predicted answer', utils.substr(doc, s, e))
        #print('True answer', utils.substr(doc, start_true, end_true))
        #writer.add_summary(stat,  step* 10 + step_accuracy_)
        #print("acc", s, e, start_true, end_true)
        #acc_accum += acc;
        #print("acc:", acc, "Total:", acc_accum)

        writer.add_summary(stat_test,  step)
        #writer.add_summary(stat,  step)
        print('AVG accuracy', acc)
    except: 
        print("Test Error", params)
    #return stat_test

def trainStep(sess, feed_dict, writer, 
              train_step, sum_loss, accuracy_avg, summary_op, 
              summary_op_train, step, profiling = False):
    
    #global last_avg_accuracy
    #start_true, end_true, doc, que, doc_v, que_v = sess.run(next_element)
    #if start_true < 0 or end_true > max_sequence_length - 1: 
    #    print('Ignore step', start_true, end_true)
    #    return
    
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
        print("Train Error", feed_dict)
        
    #print("--- Train step  %s seconds ---" % (time.time() - start_time))
    #if step % 25 == 0 : print(step, loss, start_true, end_true)
    #start_time = time.time()
    #if profiling: writer.add_run_metadata(run_metadata, 'step%d' % step)
    #print("---State Writing  %s seconds ---" % (time.time() - start_time))
    
    #return stat, stat_train