import numpy as np
import tensorflow as tf
import random
import utils
import dataset
import train as tr

class ModelTrainer:
    def __init__(self, embeddings_file, log_path):
        self.n = 1
        self.log_path = log_path
        self.session = tf.Session()
        self.summary_op_train = tf.summary.merge_all("TRAIN_STAT")
        self.summary_op_test = tf.summary.merge_all("TEST_STAT")
        self.summary_op = tf.summary.merge_all()
        self.init_variables = tf.global_variables_initializer()
        self.embeddings = dataset.Embeddings(embeddings_file)
    
    def train(self, hparams, filename, DATASET_LENGTH, step_ = 0):
        h_param_str = utils.make_h_param_string_2(hparams)
        
        
        dropout = hparams['dropout_rate']
        learning_rate = hparams['learning_rate']
        BATCH_SIZE = hparams['batch_size']
        #step_ = 0
        print("dropout = ", dropout, "; batch_size = ", BATCH_SIZE, '; lrate=', learning_rate)
        processed_items = 0;
        with open(filename) as file_train:
            while True:
                feed_dict = tr.processLineBatch(file_train, self.embeddings, BATCH_SIZE, 
                                                self.max_sequence_length, self.max_question_length, 
                                                self.question_ph, self.document_ph, self.dropout_rate_ph,
                                                self.doc_len_ph, self.que_len_ph, self.start_true_ph, self.end_true_ph,
                                                self.batch_size_ph,
                                                self.learning_rate_ph,
                                                dropout, learning_rate)
                if feed_dict is None: break
                print("TRAIN STEP:", step_)
                tr.trainStep(self.session, 
                             feed_dict, 
                             self.writer, 
                             self.train_step_op, 
                             self.accuracy_avg_op, 
                             self.summary_op, 
                             #self.summary_op_train, 
                             step_, profiling=False)
                processed_items += BATCH_SIZE
                step_+= 1
                if (processed_items >= DATASET_LENGTH): break;
        
        return step_;
    
    def accuracyTrain(self, hparams, filename, step, batch_size = 100000):
        with open(filename) as file:
            test_params = tr.processLineBatch(file, self.embeddings, batch_size, 
                                                  self.max_sequence_length, self.max_question_length, 
                                                  self.question_ph, self.document_ph, self.dropout_rate_ph,
                                                  self.doc_len_ph, self.que_len_ph, self.start_true_ph, self.end_true_ph,
                                                  self.batch_size_ph,
                                                  self.learning_rate_ph,
                                                  1, 0)
            tr.accuracyTest(self.session, 
                            test_params, 
                            self.writer, 
                            self.accuracy_avg_op, 
                            self.summary_op_train, 
                            self.pr_start_idx_op, 
                            self.pr_end_idx_op, step)
        return 0;
    
    def accuracyValid(self, hparams, filename, step, batch_size = 100000):
        with open(filename) as file:
            test_params = tr.processLineBatch(file, self.embeddings, batch_size, 
                                                  self.max_sequence_length, self.max_question_length, 
                                                  self.question_ph, self.document_ph, self.dropout_rate_ph,
                                                  self.doc_len_ph, self.que_len_ph, self.start_true_ph, self.end_true_ph,
                                                  self.batch_size_ph,
                                                  self.learning_rate_ph,
                                                  1, 0)
            tr.accuracyTest(self.session, 
                            test_params, 
                            self.writer, 
                            self.accuracy_avg_op,  
                            self.summary_op_test, 
                            self.pr_start_idx_op, 
                            self.pr_end_idx_op, step)
        return 0;
    
    def cross_validation_training(self, hparams, filename):
        return 0;
    
    def reset(self, hparams):
        h_param_str = utils.make_h_param_string_2(hparams)
        self.writer = tf.summary.FileWriter(self.log_path + "/" + str(self.n) + "-" + h_param_str, self.session.graph)
        self.n = self.n +1
        self.session.run(self.init_variables)
        return 0;
    
    def set_ops(self, ops):
        self.train_step_op = ops['train_step_op'] 
        self.sum_loss_op = ops['sum_loss_op'] 
        self.accuracy_avg_op = ops['accuracy_avg_op']
        self.pr_start_idx_op = ops['pr_start_idx_op'] 
        self.pr_end_idx_op = ops['pr_end_idx_op']
        return 0
    
    def set_variables(self, variables):
        self.max_sequence_length = variables['max_sequence_length']
        self.max_question_length = variables['max_question_length']
        #self.learning_rate = variables['learning_rate']
        
        self.question_ph = variables['question_ph']
        self.document_ph = variables['document_ph']
        self.dropout_rate_ph = variables['dropout_rate_ph']
        self.doc_len_ph = variables['doc_len_ph']
        self.que_len_ph = variables['que_len_ph']
        self.start_true_ph = variables['start_true_ph']
        self.end_true_ph = variables['end_true_ph']
        self.batch_size_ph = variables['batch_size_ph']
        self.learning_rate_ph = variables['learning_rate_ph']
        
        return 0;
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Release ModelTrainer")
        self.session.close()
        
        
class HyperParamsSpace:
    def __init__(self, dropouts, batches, lrates):
        self.dropouts = dropouts
        self.batches = batches
        self.lrates = lrates
        
    def getRand(self):
        return {
            'dropout_rate': random.choice(self.dropouts) ,
            'batch_size': random.choice(self.batches),
            'learning_rate': random.choice(self.lrates)
        }

        
