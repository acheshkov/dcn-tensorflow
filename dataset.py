import numpy as np
import tensorflow as tf
import re
import pymorphy2
import string
import random
from gensim.models.wrappers import FastText
from gensim.models.keyedvectors import KeyedVectors
import sys

#embeddings = FastText.load_fasttext_format('ru.bin')
embeddings = KeyedVectors.load_word2vec_format('processed_ruscorpora_1_300_10.bin', binary=True)
morph = pymorphy2.MorphAnalyzer()
#filenames = ["./train_train_task_b.csv"]

# :: filename -> (float32, float32, float32) -> IO ()
# slit dataset file on train, validate, test
def splitDataset(filename, pbs=(0.6, 0.2, 0.2)):
    fin = open(filename, 'rb')
    f_train = open('train_' + filename, 'wb')
    f_valid = open('valid_' + filename, 'wb')
    f_test = open('test_' + filename, 'wb')
    for line in fin:
        r = random.random()
        if r < pbs[0]:
            f_train.write(line)
        elif r < pbs[0] + pbs[1]:
            f_test.write(line)
        else:
            f_valid.write(line)
            
    fin.close()
    f_train.close()
    f_valid.close()
    f_test.close()
    
# function to prepare ruscorpora embedding
def prepareDataset(filename):
    fin = open(filename, 'rb')
    fout = open('processed_' + filename, 'wb')
    for line in fin:
        ss = line.decode().split(' ')
        ss[0] = ss[0].split('_')[0]
        fout.write(' '.join(ss).encode())

            
    fin.close()
    fout.close()

# Returns byte[] without padding zeroes
def removePadding(byteString):
    return bytes(filter(lambda s: s != 0, byteString))

# the same as removePadding but works on list of byte[]
def removePaddingList(manyByteString):
    return list(map(removePadding, manyByteString))
    #return bytes(filter(lambda s: s != 0, byteString))

# parameters are byte[]
def contains(small, big):
    small = removePaddingList(small)
    big = removePaddingList(big)
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i + len(small) - 1
    return -1, -1



# Returns: string[]
def sentenceToTokens(sentence):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    sentence_without_punct = sentence.translate(table)
    sentence_without_punct_filtered = list(filter(lambda s: s!='', sentence_without_punct.split(" ")))
    return list(map(lambda s: morph.parse(s)[0].normal_form, sentence_without_punct_filtered))


def tokenize(sentence):
    decoded = sentence.decode()
    filtered = re.sub('[^ёa-яA-Яa-zA-Z0-9-_*.\s]', '', decoded)
    tokens = sentenceToTokens(filtered)
    return len(tokens), np.array(list(map(lambda s: s.encode('utf-8'), tokens)))

# :: byteString -> [float]
def word2vec(word):
    try:
        res = np.array(embeddings[removePadding(word).decode()], dtype=float)
    except:
        #print("err", removePadding(word).decode())
        res = np.array([0.0 for x in range(0, 300)], dtype=float)
    return res

# :: string -> [float]
def word2vec_onstring(word):
    try:
        res = np.array(embeddings[removePadding(word.encode()).decode()], dtype=float)
    except:
        #print("err", word)
        res = np.array([0.0 for x in range(0, 300)], dtype=float)
    return res

# :: [string] -> [[float32]]
def sentence2Vectors(sentence, max_sequence_size):
    arr = np.array(list(map(lambda s: word2vec(s), sentence[0:max_sequence_size])))
    l = len(sentence[0:max_sequence_size])
    padded = np.pad(arr, ((0, max_sequence_size - l), (0, 0)), 'constant')
    return padded

# :: [string] -> [[float32]]
def sentence2Vectors_onstring(sentence, max_sequence_size):
    arr = np.array(list(map(lambda s: word2vec_onstring(s), sentence[0:max_sequence_size])))
    l = len(sentence[0:max_sequence_size])
    padded = np.pad(arr, ((0, max_sequence_size - l), (0, 0)), 'constant')
    return padded


# :: string -> (int, int, [[string]], [[string]], [[double]], [[double]])
def processLineV2(max_sequence_size):
    def _processLine(str):

        start_pos, end_pos, doc, que = str.split(',')
        start_pos = int(start_pos)
        end_pos = int(end_pos)
        #dlen, document = tf.py_func(tokenize, [doc], (tf.int64, tf.string), name="doc_tok")
        #qlen, question = tf.py_func(tokenize, [q], (tf.int64, tf.string), name="que_tok")
        #alen, answer = tf.py_func(tokenize, [a], (tf.int64, tf.string), name="ans_tok")
        #print(answer)
        #start_pos, end_pos = tf.py_func(contains, [answer, document], (tf.int64, tf.int64), name="contains")
        document = doc.split(' ')
        question = que.split(' ');
        question_vec = tf.py_func(sentence2Vectors, [question, max_sequence_size], tf.float64, name="que_sentence2Vectors")
        document_vec = tf.py_func(sentence2Vectors, [document, max_sequence_size], tf.float64, name="doc_sentence2Vectors")
        question_vec.set_shape([max_sequence_size, 300]);
        document_vec.set_shape([max_sequence_size, 300]);

        return start_pos, end_pos, document, question, document_vec, question_vec
    return _processLine

# Parse line of preprocessed CSV dataset
def processCSVLine(str, max_doc_length, max_que_length):
    start_pos, end_pos, doc, que = str.split(';')
    start_pos = int(start_pos)
    end_pos = int(end_pos)
    document = doc.split(' ')
    question = que.split(' ')
    doc_v = sentence2Vectors_onstring(document, max_doc_length)
    que_v = sentence2Vectors_onstring(question, max_que_length)
    return start_pos, end_pos, document, question, doc_v, que_v


# :: string -> (int, int, [[string]], [[string]], [[double]], [[double]])
def processLine(max_sequence_size, max_que_size):
    def _processLine(str):
        try:
            did,qid,doc,q,a = tf.decode_csv(str, [[0], [0], ["empty"], [""], [""]])
            dlen, document = tf.py_func(tokenize, [doc], (tf.int64, tf.string), name="doc_tok")
            qlen, question = tf.py_func(tokenize, [q], (tf.int64, tf.string), name="que_tok")
            alen, answer = tf.py_func(tokenize, [a], (tf.int64, tf.string), name="ans_tok")
            #print(answer)
            start_pos, end_pos = tf.py_func(contains, [answer, document], (tf.int64, tf.int64), name="contains")
            question_vec = tf.py_func(sentence2Vectors, [question, max_que_size], tf.float64, name="que_sentence2Vectors")
            document_vec = tf.py_func(sentence2Vectors, [document, max_sequence_size], tf.float64, name="doc_sentence2Vectors")
            question_vec.set_shape([max_que_size, 300]);
            document_vec.set_shape([max_sequence_size, 300]);
        except:
            print('Error', str)
            return -1, -1, None, None, None, None, None
        return start_pos, end_pos, dlen, document, question, document_vec, question_vec
    return _processLine

     

def getDataset(filenames, max_sequence_size = 1000, max_que_size = 40):
    dataset = tf.contrib.data.TextLineDataset(filenames);
    dataset = dataset.skip(1)
    dataset = dataset.map(processLine(max_sequence_size, max_que_size))
    dataset = dataset.filter(lambda s,e,dlen,doc,que,doc_v,que_v: s >= 0 )
    dataset = dataset.filter(lambda s,e,dlen,doc,que,doc_v,que_v: dlen < max_sequence_size )
    dataset = dataset.map(lambda s,e,dlen,doc,que,doc_v,que_v: (s, e, doc, que, doc_v, que_v))
    #dataset = dataset.filter(lambda s,e,doc,que,doc_v,que_v: e < max_sequence_size - 1 )
    
    return dataset

def getDatasetV2(filenames, max_sequence_size = 1000):
    dataset = tf.contrib.data.TextLineDataset(filenames);
    dataset = dataset.map(processLineV2(max_sequence_size))
    #dataset = dataset.filter(lambda s,e,dlen,doc,que,doc_v,que_v: s >= 0 )
    #dataset = dataset.filter(lambda s,e,dlen,doc,que,doc_v,que_v: dlen < max_sequence_size )
    #dataset = dataset.map(lambda s,e,dlen,doc,que,doc_v,que_v: (s, e, doc, que, doc_v, que_v))
    #dataset = dataset.filter(lambda s,e,doc,que,doc_v,que_v: e < max_sequence_size - 1 )
    
    return dataset

# :: FileName -> Int -> Int -> Size -> [sample]
def readDatasetToMemory(filename, max_sequence_length, max_question_length, size = -1):
    dataset = []
    step = 0
    with open(filename) as hfile:
        for line in hfile:
            #try:
            start_true, end_true, doc, que, doc_v, que_v = processCSVLine(line, max_sequence_length, max_question_length)
            dataset.append((start_true, end_true, doc, que, doc_v, que_v))
            #except tf.errors.OutOfRangeError:
            #    print("End of dataset")  # ==> "End of dataset"
            #    break;
            #except: 
            #    print('Error read line', "skip");
            step = step + 1
            if size >=0 and step >= size: break
    print("Dataset Readed", sys.getsizeof(dataset) / 1024, 'KB')
    return dataset
