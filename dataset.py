import numpy as np
import tensorflow as tf
import re
import pymorphy2
import string
import random
from gensim.models.wrappers import FastText

embeddings = FastText.load_fasttext_format('ru.bin')
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
            f_valid.write(line)
        else:
            f_test.write(line)
            
    fin.close()
    f_train.close()
    f_valid.close()
    f_test.close()

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
    return np.array(list(map(lambda s: s.encode('utf-8'), tokens)))

# :: string -> [float]
def word2vec(word):
    try:
        res = np.array(embeddings[removePadding(word).decode()], dtype=float)
    except:
        #print("err", removePadding(word).decode())
        res = np.array([0.0 for x in range(0, 300)], dtype=float)
    return res

# :: [string] -> [[float32]]
def sentence2Vectors(sentence, max_sequence_size):
    arr = np.array(list(map(lambda s: word2vec(s), sentence[0:max_sequence_size])))
    l = len(sentence[0:max_sequence_size])
    padded = np.pad(arr, ((0, max_sequence_size - l), (0, 0)), 'constant')
    return padded



# :: string -> (int, int, [[string]], [[string]], [[double]], [[double]])
def processLine(max_sequence_size):
    def _processLine(str):
        did,qid,doc,q,a = tf.decode_csv(str, [[0], [0], ["empty"], [""], [""]])
        document = tf.py_func(tokenize, [doc], tf.string)
        question = tf.py_func(tokenize, [q], tf.string)
        answer = tf.py_func(tokenize, [a], tf.string)
        #print(answer)
        start_pos, end_pos = tf.py_func(contains, [answer, document], (tf.int64, tf.int64))
        question_vec = tf.py_func(sentence2Vectors, [question, max_sequence_size], tf.float64)
        document_vec = tf.py_func(sentence2Vectors, [document, max_sequence_size], tf.float64)
        question_vec.set_shape([max_sequence_size, 300]);
        document_vec.set_shape([max_sequence_size, 300]);
        return start_pos, end_pos, document, question, document_vec, question_vec
    return _processLine

     

def getDataset(filenames, max_sequence_size = 1000):
    dataset = tf.contrib.data.TextLineDataset(filenames);
    dataset = dataset.skip(1)
    dataset = dataset.map(processLine(max_sequence_size))
    #dataset = dataset.filter(lambda s,e,doc,que,doc_v,que_v: (s >= 0 ))
    
    return dataset
