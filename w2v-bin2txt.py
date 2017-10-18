from gensim.models.keyedvectors import KeyedVectors

w2v_model = KeyedVectors.load_word2vec_format('ruscorpora_1_300_10.bin', binary=True, encoding='utf-8')
w2v_model.save_word2vec_format('ruscorpora_1_300_10.txt', binary=False)