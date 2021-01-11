import os
import pandas as pd
import numpy as np
import gensim

def get_embedding_vector(KEY, tokens):
    vectors = []
    for token in tokens:
        vector = None

        if KEY == 'conceptnet':
            vector = conceptnet_embedding_dict.get(token)
        elif KEY == 'glove':
            vector = glove_embedding_dict.get(token)
        elif KEY == 'w2v':
            if token in w2v_embedding_dict:
                vector = w2v_embedding_dict[token]

        if vector is not None:
            if KEY == 'conceptnet':
                vector = np.fromstring(vector, dtype=float, sep=' ')  # when return: embedding
            vectors.append(vector)
    return vectors


if __name__ == '__main__':
    new_word_list = pd.read_csv('./dataset/keyword_definition.csv')  # column: keyword, definition

    # ConceptNet Numberbatch
    """
    conceptnet_embedding_list = pd.read_csv('./dataset/wordEmbedding_en.csv',
                                            encoding='utf-8', usecols=['word', 'vec300'], sep=' ')
    conceptnet_embedding_dict = conceptnet_embedding_list.set_index('word')['vec300'].to_dict()
    

    # GloVe (Load pre-trained 300dims GloVe Embedding data)
    glove_embedding_dict = dict()
    f = open('./dataset/glove.6B.300d.txt', encoding="utf8")

    for line in f:
        splited = line.split()
        word = splited[0]
        vector_arr = np.asarray(splited[1:], dtype='float32')
        glove_embedding_dict[word] = vector_arr
    f.close()
    """
    # Word2Vec (Load pre-trained 300dims Word2Vec Embedding data)
    w2v_embedding_dict = gensim.models.KeyedVectors.load_word2vec_format('./dataset/GoogleNews-vectors-negative300.bin.gz',
                                                                         binary=True)

    for i, (word, definition) in enumerate(zip(new_word_list['keyword'], new_word_list['definition'])):
        try:
            print(i, word)
            tokens = definition.lower().split()
            vectors = get_embedding_vector('w2v', tokens)

            # make a DataFrame about vectors
            df_vec = pd.DataFrame(np.array(vectors))

            # save vectors to csv file using DataFrame
            fpath = "/home/ida/workspace/jihye/dataset/input_vector_keyword_w2v/"
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            word = word.replace(' ', '_')
            fname = word + '_vector.csv'
            full_fname = fpath + fname

            df_vec.to_csv(full_fname, sep=',', index=False, header=False)
        except Exception as e:
            print("ERROR: ")
            print(e)
