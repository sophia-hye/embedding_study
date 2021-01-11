#@ Title: Make a dataset
#@ Author: Jihye Kim (Gachon)
#@ Date: 2020.06.08
#@ Version: 1.0
#@ Description: Make a word embedding dataset and store a csv file

#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import requests
import json
import os
import re
import csv
import copy

short_form = pd.read_csv('./dataset/English_short_form.csv')

def preprocessing(data):
    if type(data) == float:
        return ''
    else:
        for short_, long_ in zip(short_form['short form'], short_form['long form']):
            data = re.sub(short_.lower(), long_, data.lower())
        data = data.replace("'s", ' ')  # 소유격
        data = re.sub('[!|?|,|(|)|:|—]', ' ', data)
        #data = re.sub('\W', ' ', data)  # [^a-zA-Z0-9]
        data = re.sub('[+|-]', '_', data)  # (컨셉넷에서는 띄어쓰기의 경우 언더바로 표기함)
        # glove 또는 word2vec 같은 경우, 대소문자를 구분이 있으며 
        # 띄어쓰기의 경우 -로 표기하기 때문에 -를 _로 바꿔주는 전처리가 필요없음
        data = re.sub('\s\s+', ' ', data)
        data = data.strip()
        data = data.lower()  # 소문자로 바꾸기 (컨셉넷)
        return (data)

url = 'http://192.9.24.248:19002/query/service'
def get_vec300(word):
    statement = 'use conceptnet;'
    statement += 'select vec300 from numberbatch_en where word="'+word+'";'

    # since server state is unstable, response from the server sometimes is nothing.
    # for avoiding this problem, try same requests upto 10 times
    for i in range(0, 10):
        try:
            req = requests.post(url, data={'statement': statement})
            if req.status_code == 200:
                result = req.json()['results']
                if len(result) >= 1:
                    return result[0]['vec300'].split()
        except KeyError:
            # KeyError is happened
            a=1
    
    # consider no result with no matched data during 10 times
    # if no matched data, return empty list
    return []

def get_conceptnet(word):
    vector = embedding_dict.get(word)
    if vector is None:
        return []
    else:
        vector = np.fromstring(vector, dtype=float, sep=' ')
        return vector

def get_w2v(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return []

def get_glove(word):
    vector = glove_embedding_dict.get(word)
    if vector is None:
        return []
    else:
        return vector

def get_vectors(data):
    # if data is x_tokens (words)
    if isinstance(data, list):
        vectors = []
        new_data = copy.deepcopy(data)
        for i, d in enumerate(data):
            #temp = get_vec300(d)  # conceptnet AsterixDB
            temp = get_conceptnet(d)  # conceptnet
            #temp = get_w2v(d)  # word2vec
            #temp = get_glove(d)  # glove
            if temp == []:
                # delete that data(token) in copied data set(new_data)
                new_data.remove(d)
            else:
                vectors.append(np.array(temp))
        return (new_data, vectors)
    else: # if data is y (word)
        temp = get_vec300(data)
        # if there's no result (dense vectors) from the DB
        if temp == []:
            return (data, None) # there's no ground truth
        else:
            return (data, temp)


# --------------------------------------------------------
embedding_list = pd.read_csv('./dataset/wordEmbedding_en.csv', encoding='utf-8', usecols=['word', 'vec300'], sep=' ')
embedding_dict = embedding_list.set_index('word')['vec300'].to_dict()
"""
# --------------------------------------------------------
import gensim
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./dataset/GoogleNews-vectors-negative300.bin.gz', binary=True)

# --------------------------------------------------------
glove_embedding_dict = dict()
f = open('./dataset/glove.6B.300d.txt', encoding="utf8")

for line in f:
    splited = line.split()
    word = splited[0]
    vector_arr = np.asarray(splited[1:], dtype='float32')
    glove_embedding_dict[word] = vector_arr
f.close()
"""
# ------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('./dictionary_merriam_webster/merriam_webster_word_def.csv')
total_dataset = len(df) # 23870
print('total dataset: {}'.format(total_dataset))

# make new folders and set file path to save data
filepath = os.path.join(os.path.curdir + '/dataset' +'/english_23870_conceptnet')
if not (os.path.isdir(filepath)):
    os.makedirs(filepath)

filepath_w = os.path.join(filepath + '/words')
if not (os.path.isdir(filepath_w)):
    os.makedirs(filepath_w)

filepath_v = os.path.join(filepath + '/vectors')
if not (os.path.isdir(filepath_v)):
    os.makedirs(filepath_v)


file_idx = 0
for i in range(0, len(df)):
    # get a row from the dataframe
    data = df.iloc[i, :]
    X, y = data['definition'], data['word']
    
    y, y_vectors = get_vectors(preprocessing(y))
    # convert word into dense vectors; conceptnet numberbatch
    if y_vectors is not None:
        # check if X is float or not
        try:
            x_tokens = preprocessing(X).split()

            # get 300 dims vectors from conceptnet word embedding
            # conceptnet numberbatch is stored in AsterixDB
            x_tokens, x_vectors = get_vectors(x_tokens)

            # make a DataFrame about english words
            words = x_tokens.copy()
            words.append(y)
            df_word = pd.DataFrame(words)

            # save words to csv file using DataFrame
            filename_w = os.path.join(filepath_w, 'words'+str(file_idx)+'.csv')
            df_word.to_csv(filename_w, sep=',', index=False, header=False)

            # make a DataFrame about vectors
            vectors = x_vectors.copy()
            vectors.append(y_vectors)
            df_vec = pd.DataFrame(np.array(vectors))

            # save vectors to csv file using DataFrame
            filename_v = os.path.join(filepath_v, 'vectors'+str(file_idx)+'.csv')
            df_vec.to_csv(filename_v, sep=',', index=False, header=False)

            file_idx += 1
        except AttributeError:
            print('\n[idx: {:5d}]\t[AttributeError]\t:', X, '\n')
    else:
        print('[idx: {:5d}]\t{} has no word embedding!'.format(i, y))

