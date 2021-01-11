import pandas as pd
import numpy as np
import requests
import json
import os
import csv
import copy

import nltk
nltk.download('wordnet')
#from nltk.stem import WordNetLemmatizer
import re


def preprocessing(data):
    # 불필요한 문자(열) 제거
    stopword = ['(adjective)', '(noun)', '(adverb)', '(verb)',
                '/colour', '/harbour', "'s", "’s", ";", ":", "[", "]",
                '(', ')', '-', "'", '"',"’", '/', ".", '“', '”', '?']
    for s in stopword:
        data = data.replace(s, ' ')

    data = ' '.join(data.split())  # 중복 공백 제거
    data = data.lower()  # 모두 소문자로

    # Lemmatization 원형 (lemma) 찾기
    lem = nltk.WordNetLemmatizer()
    lemmatized_data = ''
    for word in data.split(' '):
        new_word = lem.lemmatize(word)
        lemmatized_data += ' ' + new_word

    lemmatized_data = ' '.join(lemmatized_data.split())  # 중복 공백 제거
    return lemmatized_data


# Referrence:
# https://stackoverflow.com/questions/53801998/python-json-request-shows-a-keyerror-even-though-key-exists
url = 'http://192.9.24.248:19002/query/service'
def get_vec300(word):
  statement = 'use conceptnet; '
  statement += 'select vec300 from numberbatch_en where word="'+word+'";'

  # 서버에 request를 보내고 응답을 받아야 하는데 서버의 상태가 불안정하여
  # 간혹 DB에 데이터가 있음에도 응답이 안올 때가 있음. 이를 대비하여 10번 request를 시도해봄
  for i in range(0, 10):
      try:
          req = requests.post(url, data={'statement': statement})
          if req.status_code == 200:
              result = req.json()['results']
              if len(result) >= 1:
                  return result[0]['vec300'].split(' ')
      except KeyError:
          print("[KeyError] status code: " + str(req.status_code) + "\tword: " + word)

  print("[No Data in DB] word: " + word)
  # 10번 동안 결과가 없을 경우, 검색 단어에 해당하는 임베딩 벡터가 DB에 없는 것으로 간주하여 빈 리스트 반환
  return []


def get_vectors(data):
    if isinstance(data, list):
        vectors = []
        new_data = copy.deepcopy(data)
        #print("[get_vectors] data length: " + str(len(data)))
        for i, d in enumerate(data):
            temp = get_vec300(d)
            if temp == []:
                #del data[data.index(i)]
                new_data.remove(d) # delete that data(token) in data set(data)
                #print("remove " + d + "\tindex " + str(i))
            else:
                vectors.append(np.array(temp))
        #print("[get_vectors] data length: "+str(len(new_data)))
        #print("[get_vectors] vector length: " + str(len(vectors)))
        return (new_data, vectors)
    else:
        temp = get_vec300(data)
        if temp == []: # if there's no result (dense vectors) from the DB
            return (data, None) # there's no ground truth
        else:
            return (data, temp)



# Load and Read data from a csv file
# When using 'english_words_1009.csv' file
df = pd.read_csv('english_words_1009.csv', delimiter=',',
        encoding='utf-8', usecols=['word','definition'], index_col=None)
total_dataset = len(df) # 1009

print("total dataset length: " + str(total_dataset))

# Shuffle all data
df = df.sample(frac=1).reset_index(drop=True)

# Make new folders and set file path to save data
filepath = os.path.join(os.path.curdir + "/dataset" + "/english_1009")
if not (os.path.isdir(filepath)):
    os.makedirs(filepath)

file_idx = 0
for i in range(0, len(df)):
    # get a row from the dataframe
    data = df.iloc[i, :]

    x, y = data['definition'], data['word']
    x_tokens = preprocessing(x).split(' ')
    y = preprocessing(y)

    # convert word(s) into dense vectors (conceptnet numberbatch)
    y, y_vectors = get_vectors(y)
    if y_vectors is not None:
        # get 300 dims vectors from word embedding (conceptnet numberbatch stored in AsterixDB)
        x_tokens, x_vectors = get_vectors(x_tokens)

        # make a DataFrame about english words
        words = x_tokens.copy()
        words.append(y)
        df_word = pd.DataFrame(words)

        # save words to csv file using DataFrame
        filename_w = os.path.join(filepath, "words" + str(file_idx) + ".csv")
        df_word.to_csv(filename_w, sep=',', index=False, header=False)

        # make a DataFrame about vectors
        vectors = x_vectors.copy()
        vectors.append(y_vectors)
        df_vec = pd.DataFrame(np.array(vectors))

        # Save vectors to csv file using DataFrame
        filename_v = os.path.join(filepath, "vectors" + str(file_idx) + ".csv")
        df_vec.to_csv(filename_v, sep=',', index=False, header=False)
        #vectors = np.array(vectors)
        #np.savetxt(filename_v, vectors, fmt='%.4f', delimiter=',')

        print("y label word: " + y + "\tword length: " + str(len(words)) + "\t\tvector length: " + str(len(vectors)))
        file_idx += 1
    else:
        print("y label word: " + y + "no word embedding value!")
        # if the result of y_input is None, there's no ground truth.
        # Even if the model can predict a value, we can't get an accuracy and calculate a loss.
        # Therefore do NOT anything!

