import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

#preprocessing text
def preprocessing(sent):
  import re

  sent = re.sub('\\xa0|\\n|\n|\t', ' ', sent)
  sent = re.sub("'", "’", sent)

  short_form = pd.read_csv('English_short_form.csv')
  for short_, long_ in zip(short_form['short form'], short_form['long form']):
    sent = re.sub(short_.lower(), long_, sent.lower())

  sent = re.sub('\s\s+', ' ', sent)

  converting_list = pd.read_csv('English_converting_list.csv')
  # remove URL
  sent_ = ''
  for token in sent.split():
    if re.search('http://|https://|www.|twitter.com', token):
      continue
    else:
      for short_, long_ in zip(converting_list['short form'], converting_list['long form']):
        if token.lower() == short_.lower():
          token = long_
      sent_ += token + ' '
  sent = sent_

  # remove emoticons [^a-zA-Z0-0]
  sent = re.sub('\W', ' ', sent)
  sent = re.sub('\s\s+', ' ', sent)
  sent = sent.strip()

  return sent.lower()

def load_data(filename):
    df = pd.read_csv(filename)
    df.dropna(axis=0, inplace=True)

    # category encoding
    from sklearn.preprocessing import LabelEncoder

    lb_encoder = LabelEncoder()
    df['label'] = lb_encoder.fit_transform(df['sentiment'])

    refine_text = []
    for i, text in enumerate(tqdm(df['text'], desc='preprocessing')):
        refine_text.append(preprocessing(text))
    df['refine_text'] = refine_text

    return df


def get_embedding_vector(refine_text, KEY):
    if KEY == 'new-embedding':
        new_embedding_list = pd.read_csv('prediction_embedding.csv')

        new_word_lower = []
        for w in new_embedding_list['new_word']:
            new_word_lower.append(w.lower())
        new_embedding_list['new_word_lower'] = new_word_lower

        # DataFrame to dictionary
        new_embedding_dict = new_embedding_list.set_index('new_word_lower')['embedding'].to_dict()

    X_vector = []
    new_word = []
    for i, sent in enumerate(tqdm(refine_text, desc='str2num', mininterval=60)):
        tokens = sent.split()
        vectors = []
        for token in tokens:
            if KEY == 'new-embedding' and token in new_word_lower:
                vector = new_embedding_dict.get(token)
            else:
                vector = embedding_dict.get(token)

            if vector is None:  # token is not in conceptnet
                new_word.append(token)
            else:
                vector = np.fromstring(vector, dtype=float, sep=' ')  # when return: embedding
                # vector = np.array(list(vector), dtype=float) # when return: embedding.split()
                vectors.append(vector)
        X_vector.append(vectors)

    print("be saving the vector into csv file")
    if KEY == 'new-embedding':
        df_vec = pd.DataFrame({'X_vector': X_vector})
        df_vec.to_csv('new-Embedding_vector.csv', index=False, header=False)
    else:  # KEY == 'conceptnet'
        df_vec = pd.DataFrame({'X_vector': X_vector})
        df_vec.to_csv('conceptnet_vector.csv', index=False, header=False)
    print("finish to save")
    return X_vector, new_word




train_df = load_data('train.csv')
#test_df = load_data('test.csv')

embedding_list = pd.read_csv('wordEmbedding_en.csv', encoding='utf-8', usecols=['word', 'vec300'], sep=' ')
embedding_dict = embedding_list.set_index('word')['vec300'].to_dict()

X, new_word = get_embedding_vector(train_df['refine_text'], KEY='conceptnet')
#X, new_word = get_embedding_vector(test_df['refine_text'], KEY='conceptnet')

df = pd.DataFrame({'new_word': new_word})
df.to_csv('conteptnet_new_word_list_test.csv', index=False, header=False)
