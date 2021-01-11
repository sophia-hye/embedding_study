import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(filename, column_names=None):
  file_type = filename.split('.')[-1]
  if file_type == 'csv':
    df = pd.read_csv(filename)
  elif file_type == 'tsv':
    df = pd.read_csv(filename, sep="\t", names=column_names)
  else:
    print("Not supported file type: ", file_type)
    return None
  df.dropna(axis=0, inplace=True)

  return df

def label_encoding(label_column_name, df):
  # category encoding
  from sklearn.preprocessing import LabelEncoder

  lb_encoder = LabelEncoder()
  df['label'] = lb_encoder.fit_transform(df[label_column_name])

def get_new_words_list(sent_data):
  embedding_list = pd.read_csv('wordEmbedding_en.csv', usecols=['word', 'vec300'], sep=' ')
  embedding_dict = embedding_list.set_index('word')['vec300'].to_dict()

  new_words = {}
  for i, sent in enumerate(sent_data):
    tokens = sent.lower().split()
    for token in tokens:
      vector = embedding_dict.get(token)
      if vector is None:
        if new_words.get(token):
          new_words[token] += 1
        else:
          new_words[token] = 1
  return new_words

#preprocessing text
def preprocessing(sent):
  import re

  sent = re.sub('\\xa0|\\n|\n|\t', ' ', sent)
  sent = re.sub("'", "’", sent)
  sent = re.sub('@(\w+)?', '', sent)  # 계정 이름 제거

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
  #sent = re.sub('\W', ' ', sent)
  sent = re.sub('[^a-zA-Z0-0]', ' ', sent)
  sent = re.sub('united states', 'united_states', sent)
  sent = re.sub('united kingdom', 'united_kingdom', sent)
  sent = re.sub('piner rd', 'piner_rd', sent)
  sent = re.sub('chandanee magu', 'chandanee_magu', sent)
  sent = re.sub('anu aggarwal', 'anu_aggarwal', sent)
  sent = re.sub('gbonyin lga', 'gbonyin_lga', sent)
  sent = re.sub('\s\s+', ' ', sent)
  sent = sent.strip()

  return sent.lower()

def refine_text(refine_column):
  refine_text = []
  for i, text in enumerate(tqdm(refine_column, desc='preprocessing')):
      refine_text.append(preprocessing(text))
  return refine_text


"""
#####################################
# Dataset: Tweet Sentiment Extraction (train.tsv)
column_names = ['category', 'answer', 'addtional', 'sentence']
train_df = load_data('train.tsv', column_names)
train_df.reset_index(drop=True, inplace=True)

train_df['refine_text'] = refine_text(train_df['sentence'])
train_new_word_list = get_new_words_list(train_df['refine_text'])
print(train_new_word_list)
"""

"""
# Dataset: BBC News Classification (BBC News Train.csv)
column_names = ['ArticleId', 'Text', 'Category']
train_df = load_data('BBC News Train.csv')

train_df['refine_text'] = refine_text(train_df['Text'])
train_new_word_list = get_new_words_list(train_df['refine_text'])
print(len(train_new_word_list))
df = pd.DataFrame(train_new_word_list.keys(), columns=['word'])
df.to_csv('BBC_new_word.csv', index=False)
#print(train_new_word_list)
"""

# Dataset: Real or Not? NLP with Disaster Tweets (train.csv)
train_df = load_data('train.csv')
train_df.reset_index(drop=True, inplace=True)

train_df['refine_text'] = refine_text(train_df['text'])
train_df.to_csv('train_tweet_disaster.csv', index=False)
#train_new_word_list = get_new_words_list(train_df['refine_text'])
#print(len(train_new_word_list))
#df = pd.DataFrame(train_new_word_list.keys(), columns=['word'])
#df.to_csv('Disaster_new_word.csv', index=False)