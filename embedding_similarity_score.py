import pandas as pd
import numpy as np
from tqdm import tqdm
import requests

from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
  a = np.fromstring(a, dtype=float, sep=' ')
  b = np.fromstring(b, dtype=float, sep=' ')

  cos_sim = dot(a, b)/(norm(a)*norm(b))
  return cos_sim


if __name__ == '__main__':
    DATA_PATH = "/home/ida/workspace/jihye/dataset/"
    embedding_list = pd.read_csv(DATA_PATH+'wordEmbedding_en.csv', usecols=['word', 'vec300'], sep=' ', encoding='utf-8')
    embedding_dict = embedding_list.set_index('word')['vec300'].to_dict()

    new_embedding_list = pd.read_csv('prediction_embedding_10_new_words.csv')

    new_word_lower = []
    for w in new_embedding_list['new_word']:
        new_word_lower.append(w.lower())
    new_embedding_list['new_word_lower'] = new_word_lower

    # DataFrame to dictionary
    new_embedding_dict = new_embedding_list.set_index('new_word_lower')['embedding'].to_dict()

    #####################################
    # New embedding Similarity Score
    """
    for new_word in new_embedding_dict.keys():
        print(new_word)
        similarity_df = pd.DataFrame(columns=['word', 'similarity'])
        for word in tqdm(embedding_dict.keys(), desc='similarity', mininterval=1):
            cos_sim = cosine_similarity(new_embedding_dict[new_word], embedding_dict[word])
            similarity_df = similarity_df.append({'word': word, 'similarity': cos_sim}, ignore_index=True)
        similarity_df.to_csv('similarity_score_' + new_word + '.csv')
    """
    #####################################
    new_word = 'ppe'
    print(new_word)
    similarity_df = pd.DataFrame(columns=['word', 'similarity'])
    for word in tqdm(embedding_dict.keys(), desc='similarity', mininterval=1):
        cos_sim = cosine_similarity(embedding_dict[new_word], embedding_dict[word])
        similarity_df = similarity_df.append({'word': word, 'similarity': cos_sim}, ignore_index=True)
    similarity_df.to_csv('similarity_score_' + new_word + '_conceptnet.csv')
    """
    #####################################
    # top-N list of New embedding
    topN = 100
    rank = list(range(1, topN + 1))
    rank_df = pd.DataFrame(rank, columns=['rank'])
    for new_word in new_embedding_dict.keys():
        similarity_df = pd.read_csv('similarity_score_' + new_word + '.csv')
        sorted_df = similarity_df.sort_values(by='similarity', ascending=False).head(100)
        rank_df[new_word + '_word'] = list(sorted_df['word'])
        rank_df[new_word + '_score'] = list(sorted_df['similarity'])

    rank_df.to_csv('top-' + str(topN) + '_similarity.csv')

    
    #####################################
    # ConceptNet Similarity Score
    def get_embedding(word):
        url = 'http://192.9.24.248:19002/query/service'
        statement = 'use conceptnet;'
        statement += 'select vec300 from numberbatch_en where word="' + word + '";'
        req = requests.post(url, data={'statement': statement})
        result = req.json()['results']

        return result[0]['vec300']


    Numberbatch = list(pd.read_csv('new_word_list.csv')['Numberbatch'])
    new_word_list = list(pd.read_csv('new_word_list.csv')['word'])

    for new_word, existing in zip(new_word_list, Numberbatch):
        print(new_word, existing)
        if existing:
            similarity_df = pd.DataFrame(columns=['word', 'similarity'])
            word_embedding = get_embedding(new_word.lower())
            for word in tqdm(embedding_dict.keys(), desc='similarity', mininterval=1):
                cos_sim = cosine_similarity(word_embedding, embedding_dict[word])
                similarity_df = similarity_df.append({'word': word, 'similarity': cos_sim}, ignore_index=True)
            similarity_df.to_csv('conceptnet_similarity_score_' + new_word + '.csv')

    #####################################
    # top-N list of ConceptNet
    topN = 100
    rank = list(range(1, topN + 2))
    rank_df = pd.DataFrame(rank, columns=['rank'])
    for new_word, existing in zip(new_embedding_dict.keys(), Numberbatch):
        if existing:
            similarity_df = pd.read_csv('conceptnet_similarity_score_' + new_word + '.csv')
            sorted_df = similarity_df.sort_values(by='similarity', ascending=False).head(101)
            rank_df[new_word + '_word'] = list(sorted_df['word'])
            rank_df[new_word + '_score'] = list(sorted_df['similarity'])

    rank_df.drop(0, inplace=True)
    del rank_df['rank']
    rank_df['rank'] = list(range(1, topN + 1))

    rank_df.to_csv('conceptnet_top-' + str(topN) + '_similarity.csv')
    """

