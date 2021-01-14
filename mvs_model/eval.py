"""main func for training"""
import os
import re
import time
import random

import ujson
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

import config as cfg
import dataloader
import utils
#import cnn_2d_fclayer as m
#import cnn_2d_8layer as m
#import cnn_1d_8layer as m
import model as m
cnn_1d = True


if __name__ == '__main__':

	root = cfg.BASIC_PATH  # "/home/ida/workspace/jihye/dataset"
	max_seq_len = cfg.MAX_SEQ_LEN  # 100
	model_path = cfg.LOAD_PATH  # "/home/ida/workspace/jihye/model_pipeline/mvs_w2v/output_model"

	models = []
	for i in range(cfg.NUM_MODEL):
		_model = m.CNN(cfg.IN_DIM, cfg.OUT_DIM, cfg.NUM_FILTER, cfg.KERNEL, cfg.STRIDE, cfg.PADDING)
		if cfg.USE_CUDA:
			_model = _model.cuda()
		_model.load_state_dict(torch.load(f'{model_path}/model_{i}.pt'))
		_model.eval()
		models.append(_model)

	"""
	filename = os.path.join(root, 'new_word_list.csv')
	new_word_list = pd.read_csv(filename)  # column: word, definition, definition reference

	
	for i, (w, d) in enumerate(zip(new_word_list['word'], new_word_list['definition'])):
		try:
			print(i, w)
			tokens = d.lower().split()

			vectors = []
			for token in tokens:
				vector = utils.get_embedding_vector(token)
				if vector is not []:
					vectors.append(vector)

			# make a DataFrame about vectors
			df_vec = pd.DataFrame(np.array(vectors))

			# save vectors to csv file using DataFrame
			filepath = os.path.join(root, 'google_news_data')
			if not os.path.exists(filepath):
				os.makedirs(filepath)
			file_name = os.path.join(filepath, 'vectors'+str(i)+'.csv')

			df_vec.to_csv(file_name, sep=',', index=False, header=False)
		except Exception as e:
			print("ERROR: ")
			print(e)
	"""

	# get data path
	test_data_list, _ = dataloader.get_train_val_list(path=root+'/input_vector_keyword_glove', train_prob=1)
	# get input vectors
	#test_data = dataloader.WordEmbeddingDataLoader(test_data_list, ensemble=False, train=False)
	# divide data into batch size
	#test_data_loader = DataLoader(test_data, batch_size=10, shuffle=False)

	prediction_embedding = {}
	# [START PREDICTION]
	with torch.no_grad():		 
		for idx, test_data in enumerate(test_data_list):
			# e.g., test_data: /home/ida/workspace/jihye/dataset/input_vector_keyword/military_vector.csv
			# e.g., new_word: military
			new_word = test_data.split('/')[-1]
			new_word = re.sub('_vector.csv', '', new_word)
			# print('name: ', new_word)
			# print('file: ', test_data)

			vector = pd.read_csv(test_data,
								 encoding='utf-8',
								 index_col=None)
			vector = vector.values.tolist()  # convert dataframe to list type
			zero_padding = torch.zeros(cfg.MAX_DIM - len(vector), 300, dtype=torch.float)

			X = torch.from_numpy(np.array(vector[:])).float()
			X = torch.cat([X, zero_padding], dim=0)  # dim=[193*300]
			if cnn_1d:
				X = X.transpose(0, 1)  # [300*193] embedding dims first
			else:  # 2d cnn
				X.unsqueeze_(0)  # [1*193*300] adding 1 channel

			# make a batch including 1 item
			X = torch.unsqueeze(X, dim=0)  # [1*300*193] or [1*1*193*300]
			if cfg.USE_CUDA:
				X = X.cuda()
			
			outputs = []
			for _model in models:
				y_pred = _model(X)  # [1*10]
				y_pred.squeeze_(dim=0)  # [10]
				outputs.extend(y_pred)  # [1,2, ..., 10, 11, ..., 300]
			
			vector_str = ''
			for output in outputs:
				vector_str += str(output.item()) + ' '
			vector_str = vector_str.strip()

			prediction_embedding[new_word] = vector_str

		df = pd.DataFrame(list(prediction_embedding.items()), columns=['new_word', 'embedding'])
		df.to_csv('prediction_embedding_keyword_glove.csv')
		# df.to_excel('prediction_embedding.xlsx')
	# [END PREDICTION]

