# -*- coding: utf-8 -*-
"""
@Title      CNN based new embedding model
@Author     Jihye Kim
@Date       2020.07.13 ~
@Version    1.0
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import os
import pandas as pd
import numpy as np


#MAX = 513 # english_1009
#MAX = 193 # english_23870_first
#MAX = 2161 # english_23870_all_merged
MAX = 193 # english_23870_all

class WordEmbeddingDataLoader(DataLoader):
  def __init__(self, lists):
    self.lists = lists
  
  def __getitem__(self, idx):
    try:
      vector = pd.read_csv(self.lists[idx], delimiter=',', encoding='utf-8', index_col=None)
      vector = vector.values.tolist() # Convert DataFrame to list

      # Divide data into X and y
      y = torch.tensor(np.array(vector[-1])).float()
      del vector[-1] # Delete y from data
      zero_x_padding = torch.zeros(MAX-len(vector), 300, dtype=torch.float)
      x = torch.from_numpy(np.array(vector[:])).float()
      x = torch.cat([x, zero_x_padding], dim=0)
    except IndexError:
      print('\n\nIndexError: vector size: ',len(vector),'\n\n')
    return (x,y)
    
  def __len__(self):
    return len(self.lists)
  
def get_list(path, train_prob=0.8):
    lists = list(sorted(os.listdir(path)))
    train_size = int(len(lists)*train_prob)
    test_size = int(len(lists)*0.2)

    lists = [os.path.join(path,l) for l in lists]
    #return lists[:train_size], lists[train_size:]
    return lists[:test_size], lists[:test_size]


url = 'http://192.9.24.248:19002/query/service'
def get_vec300(word):
  statement = 'use conceptnet;'
  statement += 'select vec300 from numberbatch_en where word="' + word + '";'

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
      a = 1

  # consider no result with no matched data during 10 times
  # if no matched data, return empty list
  return []


def get_vectors(data):
  # if data is x_tokens (words)
  if isinstance(data, list):
    vectors = []
    new_data = copy.deepcopy(data)
    for i, d in enumerate(data):
      temp = get_vec300(d)
      if temp == []:
        # delete that data(token) in copied data set(new_data)
        new_data.remove(d)
      else:
        vectors.append(np.array(temp))
    return (new_data, vectors)
  else:  # if data is y (word)
    temp = get_vec300(data)
    # if there's no result (dense vectors) from the DB
    if temp == []:
      return (data, None)  # there's no ground truth
    else:
      return (data, temp)

# Read datasets from files
ROOT_PATH = os.path.join("/home/ida/workspace/jihye/dataset", "english_23870_first", "vectors") # 11,931
#ROOT_PATH = os.path.join("/home/ida/workspace/jihye/dataset", "english_23870_all_merged", "vectors") # 21,812
#ROOT_PATH = os.path.join("/home/ida/workspace/jihye/dataset", "english_23870_all", "vectors") #334,297 -> 312,409



train_list, val_list = get_list(ROOT_PATH)
trn_dataset = WordEmbeddingDataLoader(train_list)
val_dataset = WordEmbeddingDataLoader(val_list)


# Divide dataset into batch_size
batch_size = 32
trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Construct model on cuda if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# CNN model
class CNN(nn.Module):
  # @in_dim: dimentions of inputted data
  # @out_dim: dimentions of output data which user want to get
  # @nf: number of filters
  def __init__(self, in_dim, out_dim, nf):
    # Always start inheriting torch.nn.Module
    super(CNN, self).__init__()
    
    kernel = 3
    stride = 1
    padding = 1
    self.net = nn.Sequential(
      nn.Conv1d(in_dim, nf*512, kernel, stride, padding, bias=False),
	  nn.ReLU(True),
      nn.Conv1d(nf*512, nf*256, kernel, stride, padding, bias=False),
      nn.ReLU(True),
      # [B, 513, 300] -> [B, 512, 300]
      nn.Conv1d(nf*256, nf*128, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*128),
      nn.ReLU(True),
      # [B, 512, 300] -> [B, 128, 300]
      nn.Conv1d(nf*128, nf*64, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*64),
      nn.ReLU(True),
      # [B, 128, 300] -> [B, 64, 300]
      nn.Conv1d(nf*64, nf*32, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*32),
      nn.ReLU(True),
      # [B, 64, 300] -> [B, 32, 300]
      nn.Conv1d(nf*32, nf*16, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*16),
      nn.ReLU(True),
      # [B, 32, 300] -> [B, 16, 300]
      nn.Conv1d(nf*16, nf*8, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*8),
      nn.ReLU(True),
      # [B, 16, 300] -> [B, 8, 300]
      nn.Conv1d(nf*8, nf*4, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*4),
      nn.ReLU(True),
      # [B, 8, 300] -> [B, 4, 300]
      nn.Conv1d(nf*4, nf*2, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*2),
      nn.ReLU(True),
      # [B, 4, 300] -> [B, 2, 300]
      nn.Conv1d(nf*2, nf, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf),
      nn.ReLU(True),
      # [B, 2, 300] -> [B, 1, 300]
      nn.Conv1d(nf, out_dim, kernel, stride, padding, bias=False),
      nn.Tanh() # range: -1 ~ 1
      
      # [B, 513, 300] -> [B, 128, 300]
	  #nn.Conv1d(in_dim, nf*64, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*64),
	  #nn.ReLU(True),
      # [B, 128, 300] -> [B, 32, 300]
	  #nn.Conv1d(nf*64, nf*16, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*16),
	  #nn.ReLU(True),
      # [B, 32, 300] -> [B, 8, 300]
	  #nn.Conv1d(nf*16, nf*4, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf*4),
	  #nn.ReLU(True),
      # [B, 8, 300] -> [B, 2, 300]
	  #nn.Conv1d(nf*4, nf, kernel, stride, padding, bias=False),
      #nn.BatchNorm1d(nf),
	  #nn.ReLU(True),
      # [B, 2, 300] -> [B, 1, 300]
	  #nn.Conv1d(nf, out_dim, kernel, stride, padding, bias=False),
	  #nn.Tanh()
    )

    if use_cuda:
      self.net = self.net.cuda()
  
  def forward (self, x):
    # add channel dimensions
    #x = torch.unsqueeze(x, 1) # [64, 513, 300] -> [64, 1, 513, 300]
    # gpu allocation
    # x: [batch size, height, width] -> [B, 513, 300]
	#if use_cuda:
	#  y_pred = nn.parallel.data_parallel(self.net, x, device_ids=[device])
	#else:
	#  y_pred = self.net(x)
    
    y_pred = self.net(x)
    y_pred = torch.squeeze(y_pred, 1) # [B, 300])
    return y_pred

# Define parameters of a CNN model
in_dim, out_dim, num_filters = MAX, 1, 2

# Set loss & optimizer function for backpropagation
learning_rate = 0.001
#optimizer = optim.SGD(cnn.parameters(), lr=learning_rate) # SGD
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate) # Adam
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer) # learning rate decay

criterion = nn.MSELoss() # L2 loss (Euclidean distance)
# Cosine Similarity on the basis of dim1
cos = nn.CosineSimilarity(dim=1) # eps: small value to avoid division by zero (default=1e-8)

# hyper-parameters
num_epochs = 500
num_batches = len(trn_loader)
every_n_batches = 100 # every nth batches, test validation dataset and print train and validation losses

# Load the saved model
cnn = CNN(in_dim, out_dim, num_filters) # Define a CNN model
#PATH = './dataset/english_23870_all_merged/models/cnn_10layer_32_500.pth'
PATH = './dataset/english_23870_first/models/cnn_10layer_32_500.pth'
cnn.load_state_dict(torch.load(PATH))
cnn.eval() #드롭아웃 및 배치 놈을 평가 모드로 설정

# Predict
definitions = ['personal protective equipment', #ppe
     'search for information about someone or something on the Internet using the search engine Google', #google
     'searching for information about someone or something on the Internet using the search engine Google', #googling
     'an element of a culture or system of behavior that may be considered to be passed from one individual to another by nongenetic means, especially imitation'] #meme
targets = ['ppe', 'google', 'googling', 'meme']

preprocessing = []
for d in definitions:
  preprocessing.append(d.lower())

for i, x, y in enumerate(zip(definitions,targets)):
  x_tokens = x.split()
  y_vec = get_vectors(y)
  x_tokens, x_vectors = get_vectors(x_tokens)

  y_true = torch.tensor(np.array(y_vec)).float()
  zero_x_padding = torch.zeros(MAX - len(x_vectors), 300, dtype=torch.float)
  x = torch.from_numpy(np.array(x_vectors)).float()
  x = torch.cat([x, zero_x_padding], dim=0)

  y_pred = cnn(x)



