# -*- coding: utf-8 -*-
"""
@Title      CNN based new embedding model
@Author     Jihye Kim
@Date       2020.05.16 ~
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
MAX = 193 # english_23870_first
#MAX = 2161 # english_23870_all_merged
#MAX = 193 # english_23870_all

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
	#test_size = int(len(lists)*(1-train_prob))

    lists = [os.path.join(path,l) for l in lists]
    return lists[:train_size], lists[train_size:]
    #return lists[:test_size], lists[:test_size]



# Read datasets from files
#ROOT_PATH = os.path.join("/mnt/sdb/workspace/jihye/dataset", "english_1009", "vectors")
#ROOT_PATH = os.path.join("/mnt/sdb/workspace/jihye/dataset", "english_23870_first", "vectors") # 11,931
#ROOT_PATH = os.path.join("/mnt/sdb/workspace/jihye/dataset", "english_23870_all_merged", "vectors") # 21,812
#ROOT_PATH = os.path.join("/mnt/sdb/workspace/jihye/dataset", "english_23870_all", "vectors") # 334,297

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
      nn.Conv1d(nf*256, nf*128, kernel, stride, padding, bias=False),
      nn.ReLU(True),
      nn.Conv1d(nf*128, nf*64, kernel, stride, padding, bias=False),
      nn.ReLU(True),
      nn.Conv1d(nf*64, nf*32, kernel, stride, padding, bias=False),
      nn.ReLU(True),
      nn.Conv1d(nf*32, nf*16, kernel, stride, padding, bias=False),
      nn.ReLU(True),
      nn.Conv1d(nf*16, nf*8, kernel, stride, padding, bias=False),
      nn.ReLU(True),
      nn.Conv1d(nf*8, nf*4, kernel, stride, padding, bias=False),
      nn.ReLU(True),
      nn.Conv1d(nf*4, nf*2, kernel, stride, padding, bias=False),
      nn.ReLU(True),
      nn.Conv1d(nf*2, nf, kernel, stride, padding, bias=False),
      nn.ReLU(True),
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

# Define a CNN model
in_dim, out_dim, num_filters = MAX, 1, 2
cnn = CNN(in_dim, out_dim, num_filters)

# Set loss & optimizer function for backpropagation
learning_rate = 0.001 
#optimizer = optim.SGD(cnn.parameters(), lr=learning_rate) # SGD
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate) # Adam
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer) # learning rate decay

criterion = nn.MSELoss() # L2 loss (Euclidean distance)
# Cosine Similarity on the basis of dim1
cos = nn.CosineSimilarity(dim=1) # eps: small value to avoid division by zero (default=1e-8)


##################################################################
##################################################################
### one data overfitting test ###
"""
X, y_true = torch.rand(1, 193, 300), torch.rand(1, 300)
for i in range(3):
    vector = pd.read_csv('/mnt/sdb/workspace/jihye/dataset/english_23870_first/vectors/vectors'+str(i)+'.csv', 
            delimiter=',', encoding='utf-8', index_col=None)
    vector = vector.values.tolist() # Convert DataFrame to list

    # Divide data into X and y
    y = torch.tensor(np.array(vector[-1])).float()
    y = torch.unsqueeze(y, 0) # [300] -> [1, 300]
    del vector[-1] # Delete y from data
    zero_x_padding = torch.zeros(MAX-len(vector), 300, dtype=torch.float)
    x = torch.from_numpy(np.array(vector[:])).float()
    x = torch.cat([x, zero_x_padding], dim=0)
    x = torch.unsqueeze(x, 0) # [193, 300] -> [1, 193, 300] expand dimension

    if i == 0:
        X = x.clone().detach()
        y_true = y.clone().detach()
        print('X Shape: ',X.shape)
        print('y Shape: ',y_true.shape)
    else:
        X = torch.cat([X, x], dim=0)
        print('X Shape: ', X.shape)
        y_true = torch.cat([y_true, y], dim=0)
        print('y Shape: ', y_true.shape)

#print('X Shape: ',X.shape)
#print('y Shape: ',y_true.shape)

for epoch in range(200): # Batchnorm1d (x): 300, (o): 2000
    optimizer.zero_grad()
    y_pred = cnn(X)
    
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()
    
    cos_output = cos(y_pred, y_true)
    acury = torch.mean(cos_output)
    
    del y_pred
    
    with torch.no_grad():
        val_y_pred = cnn(X)
        val_loss = criterion(val_y_pred, y_true)
        val_cos_output = cos(val_y_pred, y_true)
        val_acury = torch.mean(val_cos_output)
        
        if (epoch+1)%20 == 0: 
            print("train accuracy: {:.4f} | test accuracy: {:.4f}".format(acury, val_acury))
            print("epoch: {}/{} | train loss: {:.4f} | test loss: {:.4f}".format(epoch+1, 2000, loss.item(), val_loss.item()))
"""
##################################################################
##################################################################



# hyper-parameters
num_epochs = 500
num_batches = len(trn_loader)
every_n_batches = 100 # every nth batches, test validation dataset and print train and validation losses

# every one epoch, save avg. loss & accuracy value
trn_loss_list, val_loss_list = [], [] # train and validation loss
trn_acury_list, val_acury_list = [], [] # train and validation accuracy

# every n-th batches, save avg. loss & accuracy value
#trn_nth_loss_list, val_nth_loss_list = [], [] # train and validation loss
#trn_nth_acury_list, val_nth_acury_list = [], [] # train and validation accuracy

#from apex import amp
#import apex.amp as amp
# Initialization
#opt_level = 'O1'
#cnn, optimizer = amp.initialize(cnn, optimizer, opt_level=opt_level)

best_acury = 0
for epoch in range(num_epochs):
  trn_loss = 0.0 # every epoch
  trn_acury = 0.0 # every epoch
  #trn_nth_loss = 0.0 # every nth batches (ex. every 100th batches)
  #trn_nth_acury = 0.0 # every nth batches (ex. every 100th batches)
  for i, data in enumerate(trn_loader):
    x, y_true = data
    
    if use_cuda:
      x = x.cuda()
      y_true = y_true.cuda()
    
    optimizer.zero_grad() # zero the gradient buffers (initialize gradients)
    
    y_pred = cnn(x) # forward propagation
    
    # Backward part
    loss = criterion(y_pred, y_true)
    loss.backward() # backpropagation: compute gradient of the loss with respect to model parameters
    optimizer.step() # An optimizer makes an update to its parameters

    # Train your model (apex)
	#with amp.scale_loss(loss, optimizer) as scaled_loss:
	#  scaled_loss.backward() # backward + step
    
    # accuracy
    cos_output = cos(y_pred, y_true)
    trn_acury += torch.mean(cos_output)
    #trn_nth_acury += torch.mean(cos_output)

    # trn_loss summary
    #trn_nth_loss += loss.item()
    trn_loss += loss.item()
    del loss
    del y_pred

    # display train process
    """
    if (i+1) % every_n_batches == 0: # every 100th mini-batches
      with torch.no_grad(): # important!
        val_nth_loss = 0.0
        val_nth_acury = 0.0
        for j, val in enumerate(val_loader):
          val_x, val_y_true = val

          if use_cuda:
            val_x = val_x.cuda()
            val_y_true = val_y_true.cuda()
          
          val_y_pred = cnn(val_x)
          v_loss = criterion(val_y_pred, val_y_true)
          val_nth_loss += v_loss

          # accuracy
          val_cos_output = cos(val_y_pred, val_y_true)
          val_nth_acury += torch.mean(val_cos_output)
        
        print("train accuracy: {:.4f} | test accuracy: {:.4f}".format(
            trn_nth_acury / every_n_batches, val_nth_acury / len(val_loader)))
          
        print("epoch: {}/{} | step: {}/{} | trn loss: {:.6f} | val loss: {:.6f}".format(
            epoch+1, num_epochs, i+1, num_batches, trn_nth_loss / every_n_batches, val_nth_loss / len(val_loader)))
      # store loss value
      trn_nth_loss_list.append(trn_nth_loss/every_n_batches) # 100th, 200th, ...
      val_nth_loss_list.append(val_nth_loss/len(val_loader)) # total number of validation data

      # store accuracy value
      trn_nth_acury_list.append(trn_nth_acury / every_n_batches)
      val_nth_acury_list.append(val_nth_acury / len(val_loader))
      trn_nth_loss, trn_nth_acury = 0.0, 0.0
      """

    if (i+1) == len(trn_loader): # every one epoch (after finishing all step)
      with torch.no_grad():
        val_loss = 0.0
        val_acury = 0.0
        for j, val in enumerate(val_loader):
          val_x, val_y_true = val

          if use_cuda:
            val_x = val_x.cuda()
            val_y_true = val_y_true.cuda()

          val_y_pred = cnn(val_x)
          v_loss = criterion(val_y_pred, val_y_true)
          val_loss += v_loss

          # accuracy
          val_cos_output = cos(val_y_pred, val_y_true)
          val_acury += torch.mean(val_cos_output)

        print("epoch: {:3d}/{} | trn loss: {:.4f} | val loss: {:.4f} | train accuracy: {:.4f} | val accuracy: {:.4f} | lr: {:.2e}".format(
          epoch + 1, num_epochs, trn_loss / len(trn_loader), val_loss / len(val_loader), trn_acury / len(trn_loader), val_acury / len(val_loader), optimizer.param_groups[0]['lr']))

      if (val_acury/len(val_loader)) > best_acury:
        # save trained model
        PATH = './dataset/english_23870_first/models/cnn_8layer_32_500.pth'
		#PATH = './dataset/english_23870_all/models/cnn_10layer_32_500.pth'
        os.makedirs('./dataset/english_23870_first/models', exist_ok=True) # if the folder exist, do nothing. Otherwise, make folder(s)
        torch.save(cnn.state_dict(), PATH)
        best_acury = val_acury/(len(val_loader))
      	
      # store loss value
      trn_loss_list.append(trn_loss/len(trn_loader)) # total number of train data
      val_loss_list.append(val_loss/len(val_loader)) # total number of validation data

      # store accuracy value
      trn_acury_list.append(trn_acury/len(trn_loader))
      val_acury_list.append(val_acury/len(val_loader))
      # Note! step should be called after validate()
      scheduler.step(val_loss) # Reduce lr when a metric has stopped improving

print("\n\nfinish all train and validation")


# save loss and accuracy to excel file
"""
dict1 = {"train loss": trn_loss_list,
             "validation loss": val_loss_list,
             "train accuracy": trn_acury_list,
             "validation accuracy": val_acury_list}
df1 = pd.DataFrame(dict1)
df1.to_excel("save_trn_val_loss_acury.xlsx")
print("\n\nfinish save the file1\n")


dict2 = {"train loss": trn_nth_loss_list,
             "validation loss": val_nth_loss_list,
             "train accuracy": trn_nth_acury_list,
             "validation accuracy": val_nth_acury_list}
df2 = pd.DataFrame(dict2)
df2.to_excel("save_trn_val_nth_loss_acury.xlsx")
print("\n\nfinish save the file2\n")


# save trained model
PATH = './dataset/english_23870_first/models/cnn_8layer_32_500.pth'
torch.save(cnn.state_dict(), PATH)
"""
# Load saved model
#cnn = CNN()
#cnn.load_state_dict(torch.load(PATH))

