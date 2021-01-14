import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import os
import pandas as pd
import numpy as np

import lstm


DATA_KEY = "english_23870_first"
DATASET = {"english_23870_first": 193,
           "english_23870_all_merged": 2161,
           "english_23870_all": 193}

BASIC_PATH = "/home/ida/workspace/jihye/dataset"
ROOT_PATH = os.path.join(BASIC_PATH, DATA_KEY, "vectors")
MAX_DIM = DATASET[DATA_KEY]

# construct model on cuda if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class WordEmbeddingDataLoader(DataLoader):
    def __init__(self, lists):
        self.lists = lists

    def __getitem__(self, idx):
        vector = pd.read_csv(self.lists[idx], delimiter=',', encoding='utf-8', index_col=None)
        vector = vector.values.tolist()  # convert dataframe to list type
        try:
            # divide data into X and Y
            y = torch.tensor(np.array(vector[-1])).float()  # [300]
            del vector[-1]  # delete y from data
            zero_x_padding = torch.zeros(MAX_DIM - len(vector), 300, dtype=torch.float)
            x = torch.from_numpy(np.array(vector[:])).float()  # [sequence_length*300]
            # MAX_DIM=193 (sequence length + zero padding)
            x = torch.cat([x, zero_x_padding], dim=0) # [193*300]
            return x, y

        except IndexError:
            print("\n\n[IndexError] vector size: {}\n\n".format(len(vector)))

    def __len__(self):
        return len(self.lists)


def get_train_val_list(path, train_prob=0.8):
    lists = list(sorted(os.listdir(path)))
    train_size = int(len(lists)*train_prob)

    lists = [os.path.join(path, l) for l in lists]
    return lists[:train_size], lists[train_size:]


BATCH_SIZE = 16

# load data and divide it into train and validation data
train_data, val_data = get_train_val_list(ROOT_PATH)
trn_dataset = WordEmbeddingDataLoader(train_data)
val_dataset = WordEmbeddingDataLoader(val_data)

# divide dataset into batch size
trn_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


in_dim, hidden_dim, num_layers = 300, 300, 8
drop_prob, is_batch_first, is_bidirectional = 0.2, True, False

model = lstm.LSTM(in_dim, hidden_dim, num_layers, drop_prob, is_batch_first, is_bidirectional)

# set loss and optimizer function for back-propagation
lr = 1e-3  # learning rate 0.001
# optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)  # Stochastic Gradient Descent optimizer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)  # learning rate scheduler (learning rate decay)

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()  # L2 loss (Euclidean distance) reduction='sum'
# criterion = nn.HingeEmbeddingLoss()
#criterion = nn.CosineEmbeddingLoss()  # error
cos_sim = nn.CosineSimilarity(dim=1)  # eps: small value to avoid division by zero (default=1e-8)


num_epochs = 5000
num_batches = len(trn_loader)
best_acury = 0

for epoch in range(num_epochs):
    trn_loss, trn_acury = 0.0, 0.0
    for i, data in enumerate(trn_loader):
        # x: [batch_size, sequence_length, embedding_size]
        # y: [batch_size, embedding_size]
        X, y_true = data

        if use_cuda:
            X = X.cuda()
            y_true = y_true.cuda()

        optimizer.zero_grad()  # zero the gradient buffers (initialize gradients)

        y_pred = model(X)  # forward propagation
        """
        if i == 0:
            print('y_pred: ', y_pred)
            print('y_true: ', y_true)
        """

        # back-propagation
        loss = criterion(y_pred, y_true)
        loss.backward()  # compute gradient of the loss with respect to model parameters
        optimizer.step()  # the optimizer updates model's parameters by loss
        trn_loss += loss.item()

        # accuracy
        acury = cos_sim(y_pred, y_true)
        trn_acury += torch.mean(acury)

    # after training one epoch, validation
    with torch.no_grad():
        val_loss, val_acury = 0.0, 0.0
        for j, data in enumerate(val_loader):
            X, y_true = data

            if use_cuda:
                X = X.cuda()
                y_true = y_true.cuda()

            y_pred = model(X)
            """
            if j == 0:
                print('y_pred: ', y_pred)
                print('y_true: ', y_true)
            """

            loss = criterion(y_pred, y_true)
            val_loss += loss.item()

            acury = cos_sim(y_pred, y_true)
            val_acury += torch.mean(acury)

        print("epoch: {:3d} | trn loss: {:.4f} | val loss: {:.4f} | "
              "trn accuracy: {:.4f} | val accuracy: {:.4f} | lr: {:.2e}".format
              (epoch+1, trn_loss/len(trn_loader), val_loss/len(val_loader),
               trn_acury/len(trn_loader), val_acury/len(val_loader), optimizer.param_groups[0]['lr']
        ))

        # save the best trained model
        """
        if (val_acury/len(val_loader)) > best_acury:
            SAVE_PATH = os.path.join(BASIC_PATH, DATA_KEY, "models")
            # if the folder exist, do nothing. Otherwise, make folder(s)
            os.makedirs(SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH+"cnn_1d_8layer_16batch_500epoch.pth")
            best_acury = (val_acury/len(val_loader))
        """
        scheduler.step(val_loss/len(val_loader))
