import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import os
import pandas as pd
import numpy as np
# import cnn_2d_8layer as Model
# import cnn_1d_8layer as Model
import cnn_1d_10layer as Model


DATA_KEY = "english_23870_conceptnet"
DATASET = {"english_23870_first": 193,
           "english_23870_all_merged": 2161,
           "english_23870_all": 193,
           "english_23870_conceptnet": 160,}

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
        try:
            vector = vector.values.tolist()  # convert dataframe to list type

            # divide data into X and Y
            y = torch.tensor(np.array(vector[-1])).float()  # [300]
            # y = y[:30]  # [30]
            # y = y[:10]  # [10]
            del vector[-1]  # delete y from data

            zero_x_padding = torch.zeros(MAX_DIM - len(vector), 300, dtype=torch.float)

            X = torch.from_numpy(np.array(vector[:])).float()
            X = torch.cat([X, zero_x_padding], dim=0)  # [193*300]
            X.unsqueeze_(0)  # [1*193*300] for cnn_2d_8layer
            #X = X.transpose(0, 1)  # [300*193] for cnn_1d_8layer

            return X, y
        except IndexError:
            print("\n\n[IndexError] vector size: {}\n\n".format(len(vector)))

    def __len__(self):
        return len(self.lists)


def get_train_val_list(path, train_prob=0.8):
    lists = list(sorted(os.listdir(path)))
    train_size = int(len(lists)*train_prob)

    lists = [os.path.join(path, a_list) for a_list in lists]
    return lists[:train_size], lists[train_size:]


BATCH_SIZE = 32
in_dim, out_dim, num_filters = 1, 300, 2  # for cnn_2d_8layer
# in_dim, out_dim, num_filters = 300, 300, 300  # for cnn_1d_8layer
# in_dim, out_dim, num_filters = 300, 30, 30  # for cnn_1d_10layer
# in_dim, out_dim, num_filters = 300, 10, 30  # for cnn_1d_11layer
kernel, stride, padding = 2, 1, 1
lr = 3e-3  # learning rate 0.001

model = Model.CNN(in_dim, out_dim, num_filters, kernel, stride, padding)

# set loss and optimizer function for back-propagation
optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
# optimizer = optim.SGD(model.parameters(), lr=lr)  # Stochastic Gradient Descent optimizer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)  # learning rate scheduler (learning rate decay)

criterion = nn.L1Loss()
# criterion = nn.MSELoss()  # L2 loss (Euclidean distance) reduction='sum'
'''
Negative log likelihood loss with Poisson distribution of target
: 두 확률분포 사이의 차이를 재는 함수인 CrossEntropy가 되며, 
  CrossEntropy의 경우 비교 대상의 확률분포의 종류를 특정하지 않기 때문에 손실함수로 사용하기에 좋음
'''
# criterion = nn.PoissonNLLLoss()
'''
The Kullback-Leibler divergence loss measure:
1) useful distance measure for continuous distributions
2) often useful when performing direct regression over the space of continuous output distributions
'''
# criterion = nn.KLDivLoss(reduction='batchmean')  # ['none', 'batchmean', 'sum', 'mean'(default)]
cos_sim = nn.CosineSimilarity(dim=1)  # eps: small value to avoid division by zero (default=1e-8)

# load data and divide it into train and validation data
train_data, val_data = get_train_val_list(ROOT_PATH)
trn_dataset = WordEmbeddingDataLoader(train_data)
val_dataset = WordEmbeddingDataLoader(val_data)

# divide dataset into batch size
trn_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_epochs = 500
num_batches = len(trn_loader)
best_acury = 0
fraction = 10

for epoch in range(num_epochs):
    trn_loss, trn_acury = 0.0, 0.0
    model.train()
    for i, data in enumerate(trn_loader):
        x, y_true = data

        if use_cuda:
            x = x.cuda()
            y_true = y_true.cuda()

        y_pred = model(x)  # forward propagation
        y_pred = y_pred.squeeze()  # for cnn_1d_*
        """
        if i==0:
            print('y_pred: ', y_pred)
            print('y_true: ', y_true)
        """

        optimizer.zero_grad()  # zero the gradient buffers (initialize gradients)

        # back-propagation
        loss = criterion(y_pred, y_true)
        loss.backward()  # compute gradient of the loss with respect to model parameters
        optimizer.step()  # the optimizer updates model's parameters by loss
        trn_loss += loss.item()

        # accuracy
        acury = cos_sim(y_pred, y_true)
        trn_acury += torch.mean(acury)

    model.eval()
    # after training one epoch, validation
    with torch.no_grad():
        val_loss, val_acury = 0.0, 0.0
        for j, data in enumerate(val_loader):
            x, y_true = data

            if use_cuda:
                x = x.cuda()
                y_true = y_true.cuda()

            y_pred = model(x)
            y_pred = y_pred.squeeze()
            """
            if j == 0:
                print('y_pred: ', y_pred[0])
                print('y_true: ', y_true[0])
            """

            loss = criterion(y_pred, y_true)
            val_loss += loss.item()

            acury = cos_sim(y_pred, y_true)
            val_acury += torch.mean(acury)

        print("epoch: {:3d} | trn loss: {:.6f} | val loss: {:.6f} | "
              "trn accuracy: {:.4f} | val accuracy: {:.4f} | lr: {:.2e}".format
              (epoch+1, trn_loss/len(trn_loader), val_loss/len(val_loader),
               trn_acury/len(trn_loader), val_acury/len(val_loader), optimizer.param_groups[0]['lr']
        ))

        """
        # save the best trained model
        if (val_acury/len(val_loader)) > best_acury:
            SAVE_PATH = os.path.join(BASIC_PATH, DATA_KEY, "models")
            os.makedirs(SAVE_PATH, exist_ok=True)  # if the folder exist, do nothing. Otherwise, make folder(s)
            torch.save(model.state_dict(), SAVE_PATH+"cnn_1d_8layer_16batch_500epoch.pth")
            best_acury = (val_acury/len(val_loader))
        """
        scheduler.step(val_loss/len(val_loader))
