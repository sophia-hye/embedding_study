import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import os

import config as cfg
MAX_DIM = cfg.MAX_DIM


def get_train_val_list(path, train_prob=0.8):
    lists = list(os.listdir(path))
    train_size = int(len(lists) * train_prob)

    lists = [os.path.join(path, a_list) for a_list in lists]
    return lists[:train_size], lists[train_size:]


class WordEmbeddingDataLoader(DataLoader):
    def __init__(self, lists, cnn_1d=True, ensemble=True, train=True, start=0, end=cfg.TOTAL_EMB):
        self.lists = lists
        self.ensemble = ensemble
        self.train = train  # if eval mode, don't need to handle y vector
        self.y_start = start
        self.y_end = end
        self.cnn_1d = cnn_1d

    def __getitem__(self, item):
        vector = pd.read_csv(self.lists[item],
                             delimiter=',',
                             encoding='utf-8',
                             index_col=None)
        try:
            vector = vector.values.tolist()  # convert dataframe to list type

            if self.train:
                # divide data into X (trainable features) and Y (true answer)
                y = torch.tensor(np.array(vector[-1])).float()  # dim=[300]
                if self.ensemble:
                    y = y[self.y_start:self.y_end]
                del vector[-1]

            zero_x_padding = torch.zeros(MAX_DIM - len(vector), 300, dtype=torch.float)

            X = torch.from_numpy(np.array(vector[:])).float()
            X = torch.cat([X, zero_x_padding], dim=0)  # dim=[193*300]
            if self.cnn_1d:
                X = X.transpose(0, 1)  # [300*193] for cnn_1d_8layer
            else:  # cnn_2d
                X.unsqueeze_(0)  # [1*193*300] for cnn_2d_8layer

            if self.train:
                return X, y
            else:
                return X
        except IndexError:
            print("\n\n")
            print(self.lists[item])
            print("[IndexError] vector size: {}\n\n".format(len(vector)))

    def __len__(self):
        return len(self.lists)

    def set_y_index(self, start, end):
        self.y_start = start
        self.y_end = end

    def set_ensemble(self, ensemble):
        self.ensemble = ensemble

    def get_y_index(self):
        return self.y_start, self.y_end

    def get_ensemble(self):
        return self.ensemble

