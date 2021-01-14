
import re
import os
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.model_selection import train_test_split

import model as Embedding
NUM_EMB_MODEL = 30
B = 32


def elapsed_time(start, end):
  from datetime import datetime

  fmt = '%Y-%m-%d %H:%M:%S'
  time_start = datetime.strptime(start, fmt)
  time_end = datetime.strptime(end, fmt)

  diff = time_end - time_start

  h = int(diff.seconds/3600)
  m = int((diff.seconds%3600)/60)
  s = (diff.seconds%3600)%60

  print("elapsed time: {:1d}days {:2d}:{:2d}:{:2d}".format(diff.days, h, m, s))

def get_timestamp():
  from datetime import datetime

  fmt = '%Y-%m-%d %H:%M:%S'
  return datetime.now().strftime(fmt)

def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def init_all_random(SET_SEED, r_seed=0, np_seed=0, torch_seed=0, cuda_seed=0):
  import random
  if SET_SEED:
      random.seed(r_seed)
      np.random.seed(np_seed)
      torch.manual_seed(torch_seed)
      torch.cuda.manual_seed_all(cuda_seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  else:
      print("random\n", random.getstate()[1][0])
      print("numpy.random\n", np.random.get_state()[0], np.random.get_state()[1][0])
      print("torch\n", torch.random.initial_seed())
      print("torch.cuda\n", torch.cuda.random.initial_seed())

def run_model(model, loss_fn, optimizer, train_loader, test_loader, epochs=500, device='cpu'):
    best_trn_accuracy, best_eval_accuracy = 0, 0

    for epoch in range(epochs):

        model.train()
        for X_text, y in train_loader:
            y = y.float()
            y = y.to(device=device)

            optimizer.zero_grad()

            ##############################################################
            with torch.autograd.detect_anomaly():
                probs = model(X_text)  # probs <= softmax result
                y_pred, _ = torch.max(probs, dim=1)  # y_pred : [batch]

                # loss
                trn_loss = loss_fn(y_pred, y)
                trn_loss.backward()
                optimizer.step()

            # accuracy
            pred = probs.cpu().detach().numpy()  # probs <= softmax result
            gt = y.view(-1, 1).cpu().detach().numpy()  # ground truth
            trn_accuracy = flat_accuracy(pred, gt)


            model.eval()
            with torch.no_grad():
                eval_loss, eval_accuracy = [], []
                for eval_X_text, eval_y in test_loader:
                    eval_y = eval_y.to(device=device)

                    probs = model(eval_X_text)  # probs <= softmax result
                    y_pred, _ = torch.max(probs, dim=1)  # y_pred : [batch]

                    # loss
                    loss = loss_fn(y_pred, eval_y)
                    eval_loss.append(loss.item())

                    # accuracy
                    pred = probs.cpu().detach().numpy()  # probs <= softmax result
                    gt = eval_y.view(-1, 1).cpu().detach().numpy()  # ground truth
                    val_accuracy = flat_accuracy(pred, gt)
                    eval_accuracy.append(val_accuracy)

                avg_eval_loss = np.mean(eval_loss, axis=0)
                avg_eval_accuracy = np.mean(eval_accuracy, axis=0)

                if best_eval_accuracy < avg_eval_accuracy:
                    best_eval_accuracy = avg_eval_accuracy

        if best_trn_accuracy < trn_accuracy:
          best_trn_accuracy = trn_accuracy
        if (epoch+1) % 50 == 0:
          print("{:3d}-epoch | trn loss: {:2.2f} | trn best accuracy: {:.4f} | val loss: {:2.2f} | val best accuracy: {:.4f}".format(
              epoch+1, trn_loss, best_trn_accuracy, avg_eval_loss, best_eval_accuracy))

    print("BEST TRN ACCURACY: {:.4f}".format(best_trn_accuracy))
    print("BEST VAL ACCURACY: {:.4f}".format(best_eval_accuracy))

class EMB_Classification(nn.Module):
    def __init__(self, emb_in=300, emb_out=10, emb_nf=30, emb_kernel=2, emb_stride=1, emb_padding=1,
                 clf_in=600, clf_H=100, clf_out=2, USE_CUDA=False, DEVICE='cpu'):
        super(EMB_Classification, self).__init__()

        self.emb_in, self.emb_out, self.emb_nf = emb_in, emb_out, emb_nf
        self.emb_kernel, self.emb_stride, self.emb_padding = emb_kernel, emb_stride, emb_padding
        self.clf_in, self.clf_H, self.clf_out = clf_in, clf_H, clf_out

        self.USE_CUDA = USE_CUDA
        self.DEVICE = DEVICE
        self.MAX_DIM = 193

        # Make embedding dictionary
        conceptnet_df = pd.read_csv('wordEmbedding_en.csv', encoding='utf-8', usecols=['word', 'vec300'], sep=' ')
        self.conceptnet = conceptnet_df.set_index('word')['vec300'].to_dict()

        # Make pair (keyword, definition) dictionary
        keyword_df = pd.read_csv('keyword_definition.csv', encoding='utf-8', usecols=['keyword', 'definition'])
        self.keyword = keyword_df.set_index('keyword')['definition'].to_dict()

        # load trained new-embedding models
        self.emb_models = []
        model_path = os.path.join("/home/ida/workspace/jihye/model_pipeline/embedding_model_cnn", "output_mse")
        for i in range(NUM_EMB_MODEL):
            m = Embedding.CNN(self.emb_in, self.emb_out, self.emb_nf,
                              self.emb_kernel, self.emb_stride, self.emb_padding)
            m = m.to(device=self.DEVICE)
            m.load_state_dict(torch.load(f'{model_path}/model_{i}.pt'))
            self.emb_models.append(m)

        # define fc layer for classification
        self.clf_model = nn.Sequential(
            nn.Linear(self.clf_in, self.clf_H),
            nn.ReLU(),
            nn.Linear(self.clf_H, self.clf_out),
        )
        self.clf_model = self.clf_model.to(device=self.DEVICE)

    def get_embedding(self, X_text):
        X = torch.tensor(0, device=self.DEVICE)
        for i, text in enumerate(X_text):
            tokens = text.split()
            vectors = torch.tensor(0, device=self.DEVICE)
            for t_idx, token in enumerate(tokens):
                if self.keyword.get(token):
                    d_tokens = self.keyword[token].split()  # definition tokens
                    d_vectors = []  # definition vectors
                    for dt in d_tokens:
                        d_vector = self.conceptnet.get(dt)
                        if d_vector is not None:
                            d_vector = np.fromstring(d_vector, dtype=float, sep=' ')
                            d_vectors.append(d_vector)
                    zero_padding = torch.zeros(self.MAX_DIM - len(d_vectors), 300, dtype=torch.float, device=self.DEVICE)
                    input = torch.from_numpy(np.array(d_vectors[:])).float()
                    input = input.to(device=self.DEVICE)
                    input = torch.cat([input, zero_padding], dim=0)  # [MAX_DIM, 300]
                    input = input.transpose(0, 1)  # [300, MAX_DIM]
                    input = input.unsqueeze(dim=0)  # [1 * 300 * MAX_DIM]

                    vector = torch.tensor(0, device=self.DEVICE)  # 300 dims embedding about keyword
                    for j, emb_model in enumerate(self.emb_models):
                        pred = emb_model(input)  # pred=[1, 10, 1]
                        pred = pred.squeeze(dim=0)  # pred=[10, 1]
                        if j == 0:
                            vector = pred  # [10, 1]
                        else:
                            # concatenate every 10 dims of prediction embedding about keyword
                            vector = torch.cat((vector, pred), dim=0)  # [300, 1]
                    vector = vector.squeeze(dim=1)  # [300]
                    vector = vector.unsqueeze(dim=0)  # [1, 300]
                    if torch.all(torch.eq(vectors, torch.tensor(0))):
                        # when 'vectors' is empty tensor
                        vectors = vector
                    else:
                        vectors = torch.cat((vectors, vector), dim=0)  # [n, 300]
                else:
                    vector = self.conceptnet.get(token)
                    if vector is not None:
                        vector = np.fromstring(vector, dtype=float, sep=' ')
                        vector = torch.tensor(vector, device=self.DEVICE)  # [300]
                        vector = vector.unsqueeze(dim=0)  # [1, 300]
                        if torch.all(torch.eq(vectors, torch.tensor(0))):
                            # when 'vectors' is empty tensor
                            vectors = vector
                        else:
                            vectors = torch.cat((vectors, vector), dim=0)  # [n, 300]
            if torch.all(torch.eq(vectors, torch.tensor(0))):
                # there's no match embedding about the text
                # eliminate the text data from X
                print("There's no embedding")
                print(i, text)
            else:
                #print('\n\nvectors type: ', vectors.type(), '\n\n')
                mean = torch.mean(vectors.float(), dim=0)  # [300]
                max, _ = torch.max(vectors.float(), dim=0)  # [300]
                concat = torch.cat((mean, max), dim=0)  # [600]
                concat = concat.unsqueeze(dim=0)  # 맨앞에 차원 추가 [1, 600]
                if self.USE_CUDA:
                    concat = concat.cuda()
                if i == 0:
                    X = concat
                else:
                    X = torch.cat((X, concat), dim=0)
        X = X.to(device=self.DEVICE)
        return X

    def forward(self, X_text):
        X = self.get_embedding(X_text)  # X is Double Tensor
        # clf_model expects Float Tensor
        logits = self.clf_model(X.float())  # logits:[batch, cls_out]
        probs = F.softmax(logits, dim=1)
        return probs

class tweetDataset(Dataset):
    '''Kaggle Tweets Dataset related to disaster'''
    def __init__(self, X_list, y_list):
        self.X_list = X_list
        self.y_list = y_list

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, item):
        x = self.X_list[item]
        y = self.y_list[item]
        return x, y


if __name__ == '__main__':
    data_df = pd.read_csv('train_refined.csv')
    X_train, X_test, y_train, y_test = train_test_split(data_df['keyword_text'].tolist(),
                                                        data_df['target'].tolist(),
                                                        train_size=0.8,
                                                        random_state=928,
                                                        shuffle=True,
                                                        stratify=data_df['target'].tolist())
    train_dataset = tweetDataset(X_train, y_train)
    test_dataset = tweetDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=B, shuffle=True, num_workers=2)

    # set random seeds
    init_all_random(SET_SEED=True,
                    r_seed=2147483648,
                    np_seed=2147483648,
                    torch_seed=17043145292842172883 & ((1 << 63) - 1),
                    cuda_seed=5039337088303087)

    #####################################
    USE_CUDA, DEVICE = False, 'cpu'
    if torch.cuda.is_available():
        USE_CUDA, DEVICE = True, 'cuda:0'
    #####################################

    model = EMB_Classification(USE_CUDA=USE_CUDA, DEVICE=DEVICE)

    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=3e-2)
    time_start = get_timestamp()
    best_accuracy = run_model(model, loss_fn, optimizer, train_loader, test_loader, device=DEVICE)
    time_end = get_timestamp()
    elapsed_time(start=time_start, end=time_end)

