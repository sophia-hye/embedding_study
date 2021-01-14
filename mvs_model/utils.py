"""utils for preproc & training"""
import re
import datetime

import torch
import numpy as np
import pandas as pd
import requests
from time import sleep

import torch.nn.functional as F

from sklearn.utils import resample
from torch.nn.modules.loss import _WeightedLoss
    
import config as cfg


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        
class EarlyStopping:
    def __init__(self, patience=20, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('    Early Stopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('    Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def argmax(logits):
    pred_flat = np.argmax(logits, axis=1).flatten()
    return pred_flat


def split_valid_points(num_batch, valid_split):
    """get points for multiple validations in an epoch (ex. [200, 400, 600, 800] with num_batch=800, valid_split=4)"""
    if valid_split == 0:
        return [num_batch-1]
    unit = num_batch//valid_split
    points = [unit-1]
    for i in range(valid_split-1):
        points.append(points[-1]+unit)
    points[-1] = num_batch-1
    return points


def preprocessing(sent):
    sent = re.sub('\\xa0|\\n|\n|\t', ' ', sent)
    sent = re.sub("'", "’", sent)

    short_form = pd.read_csv(os.path.join(root, 'English_short_form.csv'))
    for short_, long_ in zip(short_form['short form'], short_form['long form']):
        sent = re.sub(short_.lower(), long_, sent.lower())

    sent = re.sub('\s\s+', ' ', sent)

    converting_list = pd.read_csv(os.path.join(root, 'English_converting_list.csv'))
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


def get_embedding_vector(word):
    """get 300 dims embedding vector from AsterixDB in which Conceptnet Numberbatch is stored"""
    url = 'http://210.102.180.93:19002/query/service'
    statement = 'use conceptnet;'
    statement += 'select vec300 from numberbatch_en where word="'+word+'";'

    # since server state is unstable, response from the server sometimes is nothing.
    # for avoiding this problem, try same requests upto 10 times
    for i in range(0, 10):
        try:
            req = requests.post(url, data={'statement': statement})
            sleep(2)
            if req.status_code == 200:
                result = req.json()['results']
                if len(result) >= 1:
                    return result[0]['vec300'].split()
        except Exception as e:
            print("ERROR: ", word)
            print(e)
    # consider no result with no matched data during 10 times
    # if no matched data, return empty list
    return []