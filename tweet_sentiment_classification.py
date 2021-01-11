import torch
import pandas as pd
import numpy as np
import requests
import copy

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings(action='ignore')


#preprocessing text
def preprocessing(sent):
  import re

  sent = re.sub('\\xa0|\\n|\n|\t', ' ', sent)
  sent = re.sub("'", "’", sent)

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
  sent = re.sub('\W', ' ', sent)
  sent = re.sub('\s\s+', ' ', sent)
  sent = sent.strip()

  return sent.lower()

def load_data(filename):
    df = pd.read_csv(filename)
    df.dropna(axis=0, inplace=True)

    # category encoding
    from sklearn.preprocessing import LabelEncoder

    lb_encoder = LabelEncoder()
    df['label'] = lb_encoder.fit_transform(df['sentiment'])

    refine_text = []
    for i, text in enumerate(tqdm(df['text'], desc='preprocessing')):
        refine_text.append(preprocessing(text))
    df['refine_text'] = refine_text

    return df

##########################################################################
def get_embedding_vector(refine_text, KEY, save=False):
    if KEY == 'new-embedding':
        new_embedding_list = pd.read_csv('prediction_embedding.csv')

        new_word_lower = []
        for w in new_embedding_list['new_word']:
            new_word_lower.append(w.lower())
        new_embedding_list['new_word_lower'] = new_word_lower

        # DataFrame to dictionary
        new_embedding_dict = new_embedding_list.set_index('new_word_lower')['embedding'].to_dict()

    embedding_list = pd.read_csv('wordEmbedding_en.csv', encoding='utf-8', usecols=['word', 'vec300'], sep=' ')
    embedding_dict = embedding_list.set_index('word')['vec300'].to_dict()

    X_vector = []
    new_word = []
    num_empty = 0
    for i, sent in enumerate(tqdm(refine_text, desc='str2num', mininterval=60)):
        tokens = sent.split()
        vectors = []
        for token in tokens:
            if KEY == 'new-embedding' and token in new_word_lower:
                vector = new_embedding_dict.get(token)
            else:
                vector = embedding_dict.get(token)

            if vector is None:  # token is not in conceptnet
                new_word.append(token)
            else:
                if vector != []:
                    vector = np.fromstring(vector, dtype=float, sep=' ')  # when return: embedding
                    # vector = np.array(list(vector), dtype=float) # when return: embedding.split()
                    vectors.append(vector)
        if vectors == []:
            num_empty += 1
        else:
            X_vector.append(vectors)

    print("{:2d} na rows are dropped".format(num_empty))

    if save:
        print("be saving the vector into csv file")
        if KEY == 'new-embedding':
            df_vec = pd.DataFrame({'X_vector': X_vector})
            df_vec.to_csv('new-Embedding_vector.csv', index=False, header=False)
        else:  # KEY == 'conceptnet'
            df_vec = pd.DataFrame({'X_vector': X_vector})
            df_vec.to_csv('conceptnet_vector.csv', index=False, header=False)
        print("finish to save")

    return X_vector, new_word

##########################################################################
def get_embedding_vector_from_csv(filename):
    import re
    df_vec = pd.read_csv(filename, names=['X_vector'])

    X_vector = []
    try:
        for i, row in enumerate(df_vec['X_vector']):
            str_arr = re.sub('\n|\]\)', '', row)
            str_arr = re.sub('\s\s+', ' ', str_arr)
            str_arr = str_arr.split('array([')

            new_arr = []
            for j, vec in enumerate(str_arr):
                if j == 0:
                    continue
                else:
                    vec = vec.strip()
                    vec = np.fromstring(vec, dtype=float, sep=', ')
                    new_arr.append(vec)
            X_vector.append(new_arr)
    except Exception as e:
        print(i, j, vec)
        print(e)
    return X_vector

def calculate_mean_max(X_vector):
  X_mean, X_max = [], []
  for i, vec in enumerate(tqdm(X_vector, desc='mean&max', mininterval=10)):
    tensor_vec = torch.tensor(vec)

    mean = torch.mean(tensor_vec, dim=0)
    X_mean.append(mean)

    max = torch.max(tensor_vec, dim=0)
    X_max.append(max.values)
  return X_mean, X_max

def get_X_concat(X_mean, X_max):
  import copy
  X_concat = torch.tensor(0)

  for i, (mean, max) in enumerate(tqdm(zip(X_mean, X_max), desc='concat', mininterval=1)):
    concat = torch.cat((mean, max), dim=0) # mean=[300], max=[300]
    concat.unsqueeze_(dim=0)
    if i == 0:
      X_concat = copy.deepcopy(concat)
    else:
      X_concat = torch.cat((X_concat, concat), dim=0)
  return X_concat

##########################################################################
def get_bert_vector(refine_text):
    from transformers import BertModel, BertTokenizer
    import copy

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    X = torch.tensor(0)
    for i, text in enumerate(tqdm(refine_text, desc='BERT', mininterval=10)):
        # print(i, text)
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        with torch.no_grad():
            outputs = model(input_ids)
        last_hidden_states = outputs[0]  # [batch, tokens, hidden_dims (768)]

        apool = torch.mean(last_hidden_states, dim=1)  # tokens
        mpool, _ = torch.max(last_hidden_states, dim=1)  # tokens
        concat = torch.cat((apool, mpool), dim=1)  # 768*2=15
        if i == 0:
            X = copy.deepcopy(concat)
        else:
            X = torch.cat((X, concat), dim=0)  # batch
    return X

def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_data_loader(X, y):
    from torch.utils.data import TensorDataset, DataLoader

    y = torch.tensor(y, dtype=float)
    data_loader = DataLoader(TensorDataset(X, y),
                              batch_size=B,
                              shuffle=True,
                              num_workers=2)
    return data_loader

def get_model(D_in, H, D_out, activation_fn='ReLU'):
    if activation_fn == 'ReLU6':
        model = nn.Sequential(nn.Linear(D_in, H),
                              nn.ReLU6(),  # ReLU ReLU6 RReLU
                              nn.Linear(H, D_out), )
    elif activation_fn == 'RReLU':
        model = nn.Sequential(nn.Linear(D_in, H),
                              nn.RReLU(),  # ReLU ReLU6 RReLU
                              nn.Linear(H, D_out), )
    else:
        model = nn.Sequential(nn.Linear(D_in, H),
                              nn.ReLU(),  # ReLU ReLU6 RReLU
                              nn.Linear(H, D_out), )

    if USE_CUDA:
        model = model.cuda()
    return model

def run_model(model, lr, loss_fn, optimizer, train_loader, test_loader):
  activation_fn = list(model.children())[1]

  print("\n#####################################")
  print(f"EMBEDDING MODEL: {model_name[KEY]}")
  print(f"learning rate: {lr}\nloss function: {loss_fn.children}")
  print(f"Activation function: {activation_fn}\noptimizer: {optimizer}\n")
  best_accuracy = 0
  for epoch in range(500):
    for X, y in train_loader:
      X = X.float()
      y = y.float()

      if USE_CUDA:
        X = X.cuda()
        y = y.cuda()

      model.train()
      model.zero_grad()

      logits = model(X)  # [batch, D_out] = [32, 4]
      logits_ = F.softmax(logits, dim=1)  # logits_ : [batch, D_out]
      y_pred, _ = torch.max(logits_, dim=1)  # y_pred : [batch]

      loss = loss_fn(y_pred, y)

      pred = logits_.cpu().detach().numpy()
      gt = y.view(-1, 1).cpu().detach().numpy()
      accuracy = flat_accuracy(pred, gt)

      loss.backward()
      optimizer.step()

      with torch.no_grad():
        model.eval()

        eval_loss, eval_accuracy = [], []
        for eval_X, eval_y in test_loader:
          eval_X = eval_X.float()
          eval_y = eval_y.float()

          if USE_CUDA:
            eval_X = eval_X.cuda()
            eval_y = eval_y.cuda()

          logits = model(eval_X)
          logits_ = F.softmax(logits, dim=1)
          y_pred, _ = torch.max(logits_, dim=1)

          loss = loss_fn(y_pred, eval_y)

          pred = logits_.cpu().detach().numpy()
          gt = eval_y.view(-1, 1).cpu().detach().numpy()
          accuracy = flat_accuracy(pred, gt)

          eval_loss.append(loss.item())
          eval_accuracy.append(accuracy)
        avg_loss = np.mean(eval_loss, axis=0)
        avg_accuracy = np.mean(eval_accuracy, axis=0)

      if best_accuracy < avg_accuracy:
        best_accuracy = avg_accuracy
        torch.save(model.state_dict(), model_path[KEY])
    if (epoch+1) % 50 == 0:
      print("{:3d}-epoch | trn loss: {:2.2f} | trn accuracy: {:.4f} | val loss: {:2.2f} | val accuracy: {:.4f}".format(epoch+1, loss.item(), accuracy, avg_loss, avg_accuracy))
  print("BEST VAL ACCURACY: {:.4f}".format(best_accuracy))
  return best_accuracy

##########################################################################
def get_input_dataset(KEY):
    train_df = load_data('train.csv')
    test_df = load_data('test.csv')

    if KEY == 'bert':
        # for train
        X = get_bert_vector(train_df['refine_text'])
        train_loader = get_data_loader(X, train_df['label'])
        # for test
        X = get_bert_vector(test_df['refine_text'])
        test_loader = get_data_loader(X, test_df['label'])
        return train_loader, test_loader
    elif KEY == 'conceptnet':
        # for train
        X_vector, _ = get_embedding_vector(train_df['refine_text'], KEY)
        X_mean, X_max = calculate_mean_max(X_vector)
        X_concat = get_X_concat(X_mean, X_max)
        train_loaders = get_data_loader(X_concat, train_df['label'])
        # for test
        X_vector, _ = get_embedding_vector(test_df['refine_text'], KEY)
        X_mean, X_max = calculate_mean_max(X_vector)
        X_concat = get_X_concat(X_mean, X_max)
        test_loaders = get_data_loader(X_concat, test_df['label'])
        return train_loader, test_loader
    elif KEY == 'new-embedding':
        # for train
        X_vector, _ = get_embedding_vector(train_df['refine_text'], KEY)
        X_mean, X_max = calculate_mean_max(X_vector)
        X_concat = get_X_concat(X_mean, X_max)
        train_loaders = get_data_loader(X_concat, train_df['label'])
        # for test
        X_vector, _ = get_embedding_vector(test_df['refine_text'], KEY)
        X_mean, X_max = calculate_mean_max(X_vector)
        X_concat = get_X_concat(X_mean, X_max)
        test_loaders = get_data_loader(X_concat, test_df['label'])
    else:
        print("Key error!")
        return [], []

if __name__ == '__main__':
    #####################################
    USE_CUDA = False
    if torch.cuda.is_available():
        USE_CUDA = True
    DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
    #####################################

    model_input = {"bert": 1536,
                   "conceptnet": 600,
                   "new-embedding": 600}
    model_name = {"bert": "BERT Base Uncased",
                  "conceptnet": "ConceptNet",
                  "new-embedding": "New Embedding"}
    model_path = {"bert": "best-bert-model.pt",
                  "conceptnet": "best-conceptnet-model.pt",
                  "new-embedding": "best-new-embedding-model.pt"}

    #####################################
    B, H, D_out = 32, 100, 3
    b_train_loader, b_test_loader = get_input_dataset('bert')
    c_train_loader, c_test_loader = get_input_dataset('conceptnet')
    #n_train_loader, n_test_loader = get_input_dataset('new-embedding')
    #####################################

    train_loaders = {'bert': b_train_loader,
                     'conceptnet': c_train_loader,
                     'new-embedding': n_train_loader}
    test_loaders = {'bert': b_test_loader,
                    'conceptnet': c_test_loader,
                    'new-embedding': n_test_loader}

    res_comparison = pd.DataFrame(
        columns=['embedding model', 'activation function', 'best val accuracy', 'learning rate'])
    for try_num in range(0, 5):
        best_accuracy_list = []
        for activation_fn in ['ReLU', 'ReLU6', 'RReLU']:
            for lr in [4e-5, 3e-4, 2e-5]:
                for KEY in ['bert', 'conceptnet']:#, 'new-embedding']:
                    D_in = model_input[KEY]
                    model = get_model(D_in, H, D_out, activation_fn)

                    loss_fn = nn.MSELoss(reduction='sum')
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    time_start = get_timestamp()
                    best_accuracy = run_model(model, lr, loss_fn, optimizer, train_loaders[KEY], test_loaders[KEY])
                    time_end = get_timestamp()
                    elapsed_time(start=time_start, end=time_end)
                    if try_num == 0:
                        res_comparison.append({'embedding model': KEY, 'activation function': activation_fn,
                                               'best val accuracy': best_accuracy, 'learning rate': lr},
                                              ignore_index=True)
                    else:
                        best_accuracy_list.append(best_accuracy)
        res_comparison['best val accuracy ' + str(try_num)] = best_accuracy_list

    res_comparison.to_csv('result comparison.csv')




