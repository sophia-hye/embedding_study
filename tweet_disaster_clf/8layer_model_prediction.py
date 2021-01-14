import torch
import re
import pandas as pd
import numpy as np
import requests
import copy
import gensim

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings(action='ignore')


def load_dataset(data_key):
    if data_key == 'news':
        news_dataset = pd.read_excel('./dataset/news_dataset_preprocessing.xlsx')
        del news_dataset['Unnamed: 0']

        lb_encoder = LabelEncoder()
        news_dataset["encode_category"] = lb_encoder.fit_transform(news_dataset["category"])
        news_dataset[["category", "encode_category"]].head()

        return news_dataset
    elif data_key == 'tweet_disaster':
        #tweet_disaster_df = pd.read_csv('./dataset/train_tweet_disaster.csv')
        tweet_disaster_df = pd.read_csv('./dataset/train_refined.csv')
        return tweet_disaster_df
    else:
        print("DATA KEY ERROR: ", data_key)
        return None


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

def get_BERT_vector(sentence_list):
    from transformers import BertModel, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    X_concat = torch.tensor(0)
    for i, paragraph in enumerate(tqdm(sentence_list, desc='BERT embedding', mininterval=1)):
        input_ids = torch.tensor([tokenizer.encode(paragraph, add_special_tokens=True)])
        with torch.no_grad():
            outputs = model(input_ids)
        last_hidden_states = outputs[0]  # [batch, tokens, hidden_dim (768)]

        mean = torch.mean(last_hidden_states, dim=1)  # tokens
        max, _ = torch.max(last_hidden_states, dim=1)  # tokens
        concat = torch.cat((mean, max), dim=1)
        if i == 0:
            X_concat = copy.deepcopy(concat)
        else:
            X_concat = torch.cat((X_concat, concat), dim=0)
    return X_concat


def get_embedding_vector(KEY, sentence_list):
    eliminated_row = []
    total_eliminated_row = 0
    X_concat = torch.tensor(0)
    for i, sent in enumerate(tqdm(sentence_list, desc=KEY+' embedding', mininterval=1)):
        tokens = sent.split()
        vectors = []
        if KEY == 'new-embedding':
            for token in tokens:
                if token in new_word_lower:
                    # when new word, get embedding from new-embedding model
                    vector = new_embedding_dict.get(token)
                #elif token in convert_word_lower:
                #    vector = convert_embedding_dict.get(token)
                else:
                    # when not new word, get just conceptnet embedding
                    vector = embedding_dict.get(token)

                if vector is not None:
                    vector = np.fromstring(vector, dtype=float, sep=' ')  # when return: embedding
                    vectors.append(vector)
        elif KEY == 'conceptnet':
            #sent = re.sub('\[|\]|:|(|)|\*|#|@|!|\?|\.', '', sent)
            #tokens = sent.lower().split()
            #total = 0
            for token in tokens:
                if token in new_word_lower:
                    # when new word, don't get embedding from conceptnet
                    vector = None
                else:
                    # when not new word, get conceptnet embedding
                    vector = embedding_dict.get(token)

                if vector is not None:
                    #total += 1
                    vector = np.fromstring(vector, dtype=float, sep=' ')  # when return: embedding
                    vectors.append(vector)
        elif KEY == 'glove':
            for token in tokens:
                if token in new_word_lower:
                    vector = None
                else:
                    vector = glove_embedding_dict.get(token)

                if vector is not None:
                    vectors.append(vector)
        elif KEY == 'word2vec':
            for token in tokens:
                if token in new_word_lower:
                    vector = None
                else:
                    if token in word2vec_model:
                        vector = word2vec_model[token]
                    else:
                        vector = None

                if vector is not None:
                    vectors.append(vector)
        else:
            print("KEY Error: ", KEY)
            break

        if vectors == []:
            eliminated_row.append(i)
            total_eliminated_row += 1
        else:
            tensor_vec = torch.tensor(vectors)
            mean = torch.mean(tensor_vec, dim=0)  # tokens
            max, _ = torch.max(tensor_vec, dim=0)  # tokens
            concat = torch.cat((mean, max), dim=0)
            concat = concat.unsqueeze_(dim=0) # 맨앞에 차원 하나 추가 [1, 600]
            if i == 0:
                X_concat = copy.deepcopy(concat)
            else:
                X_concat = torch.cat((X_concat, concat), dim=0)

    """
    if KEY == 'new-embedding':
        df_vec = pd.DataFrame({'X_vector': X_vector})
        df_vec.to_csv('X_vector_newEmbedding.csv', index=False, header=False)
    elif KEY == 'conceptnet':
        df_vec = pd.DataFrame({'X_vector': X_vector})
        df_vec.to_csv('X_vector_conceptnet.csv', index=False, header=False)
    elif KEY == 'glove':
        df_vec = pd.DataFrame({'X_vector': X_vector})
        df_vec.to_csv('X_vector_glove.csv', index=False, header=False)
    elif KEY == 'word2vec':
        df_vec = pd.DataFrame({'X_vector': X_vector})
        df_vec.to_csv('X_vector_word2vec.csv', index=False, header=False)
    else:
        print("KEY Error: ", KEY)
    """

    print("total eliminated row: {}\n".format(total_eliminated_row))
    return X_concat, eliminated_row

def reset_category_list(category_list, eliminated_row):
    for index in sorted(eliminated_row, reverse=True):
        del category_list[index]
    return category_list

##########################################################################
def get_input_dataset(KEY, sentence_list, category_list):
    if KEY == 'bert':
        X_concat = get_BERT_vector(sentence_list)
        X_train, X_test, y_train, y_test = get_train_test_split(X_concat, category_list)
        train_loader, test_loader = get_train_test_loader(X_train, X_test, y_train, y_test)
        return train_loader, test_loader
    else:
        X_concat, eliminated_row = get_embedding_vector(KEY, sentence_list)
        category_list = reset_category_list(category_list, eliminated_row)
        X_train, X_test, y_train, y_test = get_train_test_split(X_concat, category_list)
        train_loader, test_loader = get_train_test_loader(X_train, X_test, y_train, y_test)
        return train_loader, test_loader
##########################################################################

def get_train_test_split(X_concat, category_list):
  from sklearn.model_selection import train_test_split

  y = torch.tensor(category_list)
  X_train, X_test, y_train, y_test = train_test_split(X_concat, y,
                                                      train_size=0.8,
                                                      random_state=928,
                                                      shuffle=True,
                                                      stratify=y)
  return X_train, X_test, y_train, y_test

def get_train_test_loader(X_train, X_test, y_train, y_test):
  from torch.utils.data import TensorDataset, DataLoader

  train_loader = DataLoader(TensorDataset(X_train, y_train),
                            batch_size = B,
                            shuffle = True,
                            num_workers = 2)
  test_loader = DataLoader(TensorDataset(X_test, y_test),
                          batch_size = B,
                          shuffle = True,
                          num_workers = 2)
  return train_loader, test_loader

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

  print(f"\nEMBEDDING MODEL: {model_name[KEY]}")
  print(f"learning rate: {lr}\nloss function: {loss_fn.children}")
  print(f"Activation function: {activation_fn}\noptimizer: {optimizer}\n")
  best_accuracy = 0
  best_trn_accuracy = 0
  for epoch in range(500):
    trn_loss, trn_accuracy = [], []
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

      trn_loss.append(loss.item())
      trn_accuracy.append(accuracy)  # for mean accuracy every batch

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
          #torch.save(model.state_dict(), model_path[KEY])

    trn_avg_loss = np.mean(trn_loss, axis=0)
    trn_avg_accuracy = np.mean(trn_accuracy, axis=0)
    if best_trn_accuracy < accuracy:
        best_trn_accuracy = accuracy
    if (epoch+1) % 50 == 0:
      print("{:3d}-epoch | trn loss: {:2.2f} | trn best accuracy: {:.4f} | val loss: {:2.2f} | val best accuracy: {:.4f}".format(epoch+1, trn_avg_loss, accuracy, avg_loss, best_accuracy))
  print("BEST TRN ACCURACY: {:.4f}".format(best_trn_accuracy))
  print("BEST VAL ACCURACY: {:.4f}".format(best_accuracy))
  print("\n#####################################\n")
  return best_accuracy


if __name__ == '__main__':
    #data_df = load_dataset('news')
    data_df = load_dataset('tweet_disaster')
    #####################################
    USE_CUDA = False
    if torch.cuda.is_available():
        USE_CUDA = True
    DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
    #####################################

    model_input = {"bert": 1536,
                   "conceptnet": 600,
                   "new-embedding": 600,
                   "glove": 600,
                   "word2vec": 600,}
    model_name = {"bert": "BERT Base Uncased",
                  "conceptnet": "ConceptNet",
                  "new-embedding": "New Embedding",
                  "glove": "Glove",
                  "word2vec": "Word2Vec"}
    model_path = {"bert": "best-bert-model.pt",
                  "conceptnet": "best-conceptnet-model.pt",
                  "new-embedding": "best-new-embedding-model.pt",
                  "glove": "best-glove-model.pt",
                  "word2vec": 'best-word2vec-model.pt'}

    #####################################
    # Make embedding dictionary
    embedding_list = pd.read_csv('wordEmbedding_en.csv', encoding='utf-8', usecols=['word', 'vec300'], sep=' ')
    embedding_dict = embedding_list.set_index('word')['vec300'].to_dict()


    print('8layer model')
    # --------------------------------------------------------
    # new word embedding (prediction)
    new_embedding_list = pd.read_csv('prediction_embedding_keyword_8layer.csv')

    new_word_lower = []
    for w in new_embedding_list['new_word']:
        new_word_lower.append(w.lower())
    new_embedding_list['new_word_lower'] = new_word_lower

    # DataFrame to dictionary
    new_embedding_dict = new_embedding_list.set_index('new_word_lower')['embedding'].to_dict()

    # ------------------------------------------------------------------------------------------------------------------
    # disaster tweets data: new-embedding
    for i in range(1):
        KEY = 'new-embedding'

        init_all_random(SET_SEED=True,
                        r_seed=2147483648,
                        np_seed=2147483648,
                        torch_seed=17043145292842172883 & ((1 << 63) - 1),
                        cuda_seed=5039337088303087)
        # init_all_random(SET_SEED=False)
        D_in = model_input[KEY]
        B, H, D_out = 32, 100, 2

        # train_loader, test_loader = get_input_dataset(KEY, data_df['refine_text'], data_df['target'])
        train_loader, test_loader = get_input_dataset(KEY, data_df['keyword_text'], data_df['target'].tolist())

        model = get_model(D_in, H, D_out, 'ReLU')
        loss_fn = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=3e-3)
        time_start = get_timestamp()
        best_accuracy = run_model(model, 3e-3, loss_fn, optimizer, train_loader, test_loader)
        time_end = get_timestamp()
        elapsed_time(start=time_start, end=time_end)
        print('\n\n')

    """
    # ------------------------------------------------------------------------------------------------------------------
    # pretrained 300차원 glove 임베딩 데이터 로드
    glove_embedding_dict = dict()
    f = open('glove.6B.300d.txt', encoding="utf8")

    for line in f:
        splited = line.split()
        word = splited[0]
        vector_arr = np.asarray(splited[1:], dtype='float32')
        glove_embedding_dict[word] = vector_arr
    f.close()

    for i in range(1):
        KEY = 'glove'
        init_all_random(SET_SEED=True,
                        r_seed=2147483648,
                        np_seed = 2147483648,
                        torch_seed=17043145292842172883 & ((1<<63)-1),
                        cuda_seed=5039337088303087)
        D_in = model_input[KEY]
        B, H, D_out = 32, 100, 2

        # train_loader, test_loader = get_input_dataset(KEY, data_df['refine_text'], data_df['target'])
        train_loader, test_loader = get_input_dataset(KEY, data_df['keyword_text'], data_df['target'].tolist())

        loss_fn = nn.MSELoss(reduction='sum')
        lr = 3e-2
        model = get_model(D_in, H, D_out, 'ReLU')
        optimizer = optim.Adam(model.parameters(), lr=lr)

        time_start = get_timestamp()
        best_accuracy = run_model(model, lr, loss_fn, optimizer, train_loader, test_loader)
        time_end = get_timestamp()
        elapsed_time(start=time_start, end=time_end)
    
    #------------------------------------------------------------------------------------------------------------------
    # 구글의 사전 훈련된 Word2vec 모델 로드
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    for i in range(1):
        KEY = 'word2vec'
        init_all_random(SET_SEED=True,
                        r_seed=2147483648,
                        np_seed=2147483648,
                        torch_seed=17043145292842172883 & ((1 << 63) - 1),
                        cuda_seed=5039337088303087)
        D_in = model_input[KEY]
        B, H, D_out = 32, 100, 2

        # train_loader, test_loader = get_input_dataset(KEY, data_df['refine_text'], data_df['target'])
        train_loader, test_loader = get_input_dataset(KEY, data_df['keyword_text'], data_df['target'].tolist())

        loss_fn = nn.MSELoss(reduction='sum')
        lr = 3e-2
        model = get_model(D_in, H, D_out, 'ReLU')
        optimizer = optim.Adam(model.parameters(), lr=lr)

        time_start = get_timestamp()
        best_accuracy = run_model(model, lr, loss_fn, optimizer, train_loader, test_loader)
        time_end = get_timestamp()
        elapsed_time(start=time_start, end=time_end)

    # ------------------------------------------------------------------------------------------------------------------
    # bert model result
    for i in range(1):
        KEY = 'bert'
        init_all_random(SET_SEED=True,
                        r_seed=2147483648,
                        np_seed=2147483648,
                        torch_seed=17043145292842172883 & ((1 << 63) - 1),
                        cuda_seed=5039337088303087)
        D_in = model_input[KEY]
        B, H, D_out = 32, 100, 2

        # train_loader, test_loader = get_input_dataset(KEY, data_df['refine_text'], data_df['target'])
        train_loader, test_loader = get_input_dataset(KEY, data_df['keyword_text'], data_df['target'].tolist())

        loss_fn = nn.MSELoss(reduction='sum')
        lr = 3e-2
        model = get_model(D_in, H, D_out, 'ReLU')
        optimizer = optim.Adam(model.parameters(), lr=lr)

        time_start = get_timestamp()
        best_accuracy = run_model(model, lr, loss_fn, optimizer, train_loader, test_loader)
        time_end = get_timestamp()
        elapsed_time(start=time_start, end=time_end)
"""