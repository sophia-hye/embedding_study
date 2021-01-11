import re
import copy
import gensim
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings(action='ignore')


def elapsed_time(start, end):
    from datetime import datetime

    fmt = '%Y-%m-%d %H:%M:%S'
    time_start = datetime.strptime(start, fmt)
    time_end = datetime.strptime(end, fmt)

    diff = time_end - time_start

    h = int(diff.seconds / 3600)
    m = int((diff.seconds % 3600) / 60)
    s = (diff.seconds % 3600) % 60

    print("elapsed time: {:1d}days {:2d}:{:2d}:{:2d}".format(diff.days, h, m, s))


def get_timestamp():
    from datetime import datetime

    fmt = '%Y-%m-%d %H:%M:%S'
    return datetime.now().strftime(fmt)


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


def get_new_embedding_vector(KEY):
    print(f"\nNEW EMBEDDING MODEL: {KEY}")

    # new word embedding (prediction of models)
    fname = "./prediction/prediction_embedding_keyword_" + KEY + ".csv"
    prediction_list = pd.read_csv(fname)

    new_words = []
    for w in prediction_list['new_word']:
        new_words.append(w.lower())
    prediction_list['new_word_lower'] = new_words

    # Convert DataFrame to dictionary
    dict = prediction_list.set_index('new_word_lower')['embedding'].to_dict()
    return dict


def get_embedding_vector(KEY, sentence_list, MVS=False):
    print(f"\nEXISTING EMBEDDING MODEL: {KEY}")
    print(f"\nUSE NEW EMBEDDING MODEL? {MVS}")
    eliminated_row = []
    total_eliminated_row = 0
    new_words = list(new_embedding_dict.keys())

    X_concat = torch.tensor(0)
    for i, sent in enumerate(tqdm(sentence_list, desc=KEY + ' embedding', mininterval=1)):
        tokens = sent.split()
        vectors = []
        for token in tokens:
            if token in new_words:
                if MVS:
                    # when new word, get embedding from new embedding model
                    vector = new_embedding_dict.get(token)
                    vector = np.fromstring(vector, dtype=float, sep=' ')
                    # print("MVS: ", len(vector))
                else:
                    vector = None
            else:
                vector = None
                if KEY == 'conceptnet' or KEY == 'glove':
                    vector = existing_embedding_dict.get(token)
                    # print(KEY, "\t", type(vector))
                elif KEY == 'w2v':
                    if token in existing_embedding_dict:
                        vector = existing_embedding_dict[token]

            if vector is not None:
                if KEY == 'conceptnet':
                    vector = np.fromstring(vector, dtype=float, sep=' ')  # when return: embedding
                vectors.append(vector)

        if vectors == []:
            eliminated_row.append(i)
            total_eliminated_row += 1
        else:
            tensor_vec = torch.tensor(vectors)
            # print("vectors: ", len(vectors))
            mean = torch.mean(tensor_vec, dim=0)  # tokens
            max, _ = torch.max(tensor_vec, dim=0)  # tokens
            concat = torch.cat((mean, max), dim=0)
            concat = concat.unsqueeze_(dim=0)  # 맨앞에 차원 하나 추가 [1, 600]
            if i == 0:
                X_concat = copy.deepcopy(concat)
            else:
                X_concat = torch.cat((X_concat, concat), dim=0)

    print("total eliminated row: {}\n".format(total_eliminated_row))
    return X_concat, eliminated_row


def reset_category_list(category_list, eliminated_row):
    for index in sorted(eliminated_row, reverse=True):
        del category_list[index]
    return category_list


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
                              batch_size=B,
                              shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=B,
                             shuffle=True,
                             num_workers=2)
    return train_loader, test_loader


##########################################################################
def get_input_dataset(KEY, sentence_list, category_list):
    X_concat, eliminated_row = get_embedding_vector(KEY, sentence_list, MVS)
    category_list = reset_category_list(category_list, eliminated_row)
    X_train, X_test, y_train, y_test = get_train_test_split(X_concat, category_list)
    train_loader, test_loader = get_train_test_loader(X_train, X_test, y_train, y_test)
    return train_loader, test_loader
##########################################################################


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def run_model(model, lr, loss_fn, optimizer, train_loader, test_loader):
    activation_fn = list(model.children())[1]

    print(f"learning rate: {lr}\nloss function: {loss_fn.children}")
    print(f"Activation function: {activation_fn}\noptimizer: {optimizer}\n")

    best_trn_accuracy, best_val_accuracy = 0, 0
    for epoch in range(500):
        model.train()
        for X, y in train_loader:
            X, y = X.float(), y.float()
            if USE_CUDA:
                X, y = X.cuda(), y.cuda()

            model.zero_grad()

            logits = model(X)  # [batch, D_out], logits=scores
            probs = F.softmax(logits, dim=1)  # normalization -> probabilities
            y_pred, _ = torch.max(probs, dim=1)  # y_pred : [batch]

            trn_loss = loss_fn(y_pred, y)
            trn_loss.backward()
            optimizer.step()

            pred = probs.cpu().detach().numpy()
            gt = y.view(-1, 1).cpu().detach().numpy()
            trn_accuracy = flat_accuracy(pred, gt)
            if best_trn_accuracy < trn_accuracy:
                best_trn_accuracy = trn_accuracy

            with torch.no_grad():
                model.eval()
                eval_loss, eval_accuracy = [], []
                for eval_X, eval_y in test_loader:
                    eval_X, eval_y = eval_X.float(), eval_y.float()
                    if USE_CUDA:
                        eval_X, eval_y = eval_X.cuda(), eval_y.cuda()

                    logits = model(eval_X)
                    probs = F.softmax(logits, dim=1)
                    y_pred, _ = torch.max(probs, dim=1)

                    val_loss = loss_fn(y_pred, eval_y)

                    pred = probs.cpu().detach().numpy()
                    gt = eval_y.view(-1, 1).cpu().detach().numpy()
                    val_accuracy = flat_accuracy(pred, gt)

                    eval_loss.append(val_loss.item())
                    eval_accuracy.append(val_accuracy)
                avg_loss = np.mean(eval_loss, axis=0)
                avg_accuracy = np.mean(eval_accuracy, axis=0)
                if best_val_accuracy < avg_accuracy:
                    best_val_accuracy = avg_accuracy
                    # torch.save(model.state_dict(), model_path[KEY])

        if (epoch + 1) % 50 == 0:
            print(
                "{:3d}-epoch | trn loss: {:.4f} | trn accuracy: {:.4f} | val loss: {:.4f} | val accuracy: {:.4f}".format(
                    epoch + 1, trn_loss, trn_accuracy, avg_loss, avg_accuracy))

    print("BEST TRN ACCURACY: {:.4f}".format(best_trn_accuracy))
    print("BEST VAL ACCURACY: {:.4f}".format(best_val_accuracy))


if __name__ == '__main__':
    # Load data as DataFrame
    tweet_disaster_df = pd.read_csv('./dataset/train_refined.csv')
    fpath = "/home/ida/workspace/jihye/dataset/"

    USE_CUDA = False
    if torch.cuda.is_available():
        USE_CUDA = True
    DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

    # Make embedding dictionary (existing embedding model)
    """
    # ConceptNet Numberbatch --------------------------------------------------------
    KEY = "conceptnet"
    existing_embedding_list = pd.read_csv(fpath+'wordEmbedding_en.csv', encoding='utf-8', usecols=['word', 'vec300'], sep=' ')
    existing_embedding_dict = existing_embedding_list.set_index('word')['vec300'].to_dict()
    
    # GloVe (Load pre-trained 300dims GloVe Embedding data) -------------------------
    KEY = "glove"
    existing_embedding_dict = dict()
    f = open(fpath+'glove.6B.300d.txt', encoding="utf8")

    for line in f:
        splited = line.split()
        word = splited[0]
        vector_arr = np.asarray(splited[1:], dtype='float32')
        existing_embedding_dict[word] = vector_arr
    f.close()
    """
    # Word2Vec (Load pre-trained 300dims Word2Vec Embedding data) ----------------------
    KEY = "w2v"
    existing_embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(fpath+'GoogleNews-vectors-negative300.bin.gz',
                                                                              binary=True)

    new_embedding_fname = {"fclayer": "mvs-s",  # Single
                           "2d8layer": "mvs-c",  # Crescendo
                           "1d8layer": "mvs-cd",  # Crescendo-Decrescendo
                           "model": "mvs-d",  # Decrescendo
                           "w2v": "w2v",  # Word2Vec
                           "glove": "glove"}  # GloVe

    # disaster tweets data: new-embedding -----------------------------------------------------------
    new_embedding_dict = get_new_embedding_vector(KEY=new_embedding_fname["w2v"])
    MVS = True

    for i in range(1):
        init_all_random(SET_SEED=True,
                        r_seed=2147483648,
                        np_seed=2147483648,
                        torch_seed=17043145292842172883 & ((1 << 63) - 1),
                        cuda_seed=5039337088303087)
        # init_all_random(SET_SEED=False)

        # Initialize model and Setting
        B, D_in, H, D_out = 32, 600, 100, 2  # 600 dims = mean 300 dims + max 300 dims
        model = nn.Sequential(nn.Linear(D_in, H),
                              nn.ReLU(),
                              nn.Linear(H, D_out), )
        if USE_CUDA:
            model = model.cuda()
        lr = 3e-3
        loss_fn = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Get model's input data
        train_loader, test_loader = get_input_dataset(KEY,
                                                      tweet_disaster_df['keyword_text'],
                                                      tweet_disaster_df['target'].tolist())

        time_start = get_timestamp()
        run_model(model, lr, loss_fn, optimizer, train_loader, test_loader)
        time_end = get_timestamp()
        elapsed_time(start=time_start, end=time_end)
        print('\n\n')
