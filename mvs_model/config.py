"""SET global parameters"""
import os
import torch

SEEDS = [123456789]
DO_TRAIN = True

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

DATA_KEY = "english_23870_first"
DATASET = {"english_23870_first": 193,
           "english_23870_all_merged": 2161,
           "english_23870_all": 193,
           "english_23870_glove": 171,
           "english_23870_w2v": 144,
           "english_23870_conceptnet": 160,}

BASIC_PATH = "/home/ida/workspace/jihye/dataset"
DATA_PATH = os.path.join(BASIC_PATH, DATA_KEY, "vectors")
SAVE_PATH = os.path.join("/home/ida/workspace/jihye/model_pipeline/mvs_conceptnet", "output_model_10_11931")
LOAD_PATH = os.path.join("/home/ida/workspace/jihye/model_pipeline/mvs_glove", "output_model")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)  # if the folder exist, do nothing. Otherwise, make folder(s)

MAX_DIM = DATASET[DATA_KEY]
TRAIN_PROB = 0.8

ADAMW_DECAY_RATE = 0.01  # [TODO] AdamW 웨잇 디케이 정도

# DROPOUT_RATE = 0.2
WARMUP_FRACTION = 0.1 

NUM_MODEL = 30
ENSEMBLE = True
TOTAL_EMB = 300
DATA_LOADER_SHUFFLE = True

IN_DIM_LIST = {
    "cnn_2d_8layer": 1,  # 300 prediction
    "cnn_2d_8layer_10": 1,  # 10 prediction
    "cnn_2d_fclayer": 1,  # 300 prediction
    "cnn_2d_fclayer_10": 1,  # 10 prediction
    "cnn_1d_8layer": 300,
    "cnn_1d_10layer": 300,
    "cnn_1d_11layer": 300}  # model.py
OUT_DIM_LIST = {
    "cnn_2d_8layer": 300,
    "cnn_2d_8layer_10": 10,
    "cnn_2d_fclayer": 300,
    "cnn_2d_fclayer_10": 10,
    "cnn_1d_8layer": 300,
    "cnn_1d_10layer": 30,
    "cnn_1d_11layer": 10}
NUM_FILTER_LIST = {
    "cnn_2d_8layer": 2,
    "cnn_2d_8layer_10": 1,
    "cnn_2d_fclayer": 1,
    "cnn_2d_fclayer_10": 1,
    "cnn_1d_8layer": 300,
    "cnn_1d_10layer": 30,
    "cnn_1d_11layer": 30}

MODEL_KEY = "cnn_1d_11layer"  # model.py
# MODEL_KEY = "cnn_1d_8layer"
# MODEL_KEY = "cnn_2d_8layer"
# MODEL_KEY = "cnn_2d_fclayer_10"
IN_DIM = IN_DIM_LIST[MODEL_KEY]
OUT_DIM = OUT_DIM_LIST[MODEL_KEY]
NUM_FILTER = NUM_FILTER_LIST[MODEL_KEY]

KERNEL = 2
STRIDE = 1
PADDING = 1
# PADDING = 0  # when 2d-cnn

EPOCHS = 500
BATCH_SIZE = 32

LR = 3e-3  # Learning rate

ES_PATIENCE = 15  # for Early Stopping
ES_DELTA = 0.0001  # for Early Stopping


#### For eval
MAX_SEQ_LEN = 100