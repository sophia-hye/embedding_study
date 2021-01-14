"""main func for training"""
import os
import re
import time
import random

import ujson
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader

import config as cfg
import mvs_d as m  # model
#import mvs_cd as m  # cnn_1d_8layer
#import mvs_c as m  # cnn_2d_8layer
#import mvs_s as m  # cnn_2d_fclayer
import engine 
import utils
import dataloader
import warmup
# import warmup_scheduler


#####################################
def set_seed(seed=777):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
#####################################

if __name__ == '__main__':
	SEEDS = cfg.SEEDS

	USE_CUDA = cfg.USE_CUDA

	best_trn_stats, best_val_stats = [], []
	#print("SET SEED {} -------------------------------------------------------------------".format(seed))
	#set_seed(seed=seed)
	print('\n\ncurrent torch\'s random seed {}'.format(torch.cuda.initial_seed()))

	train_list, valid_list = dataloader.get_train_val_list(cfg.DATA_PATH)

	unit = int(cfg.TOTAL_EMB/cfg.NUM_MODEL)
	num_model = cfg.NUM_MODEL
	print('total emb: {}\tnum model: {}\tunit: {}'.format(cfg.TOTAL_EMB, cfg.NUM_MODEL, unit))
	for model_i in range(0, 5):
		# initialize seed again whenever each model start to train
		# set_seed(seed=seed)

		start = int(unit * model_i)
		end = int(unit * (model_i + 1))
		print('start: {}\tend: {}'.format(start, end))

		train_dataset = dataloader.WordEmbeddingDataLoader(lists=train_list,
														   cnn_1d=True,  # 2d data
														   ensemble=cfg.ENSEMBLE,
														   train=True,
														   start=start,
														   end=end)
		valid_dataset = dataloader.WordEmbeddingDataLoader(lists=valid_list,
														   cnn_1d=True,
														   ensemble=cfg.ENSEMBLE,
														   train=True,
														   start=start,
														   end=end)
		"""
		train_dataset = dataloader.WordEmbeddingDataLoader(train_list, False)
		valid_dataset = dataloader.WordEmbeddingDataLoader(valid_list, False)
		"""

		# divide dataset into batch size
		train_data_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.DATA_LOADER_SHUFFLE)
		valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.DATA_LOADER_SHUFFLE)

		# for others
		model = m.CNN(cfg.IN_DIM, cfg.OUT_DIM, cfg.NUM_FILTER, cfg.KERNEL, cfg.STRIDE, cfg.PADDING)
		# for fc layer
		# model = m.CNN(cfg.IN_DIM, cfg.NUM_FILTER, cfg.OUT_DIM, cfg.STRIDE, cfg.PADDING, cfg.MAX_DIM)
		if cfg.USE_CUDA:
			model = model.cuda()

		# loss measure
		loss_criterion = nn.MSELoss(reduction='sum')  # L2 loss (Euclidean distance) reduction='sum'

		# accuracy measure : cosine similarity
		# eps: small value to avoid division by zero (default=1e-8)
		accuracy_criterion = nn.CosineSimilarity(dim=1)

		optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
		lr_scheduler = warmup.LearningRateWarmUP(optimizer=optimizer,
												 warmup_iteration=cfg.EPOCHS*cfg.WARMUP_FRACTION,
												 target_lr=cfg.LR)
		use_lr_warmup = True
		ReduceLROnPlateau_scheduler = False

		es = utils.EarlyStopping(patience=cfg.ES_PATIENCE, mode="max", delta=cfg.ES_DELTA)

		best_trn_stat, best_val_stat = None, None
		for epoch_i in range(0, cfg.EPOCHS):

			if epoch_i == cfg.EPOCHS*cfg.WARMUP_FRACTION:
				use_lr_warmup = False
				optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
				lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)  # learning rate scheduler (learning rate decay)
				ReduceLROnPlateau_scheduler = True

			print('\n[model-{} {}-epoch]'.format(model_i, epoch_i))
			train_stat = engine.do_train_at(model_i,
											epoch_i,
											train_data_loader,
											model,
											loss_criterion,
											accuracy_criterion,
											optimizer,
											lr_scheduler,
											use_lr_warmup)
			if best_trn_stat is None:
				best_trn_stat = train_stat
				#best_trn_stats.append(best_trn_stat)
			elif train_stat['acc'] > best_trn_stat['acc'] + es.delta:
				best_trn_stat = train_stat
				#best_trn_stats.append(best_trn_stat)
			else: pass  # model is not improved

			valid_stat = engine.do_validation_at(valid_data_loader,
												 model,
												 loss_criterion,
												 accuracy_criterion)

			if ReduceLROnPlateau_scheduler:
				lr_scheduler.step(valid_stat['loss'])

			es(valid_stat['acc'].detach().cpu().numpy(), model, model_path='{}/model_{}.pt'.format(cfg.SAVE_PATH, model_i))

			if es.early_stop:
				print("Early stopping at model-{} {}-epoch".format(model_i, epoch_i))
				break

			if best_val_stat is None:
				best_val_stat = valid_stat
				#best_val_stats.append(best_val_stat)
			elif valid_stat['acc'] > best_val_stat['acc'] + es.delta:
				best_val_stat = valid_stat
				#best_val_stats.append(best_val_stat)
			else: pass  # model is not improved

			print("model-{} {:3d}-epoch | trn loss: {:.6f} | val loss: {:.6f} | trn accuracy: {:.4f} | val accuracy: {:.4f} | lr: {:.2e}".format
				  (model_i, epoch_i, train_stat['loss'], valid_stat['loss'], train_stat['acc'], valid_stat['acc'],
				   optimizer.param_groups[0]['lr']))

		best_trn_stats.append(best_trn_stat)
		best_val_stats.append(best_val_stat)
	# end for loop w.r.t. model_i

	# ---------train------------------------------------------------------
	df = pd.DataFrame(best_trn_stats)
	fname_trn = '{}/best_accuracy_trn_{}.csv'.format(cfg.SAVE_PATH, model_i)
	df.to_csv(fname_trn)

	scores = [stat['acc'] for stat in best_trn_stats]
	avg_tmp = sum(scores) / len(scores)
	avg_score = round(avg_tmp.item(), 4)
	print("Average train accuracy: {}".format(avg_score))

	# ---------validation-------------------------------------------------
	df = pd.DataFrame(best_val_stats)
	fname_val = '{}/best_accuracy_val_{}.csv'.format(cfg.SAVE_PATH, model_i)
	df.to_csv(fname_val)

	scores = [stat['acc'] for stat in best_val_stats]
	avg_tmp = sum(scores) / len(scores)
	avg_score = round(avg_tmp.item(), 4)
	print("Average validation accuracy: {}".format(avg_score))

	df = None  # for cuda memory saving
	best_trn_stats, best_val_stats = [], []
	print('FINISH TO TRAIN MODELS\n')

	"""
	scores = [stat['acc'] for stat in stats]
	avg_tmp = sum(scores)/len(scores)
	avg_score = round(avg_tmp.item(), 4)
	with open('{}/model_{}_avg_accuracy_{}.json'.format(cfg.SAVE_PATH, model_i, avg_score), 'w') as f:
		f.write(ujson.dumps(stats))
	print('\n[Avg Val Score] -> {}\n'.format(avg_score))
	"""
