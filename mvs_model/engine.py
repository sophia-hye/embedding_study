"""funcs for training"""
import time 

import numpy as np
import torch
from tqdm import tqdm

import utils
import config as cfg

from torch.optim.lr_scheduler import ReduceLROnPlateau


def do_train_at(model_i,
				epoch_i,
				train_data_loader,
				model,
				loss_criterion,
				accuracy_criterion,
				optimizer,
				scheduler,
				use_lr_warmup=False):
	"""do train at (model_i, epoch_i)"""
	
	losses = utils.AverageMeter()
	accs = utils.AverageMeter()
	
	num_train_batch = len(train_data_loader)
	
	t0 = time.time()
	
	#for step, batch in enumerate(tqdm(train_data_loader, desc="Training ", ncols=100, mininterval=1)):
	for step, batch in enumerate(train_data_loader):

		model.train()  # MUST

		x, y_true = batch

		if cfg.USE_CUDA:
			x = x.cuda()
			y_true = y_true.cuda()
		
		model.zero_grad()  # MUST
		
		y_pred = model(x)  # forward propagation
		y_pred = y_pred.squeeze()  # for cnn_1d_*

		"""
		if step==0:
			print('y_pred: ', y_pred)
			print('y_true: ', y_true)
		"""

		# back-propagation
		loss = loss_criterion(y_pred, y_true)
		loss.backward()  # compute gradient of the loss with respect to model parameters
		optimizer.step()  # the optimizer updates model's parameters by loss
		if use_lr_warmup:
			scheduler.step(epoch_i)  # [!] option
		else:
			pass ##scheduler.step()  # lr_scheduler / optimizer에 따라 수정 필요

		losses.update(loss.item(), y_true.size(0))

		accuracy = accuracy_criterion(y_pred, y_true)
		accs.update(torch.mean(accuracy), y_true.size(0))

	return {
		'time': utils.format_time(time.time()-t0),
		'loss': losses.avg,
		'acc': accs.avg
	}


def do_validation_at(valid_data_loader,
					 model,
					 loss_criterion,
					 accuracy_criterion
					 ):
	"""do validation"""
	model.eval()  # MUST
	
	t0 = time.time()

	losses = utils.AverageMeter()
	accs = utils.AverageMeter()

	#for step, batch in enumerate(tqdm(valid_data_loader, desc="Validating ", ncols=100, mininterval=1)):
	for step, batch in enumerate(valid_data_loader):

		x, y_true = batch

		if cfg.USE_CUDA:
			x = x.cuda()
			y_true = y_true.cuda()
		
		with torch.no_grad():  # MUST
			y_pred = model(x)
			y_pred = y_pred.squeeze()

			"""
			if step==0:
				print('y_pred: ', y_pred)
				print('y_true: ', y_true)
			"""
		loss = loss_criterion(y_pred, y_true)
		losses.update(loss.item(), y_true.size(0))
		
		accuracy = accuracy_criterion(y_pred, y_true)
		accs.update(torch.mean(accuracy), y_true.size(0))

	return {
		'time': utils.format_time(time.time()-t0),
		'loss': losses.avg,
		'acc': accs.avg,
	}
