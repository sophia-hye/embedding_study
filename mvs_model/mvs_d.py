"""Build Model"""
import torch
import torch.nn as nn
import torch.optim as optim

#import config as cfg


class CNN(nn.Module):
	# @in_dim: dimension of inputted data
	# @out_dim: dimension of output data which user want to get
	# @nf: number of filters
	
	def __init__(self, in_dim, out_dim, nf, kernel, stride, padding,
				 dropout_rate=0.1, device='cpu'):
		super(CNN, self).__init__()  # Always start inheriting torch.nn.Module

		self.in_dim = in_dim
		self.out_dim = out_dim
		self.nf = nf  # 30
		self.kernel = kernel
		self.stride = stride
		self.padding = padding

		self.device = device

		self.dropout = nn.Dropout(dropout_rate)

		self.net = nn.Sequential(
			nn.Conv1d(self.in_dim, self.nf * 9, self.kernel, self.stride, self.padding, bias=False),
			nn.MaxPool1d(2),
			nn.ReLU(True),
			nn.Conv1d(self.nf * 9, self.nf * 8, self.kernel, self.stride, self.padding, bias=False),
			nn.MaxPool1d(2),
			nn.ReLU(True),
			nn.Conv1d(self.nf * 8, self.nf * 7, self.kernel, self.stride, self.padding, bias=False),
			nn.MaxPool1d(2),
			nn.ReLU(True),
			nn.Conv1d(self.nf * 7, self.nf * 6, self.kernel, self.stride, self.padding, bias=False),
			nn.MaxPool1d(2),
			nn.ReLU(True),
			nn.Conv1d(self.nf * 6, self.nf * 5, self.kernel, self.stride, self.padding, bias=False),
			nn.MaxPool1d(2),
			nn.ReLU(True),
			nn.Conv1d(self.nf * 5, self.nf * 4, self.kernel, self.stride, self.padding, bias=False),
			nn.MaxPool1d(2),
			nn.ReLU(True),
			nn.Conv1d(self.nf * 4, self.nf * 3, self.kernel, self.stride, self.padding, bias=False),
			nn.MaxPool1d(2),
			nn.ReLU(True),
			nn.Conv1d(self.nf * 3, self.nf * 2, self.kernel, self.stride, self.padding, bias=False),
			nn.MaxPool1d(2),
			nn.ReLU(True),
			nn.Conv1d(self.nf * 2, self.out_dim, self.kernel, self.stride, self.padding, bias=False),
			nn.MaxPool1d(2),
			nn.Tanh()  # range -1 ~ 1
		)

		if torch.cuda.is_available():
			self.net = self.net.cuda()

		# torch.nn.init.xavier_uniform_(self.net.weight)
		# sequential object has no weight
		
	def forward(self, x):
		y_pred = self.net(x)
		y_pred = torch.squeeze(y_pred, 1)  # [batch size, embedding size]

		return y_pred

