import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# construct model on cuda if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class LSTM(nn.Module):
    # @in_dim: dimension of inputted data
    # @out_dim: dimension of output data which user want to get
    # @hidden_dim: dimension of hidden layer
    # @num_layers: number of hidden layer(s)
    def __init__(self, in_dim, hidden_dim, num_layers,
                 drop_prob=0, is_batch_first=False, is_bidirectional=False):
        super(LSTM, self).__init__()  # Always start inheriting torch.nn.Module

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.is_batch_first = is_batch_first  # True or False
        self.is_bidirectional = is_bidirectional  # True or False

        self.lstm = nn.LSTM(input_size=self.in_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.drop_prob,
                            batch_first=self.is_batch_first,
                            bidirectional=self.is_bidirectional)

        if use_cuda:
            self.lstm = self.lstm.cuda()

    def forward(self, input):
        if self.is_batch_first:  # True
            # if batch_first is true,
            # first dimension means batch size
            batch_size = input.size(0)
        else:  # False
            # if batch_first is false,
            # second dimension is batch size
            # (then, first dimension means sequence length)
            batch_size = input.size(1)

            # switch 1st dim and 2nd dim each other
            input = input.transpose(0, 1)

        # initialize hidden and cell state with zeros
        # [number of layers, batch size, hidden dimension]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()

        if use_cuda:
            h = h.cuda()
            c = c.cuda()

        outputs, (h, c) = self.lstm(input, (h, c))

        if self.is_batch_first:  # True
            outputs = outputs.transpose(0, 1)

        output = outputs[-1]  # final prediction of hidden layer
        return output
