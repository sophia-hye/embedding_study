import torch
import torch.nn as nn
import torch.optim as optim


# construct model on cuda if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class CNN(nn.Module):
    # @in_dim: dimension of inputted data
    # @out_dim: dimension of output data which user want to get
    # @nf: number of filters
    def __init__(self, in_dim, out_dim, nf, kernel, stride, padding):
        super(CNN, self).__init__()  # Always start inheriting torch.nn.Module

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nf = nf
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.net = nn.Sequential(
            nn.Conv1d(self.in_dim, self.nf * 1, self.kernel, self.stride, self.padding, bias=False),
            nn.AvgPool1d(2),
            nn.ReLU(True),
            nn.Conv1d(self.nf * 1, self.nf * 2, self.kernel, self.stride, self.padding, bias=False),
            nn.AvgPool1d(2),
            nn.ReLU(True),
            nn.Conv1d(self.nf * 2, self.nf * 4, self.kernel, self.stride, self.padding, bias=False),
            nn.AvgPool1d(2),
            nn.ReLU(True),
            nn.Conv1d(self.nf * 4, self.nf * 8, self.kernel, self.stride, self.padding, bias=False),
            nn.AvgPool1d(2),
            nn.ReLU(True),
            nn.Conv1d(self.nf * 8, self.nf * 6, self.kernel, self.stride, self.padding, bias=False),
            nn.AvgPool1d(2),
            nn.ReLU(True),
            nn.Conv1d(self.nf * 6, self.nf * 4, self.kernel, self.stride, self.padding, bias=False),
            nn.AvgPool1d(2),
            nn.ReLU(True),
            nn.Conv1d(self.nf * 4, self.nf * 2, self.kernel, self.stride, self.padding, bias=False),
            nn.AvgPool1d(2),
            nn.ReLU(True),
            nn.Conv1d(self.nf * 2, self.out_dim, self.kernel, self.stride, self.padding, bias=False),
            nn.AvgPool1d(2),
            #nn.AvgPool1d((1, 2)),
            nn.Tanh()  # range -1 ~ 1

        )

        if use_cuda:
            self.net = self.net.cuda()

    def forward(self, x):
        y_pred = self.net(x)
        y_pred = torch.squeeze(y_pred, 1)  # [batch size, embedding size]
        return y_pred