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
    def __init__(self, conv_in_ch, conv_out_ch, out_dim, stride, padding, max_token_length):
        super(CNN, self).__init__()  # Always start inheriting torch.nn.Module

        self.conv_in_ch = conv_in_ch  # Conv2d의 input 채널
        self.conv_out_ch = conv_out_ch  # Conv2d의 output 채널
        self.stride = stride  # 1
        self.padding = padding  # 1

        self.filter_sizes = [2, 3, 4]
        self.mtl = max_token_length

        # fc layer
        # conv2d 채널수 * 필터 갯수 (3) * 임베딩 차원 (300)
        self.in_features = self.conv_out_ch * len(self.filter_sizes) * 300
        self.out_features = out_dim  # 최종 embedding 예측 -> 300 or 10
        self.fc = nn.Linear(self.in_features, self.out_features)

        self.tanh = nn.Tanh()  # range -1 ~ 1
        if use_cuda:
            self.fc = self.fc.cuda()

    def forward(self, x):
        # convolution -------------------------------------------------------
        conv_outs = []
        for fs in self.filter_sizes:
            self.kernel = fs
            self.mp_kernel = self.mtl - self.kernel + 1
            self.conv = nn.Sequential(
                nn.Conv2d(self.conv_in_ch, self.conv_out_ch, (self.kernel, 1), self.stride, self.padding, bias=True),
                nn.ReLU(True),
                nn.MaxPool2d((self.mp_kernel, 1))
            )
            if use_cuda:
                self.conv = self.conv.cuda()
            out = self.conv(x)
            conv_outs.append(out)
            # print("conv layer shape: ", out.size())
        # concatenate 3 conv_outputs ---------------------------------------
        out = torch.cat(conv_outs, dim=1)
        # print("concat shape: ", out.size())
        # flatten ----------------------------------------------------------
        out = torch.flatten(out, start_dim=1) # batch 제외 flatten
        # print("flatten shape: ", out.size())
        # linear fc layer --------------------------------------------------
        out = self.fc(out)
        # print("fc layer shape: ", out.size())
        # tanh -------------------------------------------------------------
        out = self.tanh(out)
        # print("tanh shape: ", out.size())

        return out
