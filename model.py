import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

class GCN(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(GCN,self).__init__()
        c_in = (order*support_len+1)*c_in
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = torch.einsum('ncvl,vw->ncwl',(x,a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,vw->ncwl',(x1,a)).contiguous()
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class TE(nn.Module):
    def __init__(self, in_dim):
        super(TE, self).__init__()
        self.te_conv = nn.Conv2d(in_dim-1, 32,kernel_size=1)

    def forward(self,x):
        time = x[:,1:,:,:]
        tem_embedding = self.te_conv(time)
        return tem_embedding


class TATT(nn.Module):
    def __init__(self, device, c_in, c_out, K, d, mask = True):
        super(TATT, self).__init__()
        # K: number of attention heads
        # d: dimension of attention outputs
        # refer to GMAN
        self.d = d
        self.K = K
        self.mask = mask
        self.conv_q = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv_k = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv_v = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv_o = nn.Conv2d(c_out, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.small_value = torch.tensor(-2**15+1.).to(torch.device(device))
        self.device = device

    def forward(self, x, tem_embedding):
        # input format: [batch, channels, num_nodes, num_time_steps]
        # tem_embedding format: [batch, tem_em_channels, num_nodes, num_time_steps]
        batch_size_ = x.shape[0]
        x = torch.cat((x, tem_embedding[:,:,:,-x.shape[3]:]), dim=1)
        query = self.conv_q(x)
        key = self.conv_k(x)
        value = self.conv_v(x)
        query = torch.cat(torch.split(query, self.d, dim=1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=1), dim=0)
        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 3, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        if self.mask:
            batch_size = x.shape[0]
            num_step = x.shape[3]
            num_vertex = x.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            mask = mask.to(torch.device(self.device))
            attention = torch.where(mask, attention, self.small_value)
        # softmax
        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 3, 1, 2)
        x = torch.cat(torch.split(x, batch_size_, dim=0), dim=1)
        x = F.relu(self.conv_o(x))
        del query, key, value, attention
        return x

class STlayer(nn.Module):
    def __init__(self,device, residual_channels, dilation_channels, dropout, supports, kt, kd, nodevec1, nodevec2):
        super(STlayer, self).__init__()
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        self.supports_len +=1
        self.TCN = nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kt), padding=(0, 0), stride=(1, 1), bias=True,dilation=(1, kd))
        self.TATT = TATT(device, 64, 32, 8, 4)
        self.GCN = GCN(dilation_channels,residual_channels,dropout,support_len=self.supports_len)
        self.bn = nn.BatchNorm2d(residual_channels)


    def forward(self, x, tem_embedding):
        residual = x
        # dilated TCN
        x = self.TCN(x)
        x = F.relu(x)
        # temporal_attention
        x = self.TATT(x, tem_embedding)
        #GCN
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.supports + [adp]
        x = self.GCN(x, new_supports)
        x = x + residual[:, :, :, -x.size(3):]
        x = self.bn(x)
        return x

class net2(nn.Module):
    def __init__(self, device, num_nodes, tcn_k, tcn_d, dropout=0.3, supports=None, in_dim=2, out_dim=12, residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512):
        super(net2, self).__init__()
        self.supports = supports
        self.layers = nn.ModuleList()
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        for d in range(tcn_d, tcn_d+3):
            self.layers.append(
                STlayer(device, residual_channels, dilation_channels, dropout, supports, tcn_k, 1, self.nodevec1, self.nodevec2))
            self.layers.append(
                STlayer(device, residual_channels, dilation_channels, dropout, supports, tcn_k, d, self.nodevec1, self.nodevec2))
        self.skip_conv_end = nn.Conv2d(residual_channels, skip_channels,kernel_size=(1, 1))
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1,1))
        self.end_conv_0 = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1,1), bias=True)
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1,1), bias=True)
        self.TE = TE(in_dim)

    def forward(self, input):
        tem_embedding = self.TE(input)
        if input.size(3) < 13:
            x = nn.functional.pad(input,(13-input.size(3),0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        i = 0
        for layer in self.layers:
            x = layer(x, tem_embedding)
            if i == 0:
                skip = x
            else:
                skip = skip[:, :, :, -x.size(3):] + x
            i += 1

        x = F.relu(skip)
        x = F.relu(self.end_conv_0(x))
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

