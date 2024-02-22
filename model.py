
import os

from sklearn.metrics import f1_score
from argparse import ArgumentParser
from gnn.dataset.DomainData import DomainData
from utils import *
from model import *
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
import time
import warnings
import math
warnings.filterwarnings("ignore", category=UserWarning)
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from numbers import Number

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_type="gcn"):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        if gnn_type == 'gcn':
            self.conv_layers = nn.ModuleList([
                GCNConv(self.in_channels, self.hidden_channels),
                GCNConv(self.hidden_channels, self.out_channels) 
            ])
        elif gnn_type == 'gat':
            self.conv_layers = nn.ModuleList([
                GATConv(self.in_channels, self.hidden_channels),
                GATConv(self.hidden_channels, self.out_channels) 
            ])
        elif gnn_type == 'gsage':
            self.conv_layers = nn.ModuleList([
                SAGEConv(self.in_channels, self.hidden_channels),
                SAGEConv(self.hidden_channels, self.out_channels) 
            ])
        elif gnn_type == 'gin':
            self.conv_layers = nn.ModuleList([
                GINConv(self.in_channels, self.hidden_channels),
                GINConv(self.hidden_channels, self.out_channels) 
            ])

        #self.prelu = nn.PReLU(self.hidden_channels)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, edge_index, edge_weight):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, edge_weight)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

class MaskFeature(nn.Module):
    def __init__(self, feat_dim, device):
        super(MaskFeature, self).__init__()
        self.s_mask_x = nn.Parameter(self.construct_feat_mask(feat_dim))
        self.t_mask_x = nn.Parameter(self.construct_feat_mask(feat_dim))
    
    def forward(self, x, domain, use_sigmoid=True, reparam=True):
        mask = self.s_mask_x if domain =='source' else self.t_mask_x
        mask = torch.sigmoid(mask) if use_sigmoid else mask
        if reparam:
            std_tensor = torch.ones_like(x, dtype=torch.float) / 2.0
            mean_tensor = torch.zeros_like(x, dtype=torch.float) -x
            z = torch.normal(mean=mean_tensor, std=std_tensor).to(x.device)
            x = x * mask + z * (1 - mask)
        else:
            x = x * mask
        return x
    
    def construct_feat_mask(self, feat_dim, init_strategy="ones"):
        mask = torch.ones(feat_dim)
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

class DropEdge(nn.Module):
    def __init__(self, source_edge_num, target_edge_num, device):
        super(DropEdge, self).__init__()
        self.s_edge_prob = self.construct_edge_prob(source_edge_num)
        self.t_edge_prob = self.construct_edge_prob(target_edge_num)

    def forward(self, prob, device, reparam=True):
        temperature = 1
        if reparam:
            eps = torch.rand(prob.size())
            gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(device)
            gate_inputs = (prob + gate_inputs) / temperature
            edge_weight = torch.sigmoid(gate_inputs)
        else:
            edge_weight = torch.sigmoid(prob)
        return edge_weight
   
    def construct_edge_prob(self, edge_num, init_strategy="ones"):
        prob = nn.Parameter(torch.ones(edge_num)*100)  #make initial weight close to 1
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                prob.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(prob, 0.0)
        return prob



class GIB(nn.Module):
    def __init__(self, hidden_dim, IB_dim):
        super(GIB, self).__init__()

        self.IB_dim = IB_dim
    
    def forward(self, encoded_output, reparam=True, num_sample=1):
        mu = encoded_output[:,:IB_dim]
        std = F.softplus(encoded_output[:, IB_dim:IB_dim*2]-5, beta=1)

        if reparam:
            encoding = self.reparametrize_n(mu, std, num_sample)
        else:
            encoding = mu

        return (mu, std), encoding

    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps =torch.Tensor(std.size()).normal_().to(std.device)

        return mu + eps * std
