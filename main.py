# coding=utf-8
import os

from sklearn.metrics import f1_score
from argparse import ArgumentParser
from gnn.dataset.DomainData import DomainData
from utils import *
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

def test(data, domain = 'source',mask=None, reparam=False):
    for model in models:
        model.eval()
    if domain == 'source':
        new_x = mask_feature(data.x, 'source', reparam)
        new_edge_weight = drop_edge(drop_edge.s_edge_prob, device, reparam)
    elif domain == 'target':
        new_x = mask_feature(data.x, 'target', reparam)
        new_edge_weight = drop_edge(drop_edge.t_edge_prob, device, reparam)

    encoded_output = encoder(new_x, data.edge_index, new_edge_weight)

    (_, _), encoded_output = gib_layer(encoded_output, reparam)

    if mask is not None:
        encoded_output = encoded_output[mask]
    logits = cls_model(encoded_output)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    macro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')
    micro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='micro')
    return accuracy, macro_f1, micro_f1

def Semi_loss(logits, seudo_label, rn_weight=None):
    softmax_out = nn.Softmax(dim=1)(logits)
    entropy = -seudo_label * torch.log(softmax_out + 1e-5)
    if rn_weight is not None:
        entropy =rn_weight * torch.sum(entropy, dim=1)
    else:
        entropy =torch.sum(entropy, dim=1)
    entropy = torch.mean(entropy)

    msoftmax = softmax_out.mean(dim=0)
    entropy -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

    return entropy 

def Entropy(softmax_out):
    entropy = -softmax_out * torch.log(softmax_out + 1e-5)
    entropy = torch.mean(torch.sum(entropy, dim=1))

    msoftmax = softmax_out.mean(dim=0)
    entropy -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    return entropy 


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

def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, 0.05)
    
    new_s_x = mask_feature(source_data.x, 'source')
    new_s_edge_weight = drop_edge(drop_edge.s_edge_prob, device)

    new_t_x = mask_feature(target_data.x, 'target')
    new_t_edge_weight = drop_edge(drop_edge.t_edge_prob, device)

    encoded_source = encoder(new_s_x, source_data.edge_index, new_s_edge_weight)
    encoded_target = encoder(new_t_x, target_data.edge_index, new_t_edge_weight)


    (s_mu, s_std), encoded_source = gib_layer(encoded_source)
    (t_mu, t_std), encoded_target = gib_layer(encoded_target)

    source_logits = cls_model(encoded_source)
    target_logits = cls_model(encoded_target)

    # classifier loss:
    cls_loss = loss_func(source_logits[label_mask], source_data.y[label_mask])

    # DA loss
    source_domain_preds = domain_model(encoded_source)
    target_domain_preds = domain_model(encoded_target)

    source_domain_cls_loss = loss_func(
        source_domain_preds,
        torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
    )
    target_domain_cls_loss = loss_func(
        target_domain_preds,
        torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
    )
    loss_grl = source_domain_cls_loss + target_domain_cls_loss


    # IB loss
    info_loss = -0.5*(1+2*s_std.log()-s_mu.pow(2)-s_std.pow(2)).sum(1).mean().div(math.log(2))
    info_loss += -0.5*(1+2*t_std.log()-t_mu.pow(2)-t_std.pow(2)).sum(1).mean().div(math.log(2))


    # mixup loss
    
    # calculate weight
    source_softmax_out = nn.Softmax(dim=1)(source_logits)
    target_softmax_out = nn.Softmax(dim=1)(target_logits)

    s_seudo_label = source_softmax_out.argmax(dim=1)
    t_seudo_label = target_softmax_out.argmax(dim=1)

    s_seudo_label[label_mask] = source_data.y[label_mask]

    
    mu = 0.5 - math.cos(min(math.pi,(2*math.pi*float(epoch) / epochs)))/2
    ks = int(source_data.y[label_mask].size(0)*mu)*3
    kt = int(source_data.y[label_mask].size(0)*mu)*3


    s_rn_weight, s_indices = get_node_central_weight(source_data, new_s_edge_weight, s_seudo_label, ks, device)
    t_rn_weight, t_indices = get_node_central_weight(target_data, new_t_edge_weight, t_seudo_label, kt, device)
    

    ##  inner
    source_softmax_out[label_mask] = F.one_hot(source_data.y,source_softmax_out.size(1)).float()[label_mask]

    inner_index = torch.randint(low=0, high=source_logits[label_mask].size(0), size=(ks,)).to(device)
    inner_labeled = torch.index_select(source_data.y[label_mask], 0, inner_index)
    inner_labeled = F.one_hot(inner_labeled,source_softmax_out.size(1))
    inner_encoded = torch.index_select(encoded_source[label_mask], 0, inner_index)
    
    beta = 1
    alpha = torch.distributions.Beta(beta, beta).sample((ks,)).unsqueeze(1).to(device)

    uns_encoded_source = alpha * encoded_source[s_indices] + (1-alpha) * inner_encoded
    yhat_source = alpha * source_softmax_out[s_indices] + (1-alpha) * inner_labeled

    un_source_logits = cls_model(uns_encoded_source)

    semi_loss = Semi_loss(un_source_logits, yhat_source, s_rn_weight)

    ##  outter 
    outer_index = torch.randint(low=0, high=source_logits[label_mask].size(0), size=(kt,)).to(device)
    outer_labeled = torch.index_select(source_data.y[label_mask], 0, outer_index)
    outer_labeled = F.one_hot(outer_labeled,source_softmax_out.size(1))
    outer_encoded = torch.index_select(encoded_source[label_mask], 0, outer_index)
    
    beta = 1
    alpha = torch.distributions.Beta(beta, beta).sample((kt,)).unsqueeze(1).to(device)

    uns_encoded_target = alpha * encoded_target[t_indices]+ (1-alpha) * outer_encoded
    yhat_target = alpha * target_softmax_out[t_indices] + (1-alpha) * outer_labeled

    un_target_logits = cls_model(uns_encoded_target)

    semi_loss = Semi_loss(un_target_logits, yhat_target, t_rn_weight)

    

    loss = cls_loss + loss_grl + 1e-3*info_loss + 5*semi_loss


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    return (new_s_edge_weight, new_t_edge_weight, new_s_x, new_t_x, encoded_source, encoded_target)
    

# Set Hyperparameters
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='dblpv7')
parser.add_argument("--target", type=str, default='citationv1')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--drop_out", type=float, default=1e-1)

parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--IB_dim", type=int, default=256)
parser.add_argument("--label_rate", type=float, default=0.05)

args = parser.parse_args()
seed = args.seed
encoder_dim = args.encoder_dim
IB_dim = args.IB_dim
label_rate = args.label_rate

id = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, dim: {}" \
    .format(args.source, args.target, seed, label_rate, args.learning_rate, args.weight_decay, 
            encoder_dim)
print(id)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False



# Load data
dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
source_data.num_classes = dataset.num_classes
print(source_data)


dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
target_data.num_classes = dataset.num_classes
print(target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)

source_train_size = int(source_data.size(0) * label_rate)
label_mask = np.array([1] * source_train_size + [0] * (source_data.size(0) - source_train_size)).astype(bool)
np.random.shuffle(label_mask)
label_mask = torch.tensor(label_mask).to(device)


# Construct network

mask_feature = MaskFeature(source_data.x.size(1), device).to(device)
drop_edge = DropEdge(source_data.edge_index.size(1),target_data.edge_index.size(1), device).to(device)

encoder = GNN(source_data.x.size(1), encoder_dim, encoder_dim, gnn_type ='gcn').to(device)

gib_layer = GIB(encoder_dim, IB_dim).to(device)

cls_model = nn.Sequential(
    nn.Linear(IB_dim, dataset.num_classes),
).to(device)

domain_model = nn.Sequential(
    GRL(),
    nn.Linear(IB_dim, 64),
    nn.ReLU(),
    nn.Dropout(args.drop_out),
    nn.Linear(64, 2),
).to(device)

loss_func = nn.CrossEntropyLoss().to(device)

models = [mask_feature, drop_edge, encoder, gib_layer, cls_model, domain_model]
params = itertools.chain(*[model.parameters() for model in models])


optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)




# Train
best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
best_macro_f1 = 0.0
epochs = 400
for epoch in range(1, epochs):
    pair_data = train(epoch)
    source_correct, _, _ = test(source_data, 'source', source_data.test_mask)
    target_correct, macro_f1, micro_f1 = test(target_data, 'target')
    print("Epoch: {}, source_acc: {}, target_acc: {}, macro_f1: {}, micro_f1: {}".format(epoch, source_correct,
                                                                                         target_correct, macro_f1,
                                                                                         micro_f1))
    if source_correct > best_source_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_macro_f1 = macro_f1
        best_micro_f1 = micro_f1
        best_epoch = epoch
        torch.save(pair_data, "tmp/{}-{}.pt".format(args.source, args.target))
print("=============================================================")
line = "{}\n - Epoch: {}, best_source_acc: {}, best_target_acc: {}, best_macro_f1: {}, best_micro_f1: {}" \
    .format(id, best_epoch, best_source_acc, best_target_acc, best_macro_f1, best_micro_f1)

print(line)


with open("log/{}-{}.log".format(args.source, args.target), 'a') as f:
    line = "{} - Epoch: {:0>3d}, best_macro_f1: {:.5f}, best_micro_f1: {:.5f}\t" \
               .format(id, best_epoch, best_macro_f1, best_micro_f1) + time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n"
    f.write(line)
