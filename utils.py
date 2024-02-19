import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import math
from collections import Counter
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj

def PPMI(cache_name, edge_index, num_nodes, improved=False):
    try:
        with open ('tmp/'+cache_name+'.pkl', 'rb') as f:
            edge_index, edge_weight, norm = pickle.load(f)
    except: 
        path_len = 5
        num_path = 40
        adj_dict = {}

        def add_edge(a, b):
            if a in adj_dict:
                neighbors = adj_dict[a]
            else:
                neighbors = set()
                adj_dict[a] = neighbors
            if b not in neighbors:
                neighbors.add(b)

        cpu_device = torch.device("cpu")
        gpu_device = torch.device("cuda")
        for a, b in edge_index.t().detach().to(cpu_device).numpy():
            a = int(a)
            b = int(b)
            add_edge(a, b)
            add_edge(b, a)

        adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}

        def sample_neighbor(a):
            neighbors = adj_dict[a]
            random_index = np.random.randint(0, len(neighbors))
            return neighbors[random_index]


        # word_counter = Counter()
        walk_counters = {}

        def norm(counter):
            s = sum(counter.values())
            new_counter = Counter()
            for a, count in counter.items():
                new_counter[a] = counter[a] / s
            return new_counter

        for _ in tqdm(range(num_path)):
            for a in adj_dict:
                current_a = a
                current_path_len = np.random.randint(1, path_len + 1)
                for _ in range(current_path_len):
                    b = sample_neighbor(current_a)
                    if a in walk_counters:
                        walk_counter = walk_counters[a]
                    else:
                        walk_counter = Counter()
                        walk_counters[a] = walk_counter

                    walk_counter[b] += 1

                    current_a = b

        normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}

        prob_sums = Counter()

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                prob_sums[b] += prob

        ppmis = {}

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                ppmi = np.log(prob / prob_sums[b] * len(prob_sums) / path_len)
                ppmis[(a, b)] = ppmi

        new_edge_index = []
        edge_weight = []
        for (a, b), ppmi in ppmis.items():
            new_edge_index.append([a, b])
            edge_weight.append(ppmi)


        edge_index = torch.tensor(new_edge_index).t().to(gpu_device)
        edge_weight = torch.tensor(edge_weight).to(gpu_device)

        #print (edge_index,edge_weight)
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32)

        with open ('tmp/'+cache_name+'.pkl', 'wb') as f:
            pickle.dump((edge_index, edge_weight, norm),f)

    return edge_index, edge_weight, norm

def index2dense(edge_index, edge_weight, nnode=2708):
    indx = edge_index.cpu().numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]= edge_weight.detach().cpu().numpy()
    new_adj = torch.from_numpy(adj).float()
    return new_adj

def get_node_central_weight(target_data, edge_weight, seudo_label, topk, device, base_w = 0.5, scale_w = 1):
    #ppr_matrix = index2dense(target_data.edge_index, edge_weight, target_data.x.size(0))
    ppr_matrix = to_dense_adj(target_data.edge_index, edge_attr = edge_weight)[0]
    gpr_matrix = []
    for iter_c in range(target_data.num_classes):
        iter_gpr = torch.mean(ppr_matrix[seudo_label == iter_c],dim=0).squeeze()
        gpr_matrix.append(iter_gpr)

    gpr_matrix = torch.stack(gpr_matrix,dim=0).transpose(0,1)
    # N * C, N: number of nodes, C: number of classes
    
    target_size = target_data.size(0)
    
    gpr_sum = torch.sum(gpr_matrix,dim=1) # N * 1
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix # N * C
    rn_matrix = torch.mm(ppr_matrix, gpr_rn)
    
    label_matrix = F.one_hot(seudo_label, gpr_matrix.size(1)).float() 
    rn_matrix = torch.sum(gpr_matrix * label_matrix,dim=1) - torch.sum(rn_matrix * label_matrix,dim=1)/(gpr_matrix.size(1)-1.0) 
    
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=True)
    id2rank       = {sorted_totoro[i][0]:i for i in range(target_size)}
    totoro_rank   = np.array([id2rank[i] for i in range(target_size)])


    topk_indices = np.array([sorted_totoro[i][0] for i in range(topk)]).astype(int)
    totoro_rank = totoro_rank[topk_indices]
    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(target_size-1)))) for x in totoro_rank]
    
    topk_indices = torch.from_numpy(np.array(topk_indices)).type(torch.long).to(device)
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor).to(device)

    return rn_weight, topk_indices

def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor