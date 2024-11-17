import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import random
import os
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected
import geopandas as gpd
from shapely.geometry import LineString

def load_data(edge_df, node_df):
    graphs = {}
    node_ids_per_inter = {}
    
    # Create a mapping from traj_id to node features
    node_features_dict = node_df.set_index('traj_id')[['tortuosity','curvature','angle_change_sin','angle_change_cos','angle_start_sin',
                                                       'angle_start_cos','angle_end_sin','angle_end_cos','direct_sin','direct_cos']].T.to_dict('list')
    # For each inter_id, construct a graph
    for inter_id, sub_df in tqdm(edge_df.groupby('inter_id')):
        # Extract node IDs and generate a unique list of nodes
        node_ids = pd.concat([sub_df['traj1'], sub_df['traj2']]).unique()
        node_ids.sort()  # Ensure node IDs are sorted
        node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        node_ids_per_inter[inter_id] = node_ids  # Save node IDs for each inter

        # Create node features with dimension 10
        node_features = torch.tensor([node_features_dict[node_id] for node_id in node_ids], dtype=torch.float)

        # Extract edge information and weights
        edge_start = sub_df['traj1'].map(node_id_to_index).to_numpy()
        edge_end = sub_df['traj2'].map(node_id_to_index).to_numpy()
        edge_index = torch.tensor([edge_start, edge_end], dtype=torch.long)

        # Compute edge weight features with dimension 11
        edge_weights = sub_df[['hsdf_dis','dis0','dis1','dis2','dis3','dis4','dis5','dis6','dis7','dis8','dis9']].to_numpy()
        # edge_weights = sub_df[['hsdf_dis','chamfer_dis' ,'extreme_dis','dis0','dis9']].to_numpy()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        # Create a Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        graphs[inter_id] = data  # Save the graph for each inter

    graphs = list(graphs.values())
    return graphs, node_ids_per_inter
def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # CUDA
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  
def exp_norm_replace(df, columns, beta):
    for column in columns:
        df[column] = np.exp(-df[column] / beta)
    return df

from torch_geometric.nn import knn_graph

import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout, MultiheadAttention
from torch_geometric.nn import TransformerConv, GraphSizeNorm

class TransformerGNNNet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes, num_heads=4, dropout=0.2):
        super(TransformerGNNNet, self).__init__()

        
        # Node feature MLP layers
        self.fc00 = Linear(num_node_features, 64)

        
        # Edge feature FFN
        self.edge_fc1 = Linear(num_edge_features, 64)
        self.edge_fc2 = Linear(64, 64)
        self.edge_fc3 = Linear(64, 64)

        
        self.conv1 = TransformerConv(64, 64, heads=num_heads, edge_dim=64, dropout=dropout)
        self.ln1 = LayerNorm(64 * num_heads) 
        
        self.dropout = Dropout(0.5)
        self.fc1 = Linear(64 * num_heads, 128)
        self.fc2 = Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # node feature learning layer
        x1 = self.fc00(x)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        
        # edge feature FFN
        edge_attr = self.edge_fc1(edge_attr)
        edge_attr = F.relu(edge_attr)
        edge_attr = self.dropout(edge_attr)
        edge_attr = self.edge_fc2(edge_attr)
        edge_attr = F.relu(edge_attr)
        edge_attr = self.dropout(edge_attr)
        edge_attr = self.edge_fc3(edge_attr)
        edge_attr = F.relu(edge_attr)
        edge_attr = self.dropout(edge_attr)

        # First TransformerConv
        x1 = self.conv1(x1, edge_index, edge_attr=edge_attr)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)

        # fully connected layer
        x = self.fc1(x1)
        x = F.relu(x)
        x = self.dropout(x)
        
        # classification
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
    
from scipy.optimize import linear_sum_assignment

def extract_top_positive_samples(output, top_k=50):
    positive_log_probabilities = output[:, 1]
    top_log_probabilities, top_indices = torch.topk(positive_log_probabilities, k=top_k, largest=True)

    return top_indices, top_log_probabilities

def find_optimal_matching(output, D): 
    output = output.detach().numpy()
    updated_D = D + (-output[:, 1].reshape(-1, 1))
    row_indices, col_indices = linear_sum_assignment(updated_D)
    matched_costs = updated_D[row_indices, col_indices]
    return row_indices, matched_costs
    





from tqdm import tqdm
class HungarianMatcher(torch.nn.Module):
    def forward(self, output, D):
        if output.shape[0] != D.shape[0]:
            raise ValueError("output and D must have the same number of rows")
        with torch.no_grad():

            probabilities = torch.exp(output[:, 1])

            updated_D = D - probabilities.unsqueeze(1)
            row_indices, col_indices = linear_sum_assignment(updated_D.cpu().numpy())
        row_indices = torch.as_tensor(row_indices, dtype=torch.long)
        col_indices = torch.as_tensor(col_indices, dtype=torch.long)

        return row_indices, col_indices



def normalize_D(D_list,device):
    D_norm_list=[]
    for D in D_list:
        for i in range(D.shape[0]):
            min_val = np.min(D[i])
            D[i][D[i] != min_val] = np.nan
        
        for j in range(D.shape[1]):
            col = D[:, j]
            valid_values = col[~np.isnan(col)]
            if valid_values.size > 0:
                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                if max_val != min_val:
                    D[:, j][~np.isnan(col)] = (col[~np.isnan(col)] - min_val)*0.1 / (max_val - min_val)
                else:
                    D[:, j][~np.isnan(col)] = 0
        
        D[np.isnan(D)] = 1
        D = torch.tensor(D, dtype=torch.float32)
        D=D.to(device)
        D_norm_list.append(D)
    return D_norm_list
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def get_loss_batch(matcher, batch,outputs, D_hsdf_list):
    total_loss = 0.0
    # batch_size = outputs.shape[0]
    # print(batch_size)
    batch_index = batch.batch 
    # print(len(outputs))
    # print(len(D_hsdf_list))
    for idx in range(len(D_hsdf_list)):
        # output = outputs[idx]
        D_hsdf = D_hsdf_list[idx]
        mask = (batch_index == idx)
        output = outputs[mask]
        # print(idx,output.shape[0], D_hsdf.shape[0])
        matched_row_indices, matched_col_indices = matcher(output, D_hsdf)
        
        updated_D = D_hsdf
    

        matched_costs = updated_D[matched_row_indices, matched_col_indices]

        sum_matched = 0 
        for i in range(output.shape[0]):
            if i not in matched_row_indices:
                sum_matched += output[i, 0]  
        sum_unmatched = 0 
        for i in range(output.shape[0]):
            if i in matched_row_indices:
                sum_unmatched += output[i, 1] 

        current_loss = (-sum_unmatched/10 - sum_matched) + matched_costs.sum() 
        total_loss += current_loss
    batch_loss = total_loss / len(D_hsdf_list)
    return batch_loss