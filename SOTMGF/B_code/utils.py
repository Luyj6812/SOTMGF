import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import pandas as pd
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import random

def load_graph1(X, adata):
    data = X
    #data = np.around(data,6)

    edge_data = np.array(adata.uns['KNN_df'])
    n, _ = data.shape
    edges_unordered = edge_data[:, 0:2]
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.ravel()))).reshape(edges_unordered.shape)
    x1 = np.array( edge_data[:, 2] )
    x2 = (edges[:, 0], edges[:, 1].astype(int))
    adj = sp.coo_matrix((x1, x2), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    #adj = normalize(adj)
    adj = adj.todense()
    
    return adj

def load_graph(adata):
    # seed_value = 1
    # random.seed(seed_value)  # 设置 random 模块的随机种子
    # np.random.seed(seed_value)  # 设置 numpy 模块的随机种子
    # torch.manual_seed(seed_value)  # 设置 PyTorch 中 CPU 的随机种子
    #data = np.loadtxt('data/DAGC/{}.txt'.format(dataset))
    data = adata.obsm['h_tem'] 
    data = np.array(data)
    data = np.around(data,6)

    ##########adj1#############
    #load_path = "./data/DAGC/bian_yuan0818.csv"
    #edge_data = pd.read_csv(load_path, header=0, index_col=None)
    #edge_data = np.array(edge_data)
    edge_data1 = np.array(adata.uns['KNN_df'])
    n, _ = data.shape
    edges_unordered = edge_data1[:, 0:2]
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.ravel()))).reshape(edges_unordered.shape)
    x1 = np.array( edge_data1[:, 2] )
    x2 = (edges[:, 0], edges[:, 1].astype(int))
    adj1 = sp.coo_matrix((x1, x2), shape=(n, n))
    adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    adj1 = adj1 + sp.eye(adj1.shape[0])
    adj1 = normalize(adj1)
    ###########adj2################
    #load_path = "F:/Lu/kongzhuan2/code_pre/DAGC_data/pruneG_151673_scc_nb3.csv"
    #load_path = "./data/DAGC/bian_prune0809.csv"
    #edge_data = pd.read_csv(load_path, header=0, index_col=None)
    #edge_data = np.array(edge_data)
    edge_data2 = np.array(adata.uns['prune_KNN_df'])
    edges_unordered = edge_data2[:, 0:2]
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.ravel()))).reshape(edges_unordered.shape)
    x1 = np.array( edge_data2[:, 2] )
    x2 = (edges[:, 0], edges[:, 1].astype(int))
    adj2 = sp.coo_matrix((x1, x2), shape=(n, n))
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
    adj2 = adj2 + sp.eye(adj2.shape[0])
    adj2 = normalize(adj2)
    ###########adj3################
    #load_path = "./data/DAGC/edge3_MEG_distances.csv"
    #edge_data = pd.read_csv(load_path, header=None, index_col=None)
    #adj3= np.array(edge_data)
    adj3 = np.array(adata.obsp['ME_EMD_mat'])
    adj3 = scipy.sparse.csr_matrix(adj3)
    adj3 = adj3 + adj3.T.multiply(adj3.T > adj3) - adj3.multiply(adj3.T > adj3)
    adj3 = adj3 + sp.eye(adj3.shape[0])
    adj3 = normalize(adj3)

    ###########adj4################
    # load_path = "./data/murine_breast_cancer/cndm_murine.csv"
    # load_path = "./data/Mouse_brain/cndm_mouse_brain.csv"
    # load_path = "./data/IDC/cndm_IDC.csv"
    # load_path = "./stMVC_test_data/CCSN/cndm_151673.csv"
    load_path = "./SOTMGF_data/CCSN/cndm_151507.csv"
    # load_path = "./stMVC_test_data/CCSN/cndm_151671.csv"
    # load_path = "./data/murine_breast_cancer/cndm_murine.csv"
    edge_data4 = pd.read_csv(load_path, header=None, index_col=None)
    adj4= np.array(edge_data4)
    adj4 = cosine_similarity(adj4)
    adj4 = scipy.sparse.csr_matrix(adj4)
    adj4 = adj4 + adj4.T.multiply(adj4.T > adj4) - adj4.multiply(adj4.T > adj4)
    adj4 = adj4 + sp.eye(adj4.shape[0])
    adj4 = normalize(adj4)

    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
    adj3 = sparse_mx_to_torch_sparse_tensor(adj3)
    adj4 = sparse_mx_to_torch_sparse_tensor(adj4)
    return adj1, adj2, adj3, adj4

def adj_norm(adj):
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj
    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)
    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    return norm_adj

def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class load_data(Dataset):
    def __init__(self, adata):
            #self.x = np.loadtxt('./data/DAGC/{}.txt'.format(dataset), dtype=float)
            #self.y = np.loadtxt('./data/DAGC/{}_label.txt'.format(dataset), dtype=int)
            self.x = np.array(adata.obsm['h_tem'])
            self.y = np.array(adata.obs['label'])
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))
