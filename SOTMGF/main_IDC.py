import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:960"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ['CUDA_VISIBLE_DEVICES']='0'
import pandas as pd
from utils import pca_spateo, scc
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import scanpy
#from precluster_split import precluster
from precluster_1221 import precluster
import scanpy as sc
import numpy as np
from MEG import MEG_construction
from prune import graph_prune
import argparse
import torch
from Train_MDAGC_eachview import train_MDGAC
#from train_MDAGC_eachview_0105 import train_MDGAC
from graph_1221 import precluster_graph
from torch.optim import Adam
import random
import scvelo as scv
torch.cuda.empty_cache()


# seed_value = 1
# random.seed(seed_value)  # 设置 random 模块的随机种子
# np.random.seed(seed_value)  # 设置 numpy 模块的随机种子
# torch.manual_seed(seed_value)  # 设置 PyTorch 中 CPU 的随机种子
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.is_available():  # 如果可以使用 CUDA，设置随机种子
#     torch.cuda.manual_seed_all(seed_value) 
#     torch.cuda.manual_seed(seed_value)  # 设置 PyTorch 中 GPU 的随机种子
#     torch.backends.cudnn.deterministic = True  # 使用确定性算法，使每次运行结果一样
#     torch.backends.cudnn.benchmark = False  # 不使用自动寻找最优算法加速运算  
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--name', type=str, default='151673')
parser.add_argument('--k', type=int, default=3)
#parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_clusters', default=16, type=int)
parser.add_argument('--n_feature', default=500, type=int)
parser.add_argument('--n_z', default=10, type=int)##feature extracted with AE
#parser.add_argument('--pretrain_path', type=str, default='pkl')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")
#args.pretrain_path = 'data/{}.pkl'.format(args.name)

####load data
label = pd.read_csv('./data/IDC/label_IDC.csv',header=0, index_col=0, encoding="gbk")
label = np.array(label).squeeze(1)
adata = scanpy.read_visium('./data/IDC/data_IDC', genome=None, count_file='filtered_feature_bc_matrix.h5')
# adata.var_names_make_unique()




##process the data
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

##process the data with scanpy routine
sc.pp.pca(adata,n_comps=100)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata.obsm['spatial'] = adata.obsm['spatial'].astype(int)
print(adata)
## true label

###scc cluster
adata = adata[:, adata.var['highly_variable']]
gene2000=adata.to_df()#提取2000维高变基因基因表达信息
gene2000 = np.array(gene2000) 
adata.obsm['X_tem'] = gene2000
adata.obs['label'] = np.array(label)
proportion_CARD = pd.read_csv('./data/IDC/Proportion_CARD_IDC_100064.csv',header=0, index_col=0, encoding="gbk")
adata.obsm['proportion_CARD'] = np.array(proportion_CARD)
#adata.obs['label'] = adata.obs['label'].astype('category')
# pca_spateo(adata, n_pca_components=50)
# scc(adata, s_neigh=10, resolution=1.1, cluster_method="louvain", key_added="label_tem", pca_key="X_pca")#adata.obs['label_step1']

# h_stagate = pd.read_csv('/media/dell/新加卷/Lu/code_kongzhuan2/codepre/STAGATE-main/output/h_stagate_151673.csv', header=None, index_col=None)
# label_stagate = pd.read_csv('/media/dell/新加卷/Lu/code_kongzhuan2/codepre/STAGATE-main/output/label_stagate_151673.csv', header=None, index_col=None)
# adata.obs['label_tem'] = np.array(label_stagate)
# adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')


precluster_graph(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 600, rad_cutoff = 280, lr=0.00001)
sc.pl.spatial(adata,color=['label_tem','label'])
#adata.obs['label_tem'] = label_tem
print(adata.obs['label_tem'])
#adata.obs['label_tem'] = adata.obs['label']


##precluster with transformer

# iter = 0
# for iter in range(0, 2):


#     print("######################################iter:",iter,"######################################")
#     print(adata.obsm['X_tem'].shape)

#     precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 1500)
#     sc.pl.spatial(adata,color=['label_tem','label'])

# #     print('label_precluster:', adata.obs['label_tem'])

# ###MEG_construction###
#     MEG_construction(adata, key_label='label_tem')
#     #print(adata)

# ###graph_construction and prune_graph

#     graph_prune(adata)

# ###train_MDAGC
#     train_MDGAC(args, adata, n_clusters=args.n_clusters, n_feature=args.n_feature, n_epoch =28, n_z=args.n_z)
#     #adata.obs['label_tem'] = adata.obs['label_MDAGC']
   
#     print(adata.obs['label_tem'])

#     #adata.obs['label_tem'] = pd.Series(adata.obs['label_tem'])
#     sc.pl.spatial(adata,color=['label_tem','label','label_fusion'])
#     #adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
    
#     torch.cuda.empty_cache()
#     iter += 1

# adata.obs['label_pred'] = pd.DataFrame(adata.obs['label_tem'])
# adata.obs['label_pred'] = adata.obs['label_tem'].astype('category')
# adata.obsm['h_fusion'] = adata.obsm['h_tem']
#adata.obsm['X_bar'] = adata.obsm['X_tem']
#sc.pl.spatial(adata,color=['label_pred'])

print("######################################iter:1######################################")
precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 1500, lr=0.000001, n_head = 4)
# sc.pl.spatial(adata,color=['label_tem','label'])

#     print('label_precluster:', adata.obs['label_tem'])

###MEG_construction###
MEG_construction(adata, key_label='label_tem')
    #print(adata)

###graph_construction and prune_graph

graph_prune(adata,rad_cutoff = 280)

###train_MDAGC
train_MDGAC(args, adata, n_feature=args.n_feature, n_epoch = 200, n_z=args.n_z)
    #adata.obs['label_tem'] = adata.obs['label_MDAGC']
   
print(adata.obs['label_tem'])

    #adata.obs['label_tem'] = pd.Series(adata.obs['label_tem'])
# sc.pl.spatial(adata,color=['label_tem','label_1','label','label_fusion'])
    #adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')

torch.cuda.empty_cache()


##############################################################################################################
print("######################################iter:2######################################")
precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 1500, lr=0.000001, n_head = 4)
#sc.pl.spatial(adata,color=['label_tem','label'])

#     print('label_precluster:', adata.obs['label_tem'])

###MEG_construction###
MEG_construction(adata, key_label='label_tem')
    #print(adata)

###graph_construction and prune_graph

graph_prune(adata,rad_cutoff = 280)

###train_MDAGC
train_MDGAC(args, adata, n_feature=args.n_feature, n_epoch = 300, n_z=args.n_z)
    #adata.obs['label_tem'] = adata.obs['label_MDAGC']
   
print(adata.obs['label_tem'])

    #adata.obs['label_tem'] = pd.Series(adata.obs['label_tem'])
# sc.pl.spatial(adata,color=['label_tem','label','label_fusion'])
    #adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
#sc.pl.spatial(adata,color=['label_fusion','label'])   
torch.cuda.empty_cache()

##############################################################################################################
print("######################################iter:3######################################")
precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 1500, lr=0.000001, n_head = 4)
#sc.pl.spatial(adata,color=['label_tem','label'])

#     print('label_precluster:', adata.obs['label_tem'])

###MEG_construction###
MEG_construction(adata, key_label='label_tem')
    #print(adata)

###graph_construction and prune_graph

graph_prune(adata,rad_cutoff = 280)

###train_MDAGC
train_MDGAC(args, adata, n_feature=args.n_feature, n_epoch = 300, n_z=args.n_z)
    #adata.obs['label_tem'] = adata.obs['label_MDAGC']
   
print(adata.obs['label_tem'])
adata.obs['label'] = adata.obs['label'].astype('category')
    #adata.obs['label_tem'] = pd.Series(adata.obs['label_tem'])
sc.pl.spatial(adata,color=['label_tem','label','label_fusion'],save = "region_IDC.pdf")
sc.pl.spatial(adata,color=['label_fusion','label'],save = "region_IDC.pdf")
    #adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
#sc.pl.spatial(adata,color=['label_fusion','label'])   
torch.cuda.empty_cache()


########################################RNA velocities#######################################################
np.savetxt("./output/MDAGC/pred_fusion_151673_last.csv", adata.obs['label_fusion'], delimiter=',')
np.savetxt("./output/MDAGC/pred_fusion_151673_tem_last.csv", adata.obs['label_tem'], delimiter=',')
np.savetxt("./output/MDAGC/h_fusion_151673_last.csv", adata.obsm['h_tem'], delimiter=',')

sc.pp.neighbors(adata, use_rep='h_tem')
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.tl.louvain(adata, key_added='louvain')
# Supplementary Fig. S4A
sc.pl.umap(adata, color=['label_fusion','label'], show = False, save = "umap_IDC.pdf")
sc.tl.paga(adata, groups = 'label_fusion')
sc.pl.paga(adata, color=['label_fusion','label'], show = False, save = "paga_IDC.pdf")