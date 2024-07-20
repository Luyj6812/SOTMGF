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
from precluster import precluster
import scanpy as sc
import numpy as np
from MEG import MEG_construction
from prune import graph_prune
import argparse
import torch
from Train_MDAGC_eachview import train_MDGAC
#from train_MDAGC_eachview_0105 import train_MDGAC
from graph_1221 import precluster_graph, precluster_graph_1
from torch.optim import Adam
import random
import anndata as ad

torch.cuda.empty_cache()
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_clusters', default=5, type=int)
parser.add_argument('--n_feature', default=500, type=int)
parser.add_argument('--n_z', default=10, type=int)##feature extracted with AE
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

####load data
label = pd.read_csv('./SOTMGF_data/murine_breast_cancer/label_murine_breast.csv',header=0, index_col=0, encoding="gbk")
label = np.array(label).squeeze(1)

path = './SOTMGF_data/murine_breast_cancer/ADT/'
adata = scanpy.read_visium(path, genome=None, count_file='GSE198353_mmtv_pymt_GEX_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

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
# proportion_CARD = pd.read_csv('./data/murine_breast_cancer/Proportion_CARD_murine_breast_cancer_processed.csv',header=0, index_col=0, encoding="gbk")
proportion_mRNA = pd.read_csv('./SOTMGF_data/murine_breast_cancer/Proportion_CARD_mRNA.csv',header=0, index_col=0, encoding="gbk")
adata.obsm['proportion_CARD'] = np.array(proportion_mRNA)

precluster_graph(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 600, rad_cutoff = 80, lr = 0.00001)
# sc.pl.spatial(adata,color=['label_tem','label'])
print(adata.obs['label_tem'])

print("######################################iter_mRNA:1######################################")
# precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch =100, lr=0.0000001, n_head = 7)
precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 1000, lr=0.000001, n_head = 4)
sc.pl.spatial(adata,color=['label_tem','label'])
###MEG_construction###
MEG_construction(adata, key_label='label_tem')
###graph_construction and prune_graph
graph_prune(adata, rad_cutoff =80)
train_MDGAC(args, adata,  n_feature=args.n_feature, n_epoch = 300, n_z=args.n_z)
print(adata.obs['label_tem'])
# sc.pl.spatial(adata,color=['label_tem','label_1','label','label_fusion'])
##############################################################################################################
print("######################################iter_mRNA:2######################################")
precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 1500, lr=0.000001, n_head = 4)
# sc.pl.spatial(adata,color=['label_tem','label'])
###MEG_construction###
MEG_construction(adata, key_label='label_tem')
graph_prune(adata, rad_cutoff=80)
###train_MDAGC
train_MDGAC(args, adata, n_feature=args.n_feature, n_epoch = 300, n_z=args.n_z)
print(adata.obs['label_tem'])
# sc.pl.spatial(adata,color=['label_tem','label','label_fusion'])  
torch.cuda.empty_cache()

np.savetxt("./output/MDAGC/pred_fusion_RNA_last.csv", adata.obs['label_fusion'], delimiter=',')
np.savetxt("./output/MDAGC/pred_fusion_RNA_tem_last.csv", adata.obs['label_tem'], delimiter=',')
np.savetxt("./output/MDAGC/h_fusion_RNA_last.csv", adata.obsm['h_tem'], delimiter=',')
adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
adata.obs['RNA'] = adata.obs['label_tem']
# sc.pp.neighbors(adata, use_rep='h_tem')
# sc.tl.umap(adata)
# sc.tl.leiden(adata)
# sc.tl.louvain(adata, key_added='louvain')
# Supplementary Fig. S4A
# sc.pl.umap(adata, color=['label_fusion','label'], show = False, save = "umap_RNA.pdf")
# sc.tl.paga(adata, groups = 'label_fusion')
# sc.pl.paga(adata, color=['label_fusion','label'], show = False, save = "paga_RNA.pdf")
sc.pl.spatial(adata,color=['label','label_tem','label_fusion'],save = "region_RNA.pdf")
sc.pl.spatial(adata,color=['label','label_tem'],save = "region_RNA_1.pdf")
sc.pl.spatial(adata,color=['RNA'],save = "region_RNA_last.pdf")
############################################################################################################################################
path = './SOTMGF_data/murine_breast_cancer/ADT/'
adata_source = scanpy.read_visium(path, genome=None, count_file='GSE198353_mmtv_pymt_GEX_filtered_feature_bc_matrix.h5')
ADT = pd.read_csv('./SOTMGF_data/murine_breast_cancer/ADT/GSE198353_mmtv_pymt_ADT.csv',header=0, index_col=0, encoding="gbk")                                                  
gene2000_ADT = np.array(ADT).T 

n_obs=1978
obs_names = np.array(ADT.columns).tolist()
var_names =np.array(ADT.index).tolist()
n_vars=len(var_names) # 234
obs=pd.DataFrame(index = obs_names)
var=pd.DataFrame(index = var_names)
print(var.head()) # 现在var没有columns(列索引), 只有index(行索引)
adata_ADT = ad.AnnData(gene2000_ADT,obs = obs, var=var, dtype='int32')
adata_ADT.obs = adata_source.obs
adata_ADT.obsm = adata_source.obsm
adata_ADT.uns['spatial'] = adata_source.uns['spatial']
adata_ADT.obsm['spatial'] = adata_source.obsm['spatial']
# adata = adata_ADT
adata_ADT.var_names_make_unique()

##process the data
sc.pp.normalize_total(adata_ADT, inplace=True)
sc.pp.log1p(adata_ADT)
sc.pp.highly_variable_genes(adata_ADT, flavor="seurat", n_top_genes=2000)
##process the data with scanpy routine
sc.pp.pca(adata_ADT,n_comps=30)
sc.pp.neighbors(adata_ADT)
sc.tl.umap(adata_ADT)
adata_ADT.obsm['spatial'] = adata_ADT.obsm['spatial'].astype(int)
print(adata_ADT)

###precluster
adata_ADT = adata_ADT[:, adata_ADT.var['highly_variable']]
gene2000=adata_ADT.to_df()#提取2000维高变基因基因表达信息
gene2000 = np.array(gene2000) 
adata_ADT.obsm['X_tem'] = gene2000
adata_ADT.obs['label'] = np.array(label)
proportion_ADT = pd.read_csv('./SOTMGF_data/murine_breast_cancer/Proportion_CARD_ADT.csv',header=0, index_col=0, encoding="gbk")
adata_ADT.obsm['proportion_CARD'] = np.array(proportion_ADT)
precluster_graph(adata_ADT, label_true = label, n_clusters=args.n_clusters, n_epoch=600, rad_cutoff=79, lr = 0.00001)
# sc.pl.spatial(adata_ADT,color=['label_tem','label'])
print(adata_ADT.obs['label_tem'])
print("######################################iter_ADT:1######################################")
# precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch =100, lr=0.0000001, n_head = 7)
precluster(adata_ADT, label_true = label, n_clusters=args.n_clusters, n_epoch = 1000, lr=0.000001, n_head = 4)
###MEG_construction###
MEG_construction(adata_ADT, key_label='label_tem')
#print(adata)
###graph_construction and prune_graph
graph_prune(adata_ADT, rad_cutoff = 79)
###train_MDAGC
# train_MDGAC(args, adata, n_feature=args.n_feature, n_epoch = 300, n_z=args.n_z)vv
train_MDGAC(args, adata_ADT,  n_feature=args.n_feature, n_epoch = 300, n_z=args.n_z)
print(adata_ADT.obs['label_tem'])
torch.cuda.empty_cache()
adata_ADT.obs['label'] = adata_ADT.obs['label'].astype('category') 
adata_ADT.obs['ADT'] = adata_ADT.obs['label_tem']
# sc.pl.spatial(adata_ADT,color=['label_tem','label'])
sc.pl.spatial(adata_ADT,color=['label','label_tem','label_fusion'],save = "region_ADT.pdf")
sc.pl.spatial(adata_ADT,color=['ADT'],save = "region_ADT_last.pdf")
np.savetxt("./output/MDAGC/pred_fusion_ADT_last.csv", adata_ADT.obs['label_fusion'], delimiter=',')
np.savetxt("./output/MDAGC/pred_fusion_ADT_tem_last.csv", adata_ADT.obs['label_tem'], delimiter=',')
np.savetxt("./output/MDAGC/h_fusion_ADT_last.csv", adata_ADT.obsm['h_tem'], delimiter=',')

################################################################################################################################################
# Supplementary Fig. S4A
# sc.pl.umap(adata_ADT, color=['label_fusion','label','label_tem'], show = False, save = "umap_ADT.pdf")
X_concat = np.concatenate((adata.obsm['h_tem'],adata_ADT.obsm['h_tem']),axis=1)
X_concat = pd.DataFrame(X_concat)
n_obs=1978
obs_names_concat = np.array(ADT.columns).tolist()
var_names_concat =np.array(X_concat.columns).tolist()
n_vars=1000# 234
obs_concat=pd.DataFrame(index = obs_names_concat)
var_concat=pd.DataFrame(index = var_names_concat)
print(var_concat.head()) # 现在var没有columns(列索引), 只有index(行索引)
adata_concat = ad.AnnData(X_concat,obs = obs_concat, var=var_concat, dtype='int32')

adata_concat.obs = adata_source.obs
adata_concat.obsm = adata_source.obsm
adata_concat.uns['spatial'] = adata_source.uns['spatial']
adata_concat.obsm['spatial'] = adata_source.obsm['spatial']
# adata_concat.obsm['X_RNA'] = adata.obsm['h_tem']
# adata_concat.obsm['X_ADT'] = adata_ADT.obsm['h_tem']
adata_concat.obsm['h_RNA'] = adata.obsm['h_tem']
adata_concat.obsm['h_ADT'] = adata_ADT.obsm['h_tem']
# adata = adata_ADT
adata_concat.var_names_make_unique()
adata_concat.obsm['spatial'] = adata_concat.obsm['spatial'].astype(int)
print(adata_concat)

###precluster
# adata_concat = adata_concat[:, adata_concat.var['highly_variable']]
sc.pp.pca(adata_concat,n_comps=30)
sc.pp.neighbors(adata_concat)
sc.tl.umap(adata_concat)
gene2000_concat=adata_concat.to_df()#提取2000维高变基因基因表达信息
gene2000_concat = np.array(gene2000_concat) 
adata_concat.obsm['X_tem'] = gene2000_concat
adata_concat.obs['label'] = np.array(label)
# adata_concat.obsm['proportion_CARD'] = np.array(proportion_CARD)

# ADT_RNA = np.concatenate(adata.obsm['proportion_CARD'], adata_ADT.obsm['proportion_CARD'])
ADT_RNA = pd.concat([proportion_mRNA, proportion_ADT],1)
ADT_RNA = np.array(ADT_RNA)
adata_concat.obsm['proportion_CARD'] = ADT_RNA


adata_concat.obs['label_tem'] = adata_ADT.obs['label_tem']
sc.pl.spatial(adata_concat,color=['label_tem','label'])
precluster(adata_concat, label_true = label, n_clusters=args.n_clusters, n_epoch = 2500, lr=0.000005, n_head = 1)
sc.pl.spatial(adata_concat,color=['label_tem','label'])
# print(adata_concat.obs['label_tem'])


adata_concat.obs['label'] = adata_concat.obs['label'].astype('category')
adata_concat.obs['integrated']  = adata_concat.obs['label_pred']
sc.pl.spatial(adata_concat,color=['label','label_tem','label_pred'],save = "region_concat.pdf")
sc.pl.spatial(adata_concat,color=['label','label_tem'],save = "region_concat_1.pdf")
sc.pl.spatial(adata_concat,color=['integrated'],save = "region_integrated_last.pdf")
# print(adata_concat.obs['label_tem'])
np.savetxt("./output/MDAGC/pred_concat_last.csv", adata_concat.obs['label_pred'], delimiter=',')
np.savetxt("./output/MDAGC/pred_fusion_concat_tem_last.csv", adata_concat.obs['label_tem'], delimiter=',')
np.savetxt("./output/MDAGC/h_fusion_concat_last.csv", adata_concat.obsm['h_tem'], delimiter=',')

sc.pp.neighbors(adata, use_rep='h_tem')
sc.pp.neighbors(adata_ADT, use_rep='h_tem')
sc.pp.neighbors(adata_concat, use_rep='h_tem')
sc.pl.umap(adata_concat, color=['integrated'], show = False, save = "umap_integrated_last.pdf")
sc.pl.umap(adata_ADT, color=['ADT'], show = False, save = "umap_ADT_last.pdf")
sc.pl.umap(adata, color=['RNA'], show = False, save = "umap_RNA_last.pdf")
sc.pl.spatial(adata_concat,color=['label'],save = "region_label_last.pdf")
