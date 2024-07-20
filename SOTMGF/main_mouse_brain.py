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
from precluster import precluster
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
###############################
from sirv import SIRV
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
import seaborn as sns
# scanpy.set_figure_params(dpi = 300)


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
parser.add_argument('--n_clusters', default=20, type=int)
parser.add_argument('--n_feature', default=500, type=int)
parser.add_argument('--n_z', default=10, type=int)##feature extracted with AE
#parser.add_argument('--pretrain_path', type=str, default='pkl')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")
#args.pretrain_path = 'data/{}.pkl'.format(args.name)

####load data
label = pd.read_csv('./SOTMGF_data/Mouse_brain/label_leiden_mouse_brain.csv',header=0, index_col=0, encoding="gbk")
adata = scv.read('./SOTMGF_data/Mouse_brain/HybISS_adata.h5ad')
proportion_CARD = pd.read_csv('./SOTMGF_data/Mouse_brain/Proportion_CARD_mouse_brain_processed.csv',header=0, index_col=0, encoding="gbk")
#proportion_CARD = proportion_CARD.reindex(label.index)
label = np.array(label).squeeze(1)
adata.obsm['proportion_CARD'] = np.array(proportion_CARD)


##process the data
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

##process the data with scanpy routine
sc.pp.pca(adata,n_comps=100)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata.obsm['spatial'] = adata.obsm['xy_loc'].astype(int)
print(adata)
## true label

###deconvolution
adata = adata[:, adata.var['highly_variable']]
gene2000=adata.to_df()#提取2000维高变基因基因表达信息
gene2000 = np.array(gene2000) 
adata.obsm['X_tem'] = gene2000
adata.obs['label'] = np.array(label)

precluster_graph(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 500, rad_cutoff = 5, lr = 0.000003)
sc.pl.scatter(adata, basis='xy_loc', color=['label_tem','label'])
#sc.pl.spatial(adata,color=['label_tem','label'])
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
# adata.obsm['X_bar'] = adata.obsm['X_tem']
# sc.pl.spatial(adata,color=['label_pred'])
####################!!!!!!epoch1=!!!!!!!!!!!!!!!!!!5######
print("######################################iter:1######################################")
precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch =100, lr=0.0000001, n_head = 7)
sc.pl.scatter(adata, basis='xy_loc', color=['label_tem','label'])
#sc.pl.spatial(adata,color=['label_tem','label'])


###MEG_construction###
MEG_construction(adata, key_label='label_tem')
    #print(adata)

###graph_construction and prune_graph

graph_prune(adata,rad_cutoff = 5)

###train_MDAGC
train_MDGAC(args, adata, n_feature=args.n_feature, n_epoch = 300, n_z=args.n_z)
    #adata.obs['label_tem'] = adata.obs['label_MDAGC']
   
print(adata.obs['label_tem'])
sc.pl.scatter(adata, basis='xy_loc', color=['label_tem','label'])
    #adata.obs['label_tem'] = pd.Series(adata.obs['label_tem'])
#sc.pl.spatial(adata,color=['label_tem','label_1','label','label_fusion'])
    #adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')

torch.cuda.empty_cache()


# ##############################################################################################################
# adata.obs['label_tem'] = adata.obs['label_2']
print("######################################iter:2######################################")
precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 1500, lr=0.0000001, n_head = 7)
#sc.pl.spatial(adata,color=['label_tem','label'])

#     print('label_precluster:', adata.obs['label_tem'])

###MEG_construction###
MEG_construction(adata, key_label='label_tem')
    #print(adata)

###graph_construction and prune_graph

graph_prune(adata,rad_cutoff = 5)

###train_MDAGC
train_MDGAC(args, adata, n_feature=args.n_feature, n_epoch = 300, n_z=args.n_z)
    #adata.obs['label_tem'] = adata.obs['label_MDAGC']
   
print(adata.obs['label_tem'])

    #adata.obs['label_tem'] = pd.Series(adata.obs['label_tem'])
#sc.pl.spatial(adata,color=['label_tem','label','label_fusion'])
    #adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
#sc.pl.spatial(adata,color=['label_fusion','label'])   
torch.cuda.empty_cache()
adata.obs['label_fusion'] = adata.obs['label_tem']
np.savetxt("./output/MDAGC/pred_fusion_mouse_brain.csv", adata.obs['label_fusion'], delimiter=',')
# ##############################################################################################################
# print("######################################iter:3######################################")
# precluster(adata, label_true = label, n_clusters=args.n_clusters, n_epoch = 1500)
# #sc.pl.spatial(adata,color=['label_tem','label'])

# #     print('label_precluster:', adata.obs['label_tem'])

# ###MEG_construction###
# MEG_construction(adata, key_label='label_tem')
#     #print(adata)

# ###graph_construction and prune_graph

# graph_prune(adata,rad_cutoff = 50)

# ###train_MDAGC
# train_MDGAC(args, adata, n_feature=args.n_feature, n_epoch = 250, n_z=args.n_z)
#     #adata.obs['label_tem'] = adata.obs['label_MDAGC']
   
# print(adata.obs['label_tem'])

#     #adata.obs['label_tem'] = pd.Series(adata.obs['label_tem'])
# #sc.pl.spatial(adata,color=['label_tem','label','label_fusion'])
#     #adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
# #sc.pl.spatial(adata,color=['label_fusion','label'])   
# torch.cuda.empty_cache()


# ########################################RNA velocities#######################################################
# scanpy.set_figure_params(dpi = 300)

RNA = scv.read('./SOTMGF_data/Mouse_brain/RNA_adata.h5ad')
# HybISS = scv.read('SIRV-datasets/Mouse_brain/HybISS_adata.h5ad')
HybISS = adata
# sc.pp.highly_variable_genes(RNA, flavor="seurat", n_top_genes=7000)
# RNA = RNA[:, RNA.var['highly_variable']]
# RNA_data = RNA.to_df()
# RNA_data.to_csv("SIRV-datasets/RNA_data_7000_mouse_brain.csv",sep=',',index=True, header=True)
# print("finish")


# Apply SIRV to integrate both datasets and predict the un/spliced expressions
# for the spatially measured genes, additionally transfer 'Region' and
# 'Subclass' label annotations from scRNA-seq to spatial data
HybISS_imputed = SIRV(HybISS,RNA,50,['Region','Subclass'])

# Normalize the imputed un/spliced expressions, this will also re-normalize the
# full spatial mRNA 'X', this needs to be undone 
scv.pp.normalize_per_cell(HybISS_imputed, enforce=True)

# Undo the double normalization of the full mRNA 'X'
HybISS_imputed.X = HybISS.to_df()[HybISS_imputed.var_names]

# Zero mean and unit variance scaling, PCA, building neibourhood graph, running
# umap and cluster the HybISS spatial data using Leiden clustering
sc.pp.scale(HybISS_imputed)
sc.tl.pca(HybISS_imputed)
sc.pl.pca_variance_ratio(HybISS_imputed, n_pcs=50, log=True, show = False, save = "1_pca_variance_ratio_mouse_brain.pdf")
sc.pp.neighbors(HybISS_imputed, n_neighbors=30, use_rep='h_tem', n_pcs=30)
sc.tl.umap(HybISS_imputed)
sc.tl.leiden(HybISS_imputed)
# Supplementary Fig. S4A
sc.pl.umap(HybISS_imputed, color=['label_fusion','label'], show = False, save = "2_umap_mouse_brain.pdf")
# Supplementary Fig. S4B
sc.pl.scatter(HybISS_imputed, basis='xy_loc', color=['label_fusion','label'], show = False, save = "3_umap_mouse_xyloc.pdf")

# Calculating RNA velocities and projecting them on the UMAP embedding and spatial
# coordinates of the tissue
scv.pp.moments(HybISS_imputed, n_pcs=30, n_neighbors=30)
scv.tl.velocity(HybISS_imputed)
scv.tl.velocity_graph(HybISS_imputed)
# Fig. 4A
scv.pl.velocity_embedding_stream(HybISS_imputed, basis='umap', color=['label_fusion','label'],  save = "4_velocity_embedding_stream_umap.pdf")
# Fig. 4B
scv.pl.velocity_embedding_stream(HybISS_imputed, basis='xy_loc', color=['label_fusion','label'],size=60,legend_fontsize=4,legend_loc='right',  save = "5_velocity_embedding_stream_xy_loc.pdf")

# Cell-level RNA velocities 
# Supplementary Fig. S5A
scv.pl.velocity_embedding(HybISS_imputed,basis='xy_loc', color=['label_fusion','label'], show = False, save = "./figures/6_velocity_embedding_xy_loc.pdf")

# Visualizing transferred label annotations on UMAP embedding and spatial coordinates
# Supplementary Fig. S7A
sc.pl.umap(HybISS_imputed, color='Region', show = False, save = "7_umap_Region.pdf")
# Supplementary Fig. S7B
sc.pl.scatter(HybISS_imputed, basis='xy_loc',color='Region',show = False, save = "8_scatter_Region_xy_loc.pdf")
# Fig. 4C
sc.pl.umap(HybISS_imputed, color='Subclass',show = False, save = "9_umap_x_subclass.pdf")
# Fig. 4D
sc.pl.scatter(HybISS_imputed, basis='xy_loc',color='Subclass',show = False, save = "10_scatter_xy_loc.pdf")

# Intepretation of RNA velocities using transferred label annotations
# Fig. 5A
scv.pl.velocity_embedding(HybISS_imputed,basis='xy_loc', color='Subclass',show = False, save = "./figures/11_velocity_embedding_xy_loc.pdf")

# Comparing cell clusters with transferred 'Subclass' and 'Class' annotations
def Norm(x):
    return (x/np.sum(x))

# Subclass annotation
cont_mat = contingency_matrix(HybISS_imputed.obs.leiden.astype(np.int_),HybISS_imputed.obs.Subclass)
df_cont_mat = pd.DataFrame(cont_mat,index = np.unique(HybISS_imputed.obs.leiden.astype(np.int_)), 
                           columns=np.unique(HybISS_imputed.obs.Subclass))

df_cont_mat = df_cont_mat.apply(Norm,axis=1)
# Supplementary Fig. S7C
plt.figure()
sns.heatmap(df_cont_mat,annot=True,fmt='.2f')
plt.yticks(np.arange(df_cont_mat.shape[0])+0.5,df_cont_mat.index)
plt.xticks(np.arange(df_cont_mat.shape[1])+0.5,df_cont_mat.columns)
plt.savefig('./figures/heatmap.pdf')