from __future__ import unicode_literals
import pandas as pd
import sklearn
import scanpy as sc
import sklearn.neighbors._base
import sys
import numpy as np
from utils import Cal_Spatial_Net, Stats_Spatial_Net, prune_spatial_Net

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

'''
################input_data###################
path_label_scc = './data/SCC/label_scc_7_0.2603.csv'
label_scc = pd.read_csv(path_label_scc,header=None,index_col=None)
label_scc = label_scc.squeeze()

path = './stMVC_test_data/DLPFC_151673'
adata = sc.read_visium(path, genome=None, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]  ####adata2提取高变基因

ref_adata = adata
ref_adata.obs['celltype'] = label_scc    ####ref_data加入标签
ref_adata.obs_names_make_unique()
gene2000= ref_adata.to_df()#提取2000维高变基因基因表达信息
print(ref_adata)
'''
############################################################################
def graph_prune(adata,rad_cutoff):
    #Cal_Spatial_Net(adata, rad_cutoff=150)
    Cal_Spatial_Net(adata, rad_cutoff)
    Stats_Spatial_Net(adata)
    #pre_labels = 'expression_scc_label'
    #adata.obs[pre_labels] = adata.obs['label_step2']
    Spatial_Net = adata.uns['Spatial_Net']
    KNN_df = adata.uns['KNN_df']
    G_df = Spatial_Net.copy()
    prune_G_df, prune_KNN_df = prune_spatial_Net(G_df, adata, label=adata.obs['label_tem'])
    print(prune_G_df)
    print(prune_KNN_df)
    adata.uns['KNN_df'] = KNN_df.iloc[:,0:3]
    adata.uns['prune_KNN_df'] = prune_KNN_df.iloc[:,0:3]
    #G_df.to_excel("./output/prune/G_df3_scc_yuan.xlsx")
    #prune_G_df.to_excel("./output/prune/prune_G_scc_3.xlsx")
    #MEG_df = adata.obsp['connectivities']
    #MEG
#graph_prune(adata)