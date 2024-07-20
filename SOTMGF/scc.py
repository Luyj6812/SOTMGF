import pandas as pd
from utils import pca_spateo, scc
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import scanpy

path_label = './data/SCC/151673_truth.csv'
label = pd.read_csv(path_label,encoding="gbk")
label = label["Cluster"]

path = './stMVC_test_data/DLPFC_151673'
adata = scanpy.read_visium(path, genome=None, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

scanpy.pp.normalize_total(adata, inplace=True)
scanpy.pp.log1p(adata)
scanpy.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

adata = adata[:, adata.var['highly_variable']]
pca_spateo(adata, n_pca_components=50)
scc(adata, s_neigh=10, resolution=1.3, cluster_method="louvain", key_added="scc", pca_key="X_pca")


