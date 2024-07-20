from sotip import *
import numpy as np
import scanpy as sc
import pandas as pd
import random
'''
###load 151673 cortex data
path = './stMVC_test_data/DLPFC_151673'
adata = sc.read_visium(path, genome=None, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]

##process the data with scanpy routine
sc.pp.pca(adata,n_comps=100)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata,resolution=2)
'''
def MEG_construction(adata,key_label):
    # 设置随机种子，让模型每次输出的结果都一样
    seed_value = 1
    random.seed(seed_value)  # 设置 random 模块的随机种子
    np.random.seed(seed_value)  # 设置 numpy 模块的随机种子
    #torch.manual_seed(seed_value)  # 设置 PyTorch 中 CPU 的随机种子
    knn = 10
    # spatial coordination used for ME
    spatial_var='spatial'

    # cluster label used for ME
    cls_key=key_label

    # order of cluster label for ME representation (adata.obsm['ME'])
    ME_var_names_np_unique = np.array(adata.obs[cls_key].cat.categories)

    # this function added a ME obsm for adata
    MED(adata,use_cls=cls_key,nn=knn,copy=False,ME_var_names_np_unique=ME_var_names_np_unique,spatial_var=spatial_var)

    ###
    # the resulted ME representation is a n*k vector
    # n is the number of MEs (cells), k is the number of unique cell clusters
    # the i,jth element of the representation is defined as the frequency of cell type_j in ME_i
    adata.obsm['ME'].shape
    np.savetxt("./output/MEG/ME.csv",adata.obsm['ME'],delimiter=',')

    ####step 2: Connectivity guided minimum graph distance (CGMGD)
    ##step 2.1 topological structure is computed with paga
    sc.tl.paga(adata,groups=cls_key)
    sc.pl.paga_compare(adata,basis='X_umap',show = False)

    ##step 2.2 the connectivities between cell clusters is used to guide the graph distance computation
    gd_method = 'paga_guided_umap'
    gd = get_ground_distance(adata,method=gd_method,cls_key=cls_key)
    #np.savetxt("./output/MECNG/CGMGD.csv",gd,delimiter=',')
    # plot the CGMGD

    ###step 3 compute pairwise ME distance
    # this step will add a X_ME_EMD_mat obsm to adata_phEMD
    adata_phEMD = MED_phEMD_mp(
        adata.copy(),         # the used anndata
        GD_method=gd_method,  # use CGMGD as ground distance
        MED_knn=knn,          # ME size is set consistently as 10
        CT_obs=cls_key,       # use leiden cluster label
        ifspatialplot=False,  # do not show intermediate result plot
        OT_method='pyemd',    # use pyemd to compute EMD as ME distance
        ME_precompyted=True,  # use precomputed ME representation (already computed in step 1)
        GD_precomputed=True,  # use precomputed ground distance (already computed in step 2)
        mp=10                # multi process to accelerate the computation
    )

    # emd_distmat is the n*n pairwise ME distance
    emd_distmat = (adata_phEMD.obsm['X_ME_EMD_mat'])

    ###step 4 construct ME graph
    # emd_distmat is computed by step 3
    adata.obsp['ME_EMD_mat'] = emd_distmat
    np.savetxt('./output/MEG/ME_EMD_mat.csv',emd_distmat,delimiter=',')
    # number of neighbors is set to construct ME graph
    n_neighbors=200
    # compute the MEG, each node is a ME, edge is the connectivity. use similar manner as scanpy
    knn_indices, knn_dists, forest = sc.neighbors.compute_neighbors_umap( emd_distmat, n_neighbors=n_neighbors, metric='precomputed' )
    adata.obsp['distances'], adata.obsp['connectivities'] = sc.neighbors._compute_connectivities_umap(
        knn_indices,
        knn_dists,
        adata.shape[0],
        n_neighbors, # change to neighbors you plan to use
    )
    aa=adata.obsp['distances'].toarray()
    bb=adata.obsp['connectivities'].toarray()
    # set the ME graph's associated information (connectivity matrix, distance matrix) to neighbors_EMD
    #adata.uns['neighbors_EMD'] = adata.uns['neighbors'].copy()
    np.savetxt("./output/MEG/MEG_connectivities.csv",bb,delimiter=",")
    np.savetxt("./output/MEG/MEG_distances.csv",aa,delimiter=",")


#MEG_construction(adata,key_label='leiden')
#print(adata.obs['leiden'])