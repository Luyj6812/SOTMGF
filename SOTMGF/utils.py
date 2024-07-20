from typing import Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from anndata import AnnData
from kneed import KneeLocator
from scipy.sparse import spmatrix
from sklearn.decomposition import PCA
import pandas as pd
import sklearn
import numpy as np
from typing import Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import anndata
from scipy.sparse import isspmatrix
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import random
import os

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    #os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def get_M(adj):
   # adj_numpy = adj.cpu().numpy()
    adj_numpy = adj
    # t_order
    t=3
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


'''scc cluster'''
def compute_pca_components(
    matrix: Union[np.ndarray, spmatrix], random_state: Optional[int] = 1, save_curve_img: Optional[str] = None
) -> Tuple[Any, int, float]:
    """
    Calculate the inflection point of the PCA curve to
    obtain the number of principal components that the PCA should retain.

    Args:
        matrix: A dense or sparse matrix.
        save_curve_img: If save_curve_img != None, save the image of the PCA curve and inflection points.
    Returns:
        new_n_components: The number of principal components that PCA should retain.
        new_components_stored: Percentage of variance explained by the retained principal components.
    """
    # Convert sparse matrix to dense matrix.
    matrix = to_dense_matrix(matrix)
    matrix[np.isnan(matrix)] = 0

    # Principal component analysis (PCA).
    pca = PCA(n_components=None, random_state=random_state)
    pcs = pca.fit_transform(matrix)

    # Percentage of variance explained by each of the selected components.
    # If n_components is not set then all components are stored and the sum of the ratios is equal to 1.0.
    raw_components_ratio = pca.explained_variance_ratio_
    raw_n_components = np.arange(1, raw_components_ratio.shape[0] + 1)

    # Calculate the inflection point of the PCA curve.
    kl = KneeLocator(raw_n_components, raw_components_ratio, curve="convex", direction="decreasing")
    new_n_components = int(kl.knee)
    new_components_stored = round(float(np.sum(raw_components_ratio[:new_n_components])), 3)

    # Whether to save the image of PCA curve and inflection point.
    if save_curve_img is not None:
        kl.plot_knee()
        plt.tight_layout()
        plt.savefig(save_curve_img, dpi=100)

    return pcs, new_n_components, new_components_stored

def pca_spateo(
    adata: AnnData,
    X_data=None,
    n_pca_components: Optional[int] = None,
    pca_key: Optional[str] = "X_pca",
    genes: Union[list, None] = None,
    layer: Union[str, None] = None,
    random_state: Optional[int] = 1,
):
    """
    Do PCA for dimensional reduction.

    Args:
        adata:
            An Anndata object.
        X_data:
            The user supplied data that will be used for dimension reduction directly.
        n_pca_components:
            The number of principal components that PCA will retain. If none, will Calculate the inflection point
            of the PCA curve to obtain the number of principal components that the PCA should retain.
        pca_key:
            Add the PCA result to :attr:`obsm` using this key.
        genes:
            The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`,
            all genes will be used.
        layer:
            The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, will use
            ``adata.X``.
    Returns:
        adata_after_pca: The processed AnnData, where adata.obsm[pca_key] stores the PCA result.
    """
    if X_data is None:
        if genes is not None:
            genes = adata.var_names.intersection(genes).to_list()
            #lm.main_info("Using user provided gene set...")
            print("Using user provided gene set...")
            if len(genes) == 0:
                raise ValueError("no genes from your genes list appear in your adata object.")
        else:
            genes = adata.var_names
        if layer is not None:
            matrix = adata[:, genes].layers[layer].copy()
            #lm.main_info('Runing PCA on adata.layers["' + layer + '"]...')
            print('Runing PCA on adata.layers["' + layer + '"]...')
        else:
            matrix = adata[:, genes].X.copy()
            #lm.main_info("Runing PCA on adata.X...")
            print(("Runing PCA on adata.X..."))
    else:
        matrix = X_data.copy()
        #lm.main_info("Runing PCA on user provided data...")
        print("Runing PCA on user provided data...")

    if n_pca_components is None:
        pcs, n_pca_components, _ = compute_pca_components(adata.X, random_state=random_state, save_curve_img=None)
    else:
        matrix = to_dense_matrix(matrix)
        pca = PCA(n_components=n_pca_components, random_state=random_state)
        pcs = pca.fit_transform(matrix)

    adata.obsm[pca_key] = pcs[:, :n_pca_components]

def spatial_adj_dyn(
   # adata: AnnData,
    adata,
    spatial_key: str = "spatial",
    pca_key: str = "pca",
    e_neigh: int = 30,
    s_neigh: int = 6,
    n_pca_components: int = 30,
):
    """
    Calculate the adjacent matrix based on a neighborhood graph of gene expression space
    and a neighborhood graph of physical space.
    """
    import dynamo as dyn

    # Compute a neighborhood graph of gene expression space.
    dyn.tl.neighbors(adata, X_data=adata.obsm[pca_key], n_neighbors=e_neigh, n_pca_components=n_pca_components)

    # Compute a neighborhood graph of physical space.
    dyn.tl.neighbors(
        adata,
        X_data=adata.obsm[spatial_key],
        n_neighbors=s_neigh,
        result_prefix="spatial",
        n_pca_components=n_pca_components,
    )

    # Calculate the adjacent matrix.
    conn = adata.obsp["connectivities"].copy()
    conn.data[conn.data > 0] = 1
    adj = conn + adata.obsp["spatial_connectivities"]
    adj.data[adj.data > 0] = 1
    return adj

# Convert sparse matrix to dense matrix.
to_dense_matrix = lambda X: np.array(X.todense()) if isspmatrix(X) else X

def scc(
    adata: anndata.AnnData,
    spatial_key: str = "spatial",
    key_added: Optional[str] = "scc",
    pca_key: str = "pca",
    e_neigh: int = 30,
    s_neigh: int = 6,
    cluster_method: Literal["leiden", "louvain"] = "leiden",
    resolution: Optional[float] = None,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    Reference:
        Ao Chen, Sha Liao, Mengnan Cheng, Kailong Ma, Liang Wu, Yiwei Lai, Xiaojie Qiu, Jin Yang, Wenjiao Li, Jiangshan
        Xu, Shijie Hao, Xin Wang, Huifang Lu, Xi Chen, Xing Liu, Xin Huang, Feng Lin, Zhao Li, Yan Hong, Defeng Fu,
        Yujia Jiang, Jian Peng, Shuai Liu, Mengzhe Shen, Chuanyu Liu, Quanshui Li, Yue Yuan, Huiwen Zheng, Zhifeng Wang,
        H Xiang, L Han, B Qin, P Guo, PM Cánoves, JP Thiery, Q Wu, F Zhao, M Li, H Kuang, J Hui, O Wang, B Wang, M Ni, W
        Zhang, F Mu, Y Yin, H Yang, M Lisby, RJ Cornall, J Mulder, M Uhlen, MA Esteban, Y Li, L Liu, X Xu, J Wang.
        Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays. Cell, 2022.
    Args:
        adata: an Anndata object, after normalization.
        spatial_key: the key in `.obsm` that corresponds to the spatial coordinate of each bucket.
        key_added: adata.obs key under which to add the cluster labels.
        pca_key: the key in `.obsm` that corresponds to the PCA result.
        e_neigh: the number of nearest neighbor in gene expression space.
        s_neigh: the number of nearest neighbor in physical space.
        cluster_method: the method that will be used to cluster the cells.
        resolution: the resolution parameter of the louvain clustering algorithm.
        copy: Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments.
            Defaults to False.
    Returns:
        Depends on the argument `copy`, return either an `~anndata.AnnData` object with cluster info in "scc_e_{a}_s{b}"
        or None.
    """
    import dynamo as dyn
    # Calculate the adjacent matrix.
    adj = spatial_adj_dyn(
        adata=adata,
        spatial_key=spatial_key,
        pca_key=pca_key,
        e_neigh=e_neigh,
        s_neigh=s_neigh,
    )
    # Perform clustering.
    if cluster_method == "leiden":
        # Leiden clustering.
        dyn.tl.leiden(adata, adj_matrix=adj, resolution=resolution, result_key=key_added)
    elif cluster_method == "louvain":
        # Louvain clustering.
        dyn.tl.louvain(adata, adj_matrix=adj, resolution=resolution, result_key=key_added)

    return adata if copy else None


def balance_populations(data):
    ct_names = np.unique(data[:,-1])
    ct_counts = pd.value_counts(data[:,-1])
    max_val = min(ct_counts.max(),np.int32(2000000/len(ct_counts)))
    balanced_data = np.empty(shape=(1,data.shape[1]),dtype=np.float32)
    for ct in ct_names:
        tmp = data[data[:,-1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        tmp_X = tmp[idx]
        balanced_data = np.r_[balanced_data,tmp_X]
    return np.delete(balanced_data,0,axis=0)

def cluster_center(X, labels):
    X = X.cpu().detach().numpy()
    #labels = labels.cpu().detach().numpy()
    labels = np.array(labels)
    #labels = labels.ravel()
    unique_labels = set(labels)
    centers = []

    for label in unique_labels:
        center = np.mean(X[labels == label], axis=0)
        centers.append(center)
    
    return centers

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def supervised_multiple_loss_function(preds, labels):
    # labels = torch.from_numpy(labels).cuda()
    cost = F.cross_entropy(preds, labels)

    return torch.mean(cost)

def loss_function(preds, labels, mu, logvar, n_nodes,  pos_weight,
                  lamda=None, robust_rep=None, prediction=None,
                  true_class=None,  loc=0):
   cost = F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

   if robust_rep is not None and lamda is not None:
      lamda = lamda.detach()
      robust_rep = robust_rep.detach()

      robust_loss = torch.sum(lamda[:, loc] * (torch.sum((mu - robust_rep) ** 2, dim=1)))
   # print(robust_loss)

   else:
      robust_loss = torch.tensor(0.0)

   if logvar is None:
      KLD = torch.tensor(0.0)
   else:
      KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

   if prediction is None:
      predict_class = torch.tensor(0.0)

   else:
      predict_class = supervised_multiple_loss_function(prediction, true_class,
                                                        )  

   return cost, KLD, robust_loss, predict_class

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
#def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it ] *indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff +1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it ] *indices.shape[1] ,indices[it ,:], distances[it ,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance' ] >0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net
    adata.uns['KNN_df'] = KNN_df

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)

def prune_spatial_Net(Graph_df, adata, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' %Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label']==Graph_df['Cell2_label'],]
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    id_cell_trans = dict(zip(np.array(coor.index),range(coor.shape[0]), ))
    KNN_df = Graph_df.copy()
    KNN_df['Cell1'] = KNN_df['Cell1'].map(id_cell_trans)
    KNN_df['Cell2'] = KNN_df['Cell2'].map (id_cell_trans)
    print('%d edges after pruning.' %Graph_df.shape[0])
    return Graph_df,KNN_df
