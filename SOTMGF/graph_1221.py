import time
from datetime import datetime
import scanpy as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import torch          # 导入 PyTorch 模块，用于深度学习任务
import numpy as np    # 导入 numpy 模块，用于数值计算
from sklearn.model_selection import train_test_split  # 导入 sklearn 库中的 train_test_split 函数，用于数据划分
from sklearn.preprocessing import StandardScaler     # 导入 sklearn 库中的 StandardScaler 类，用于数据标准化
from utils import loss_function, balance_populations, cluster_center,target_distribution, get_M
from model import TransformerModel, DECT, DAEGC
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
from transformers import get_linear_schedule_with_warmup
torch.cuda.device_count()
import torch.nn.functional as F
import random
from B_code.utils import load_graph1, normalize
from utils import Cal_Spatial_Net, Stats_Spatial_Net, prune_spatial_Net

import numpy as np
from munkres import Munkres

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn.preprocessing import normalize
from sklearn import metrics

# similar to https://github.com/karenlatong/AGC-master/blob/master/metrics.py
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    precision_macro = metrics.precision_score(y_true, new_predict, average="macro")
    recall_macro = metrics.recall_score(y_true, new_predict, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predict, average="micro")
    precision_micro = metrics.precision_score(y_true, new_predict, average="micro")
    recall_micro = metrics.recall_score(y_true, new_predict, average="micro")
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    print(f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
    return acc, nmi, ari, f1

def precluster_graph(adata, label_true, n_clusters, n_epoch,rad_cutoff, lr):
    args={
    'seed':1,
    #'lr':0.000005,151673
    #'epochs':600,
    'lr':lr,
    'update_interval':1
    }

    # 3.3 设置随机种子
    # 设置随机种子，让模型每次输出的结果都一样
    seed_value = args['seed']
    random.seed(seed_value)  # 设置 random 模块的随机种子
    np.random.seed(seed_value)  # 设置 numpy 模块的随机种子
    torch.manual_seed(seed_value)  # 设置 PyTorch 中 CPU 的随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():  # 如果可以使用 CUDA，设置随机种子
        torch.cuda.manual_seed(seed_value)  # 设置 PyTorch 中 GPU 的随机种子
        torch.backends.cudnn.deterministic = True  # 使用确定性算法，使每次运行结果一样
        torch.backends.cudnn.benchmark = False  # 不使用自动寻找最优算法加速运算

    #X = adata.obsm['X_tem']
    X = adata.obsm['proportion_CARD']
    label_pre = np.array(adata.obs['label']).reshape(X.shape[0],1)
    label_pre = pd.DataFrame(label_pre).squeeze()
    y = pd.to_numeric(label_pre, errors='coerce').fillna('0').astype('int32')
    y = np.array(y)
    print(y)
    print(label_true)
 
    # 3.7 缩放数据
    scaler = StandardScaler()  # 创建一个标准化转换器的实例
    X = scaler.fit_transform(X)  # 对训练集进行拟合（计算平均值和标准差）
    #X = np.array(X)
    X_cluster = np.array(X)
    # 3.8 转化成pytorch张量
    X = torch.tensor(X).float().to(device)
    #y = torch.tensor(y).long().to(device)
    Cal_Spatial_Net(adata,  rad_cutoff=rad_cutoff)
    adj = load_graph1(X, adata)
    from sklearn.preprocessing import normalize
    adj = normalize(adj, norm="l1")
    M = get_M(adj).to(device)
    adj = torch.from_numpy(adj).float().to(device)

    
##############################################################################################
    # model = DAEGC(num_features=X.shape[1], hidden_size=500,
    #               embedding_size=500, alpha=0.2, num_clusters=n_clusters).to(device) ####################gai0518
    model = DAEGC(num_features=X.shape[1], hidden_size=2000,
                  embedding_size=2000, alpha=0.2, num_clusters=n_clusters).to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])  # 学习率
    # 训练模型
    num_epochs = n_epoch 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_epochs)

    with torch.no_grad():
        #torch.load('./model.pth')
        A_pred, z = model.gat(X, adj, M)
    
    #adata.obsm['h_tem'] = h.squeeze(1).cpu().detach().numpy()
    
    #y_pred_spectral = SpectralClustering(n_clusters=7, gamma=0.2).fit_predict(X_cluster)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    #y_pred = kmeans.fit_predict(h.squeeze(1).data.cpu().numpy())
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pretrain')
    #eva(y,y_pred_spectral,'spectral')
    #model.train()
    t1 = time.time()
    print("start training GAT")


    for epoch in range(num_epochs):

        model.train()
        t = time.time()
        if epoch % args['update_interval'] == 0:
            # update_interval
            A_pred, z, Q = model(X, adj, M)
            q = Q.detach().data.cpu().numpy().argmax(1)
            eva(y, q, epoch)

        
        # 正向传播：将训练数据放到模型中，得到模型的输出
       
        A_pred, z, q = model(X, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj.view(-1))

        loss = 50 * kl_loss + re_loss

        # 反向传播和优化：清零梯度、反向传播计算梯度，并根据梯度更新模型参数
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 根据梯度更新模型参数
        scheduler.step()
        print('Epoch :', epoch + 1, 'loss_trian:', loss.item(),'times:', (time.time()) - t)
    
    #A_pred, z, Q = model(X, adj, M)
    #torch.cuda.empty_cache()
    A_pred, z, q = model(X, adj, M)
    q = Q.detach().data.cpu().numpy().argmax(1)
    eva(y, q, 'prdiction')
    # y_pred_spectral = SpectralClustering(n_clusters=7, gamma=0.2).fit_predict(Q.detach().data.cpu().numpy())
    # eva(y, y_pred_spectral, 'SpectralClustering')
    # np.savetxt("./output/precluster/pred_precluster_graph1221.csv", q, delimiter=',')
    # adata.obs['label_tem'] = y_pred_spectral
    # adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')

    adata.obs['label_tem'] = q
    adata.obsm['h_tem'] = z.squeeze(1).cpu().detach().numpy()
    # # kmeans = KMeans(n_clusters=7,n_init=50)
    # # preds_k = kmeans.fit_predict(z.squeeze(1).cpu().detach().numpy())
    # # kk = metrics.adjusted_rand_score(y, preds_k)  # ARI
    # # kk1 = metrics.normalized_mutual_info_score(y, preds_k)  # NMI
    # # adata.obs['label_tem'] = preds_k

    adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
    torch.cuda.empty_cache()
    return q

'''

'''
if __name__ == "__main__":
####load data
    label = pd.read_csv('./data/DAGC/151673_label.csv',header=None, index_col=None, encoding="gbk")
    label = np.array(label).squeeze(1)

    path = './stMVC_test_data/DLPFC_151673'
    adata = sc.read_visium(path, genome=None, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

##process the data
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

##process the data with scanpy routine
    sc.pp.pca(adata,n_comps=100)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    print(adata)
## true label

###scc cluster
    adata = adata[:, adata.var['highly_variable']]
    gene2000=adata.to_df()#提取2000维高变基因基因表达信息
    gene2000 = np.array(gene2000) 
    adata.obsm['X_tem'] = gene2000
    adata.obs['label'] = np.array(label)
#adata.obs['label'] = adata.obs['label'].astype('category')
#adata.obs['label_tem'] = adata.obs['label']
    precluster_graph(adata, label_true = label)


#############################################################################################
def precluster_graph_1(adata, label_true, n_clusters, n_epoch,rad_cutoff, lr):
    args={
    'seed':1,
    #'lr':0.000005,151673
    #'epochs':600,
    'lr':lr,
    'update_interval':1
    }

    # 3.3 设置随机种子
    # 设置随机种子，让模型每次输出的结果都一样
    seed_value = args['seed']
    random.seed(seed_value)  # 设置 random 模块的随机种子
    np.random.seed(seed_value)  # 设置 numpy 模块的随机种子
    torch.manual_seed(seed_value)  # 设置 PyTorch 中 CPU 的随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():  # 如果可以使用 CUDA，设置随机种子
        torch.cuda.manual_seed(seed_value)  # 设置 PyTorch 中 GPU 的随机种子
        torch.backends.cudnn.deterministic = True  # 使用确定性算法，使每次运行结果一样
        torch.backends.cudnn.benchmark = False  # 不使用自动寻找最优算法加速运算

    # X = adata.obsm['X_tem']
    X = adata.obsm['proportion_CARD']
    label_pre = np.array(adata.obs['label']).reshape(X.shape[0],1)
    label_pre = pd.DataFrame(label_pre).squeeze()
    y = pd.to_numeric(label_pre, errors='coerce').fillna('0').astype('int32')
    y = np.array(y)
    print(y)
    print(label_true)
 
    # 3.7 缩放数据
    scaler = StandardScaler()  # 创建一个标准化转换器的实例
    X = scaler.fit_transform(X)  # 对训练集进行拟合（计算平均值和标准差）
    #X = np.array(X)
    X_cluster = np.array(X)
    # 3.8 转化成pytorch张量
    X = torch.tensor(X).float().to(device)
    #y = torch.tensor(y).long().to(device)
    Cal_Spatial_Net(adata,  rad_cutoff=rad_cutoff)
    adj = load_graph1(X, adata)
    from sklearn.preprocessing import normalize
    adj = normalize(adj, norm="l1")
    M = get_M(adj).to(device)
    adj = torch.from_numpy(adj).float().to(device)
##############################################################################################
    model = DAEGC(num_features=X.shape[1], hidden_size=500,
                  embedding_size=500, alpha=0.2, num_clusters=n_clusters).to(device)
    # model = DAEGC(num_features=X.shape[1], hidden_size=2000,
    #               embedding_size=2000, alpha=0.2, num_clusters=n_clusters).to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])  # 学习率
    # 训练模型
    num_epochs = n_epoch 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_epochs)

    with torch.no_grad():
        #torch.load('./model.pth')
        A_pred, z = model.gat(X, adj, M)
    
    #adata.obsm['h_tem'] = h.squeeze(1).cpu().detach().numpy()
    
    #y_pred_spectral = SpectralClustering(n_clusters=7, gamma=0.2).fit_predict(X_cluster)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    #y_pred = kmeans.fit_predict(h.squeeze(1).data.cpu().numpy())
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    #eva(y, y_pred, 'pretrain')
    #eva(y,y_pred_spectral,'spectral')
    #model.train()
    t1 = time.time()
    print("start training GAT")


    for epoch in range(num_epochs):

        model.train()
        t = time.time()
        if epoch % args['update_interval'] == 0:
            # update_interval
            A_pred, z, Q = model(X, adj, M)
            q = Q.detach().data.cpu().numpy().argmax(1)
            # eva(y, q, epoch)

        
        # 正向传播：将训练数据放到模型中，得到模型的输出
       
        A_pred, z, q = model(X, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj.view(-1))

        loss = 50 * kl_loss + re_loss

        # 反向传播和优化：清零梯度、反向传播计算梯度，并根据梯度更新模型参数
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 根据梯度更新模型参数
        scheduler.step()
        print('Epoch :', epoch + 1, 'loss_trian:', loss.item(),'times:', (time.time()) - t)
    
    #A_pred, z, Q = model(X, adj, M)
    #torch.cuda.empty_cache()
    A_pred, z, q = model(X, adj, M)
    q = Q.detach().data.cpu().numpy().argmax(1)
    # eva(y, q, 'prdiction')
    # y_pred_spectral = SpectralClustering(n_clusters=7, gamma=0.2).fit_predict(Q.detach().data.cpu().numpy())
    # eva(y, y_pred_spectral, 'SpectralClustering')
    # np.savetxt("./output/precluster/pred_precluster_graph1221.csv", q, delimiter=',')
    # adata.obs['label_tem'] = y_pred_spectral
    # adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')

    adata.obs['label_tem'] = q
    adata.obsm['h_tem'] = z.squeeze(1).cpu().detach().numpy()
    # # kmeans = KMeans(n_clusters=7,n_init=50)
    # # preds_k = kmeans.fit_predict(z.squeeze(1).cpu().detach().numpy())
    # # kk = metrics.adjusted_rand_score(y, preds_k)  # ARI
    # # kk1 = metrics.normalized_mutual_info_score(y, preds_k)  # NMI
    # # adata.obs['label_tem'] = preds_k

    adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
    torch.cuda.empty_cache()
    return q
