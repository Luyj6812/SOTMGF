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
from model import  DECT,DECT_1
from sklearn import metrics
from sklearn.cluster import KMeans
from transformers import get_linear_schedule_with_warmup
torch.cuda.device_count()
import torch.nn.functional as F
import random
from utils import seed_everything
'''
args={
    'seed':123,
    'lr':0.000001,
    'epochs':150,
    'ncluster':7,
}
##################input data#######################
path_label = './data/SCC/151673_truth.csv'
label = pd.read_csv(path_label,encoding="gbk")
label = label["Cluster"]

path = './stMVC_test_data/DLPFC_151673'
adata = sc.read_visium(path, genome=None, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]

'''

##########################################################################
def precluster(adata, label_true, n_clusters, n_epoch, lr, n_head):
    args={
    'seed':2023,
    #'lr':0.000001,151673
    'lr':lr,
    #'epochs':1000,
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
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    #gene2000=adata.to_df()#提取2000维高变基因基因表达信息
    X = adata.obsm['X_tem']
    label_pre = np.array(adata.obs['label_tem']).reshape(X.shape[0],1)
    #label_pre = np.array(adata.obs['label']).reshape(3639,1)

    label_pre = pd.DataFrame(label_pre).squeeze()
    y = pd.to_numeric(label_pre, errors='coerce').fillna('0').astype('int32')
    y = np.array(y)

    print(y)
    print(label_true)
    #X['cell_type'] = np.array(label_pre).astype('int')
    # X_Balance = np.c_[X,y]
    # X_Balance = balance_populations(X_Balance)
    # y_balance = X_Balance[:,-1]
    # X_train, X_test, y_train, y_test = train_test_split(X_Balance[:,0:X.shape[1]], y_balance, test_size=0.3, shuffle=True)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    # 3.7 缩放数据
    scaler = StandardScaler()  # 创建一个标准化转换器的实例
    X = scaler.fit_transform(X)  # 对训练集进行拟合（计算平均值和标准差）
    #X_test1 = scaler.transform(X_test)  # 对测试集进行标准化转换，使用与训练集相同的平均值和标准差
    #X = scaler.transform(X)####normalized!!!!
    X = np.array(X)

    # 3.8 转化成pytorch张量
    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).long().to(device)
    # X_train = torch.tensor(X_train1).float().to(device)
    # y_train1 = torch.tensor(y_train).long().to(device)
    # X_test = torch.tensor(X_test1).float().to(device)
    # y_test1 = torch.tensor(y_test).long().to(device)
###train()
    # model = TransformerModel(input_size=X.shape[1],  # 输入维度
    #                          num_classes=args['ncluster'], ).to(device)  # 输出维度
    

    #seed_everything(seed=1)
    # model = DECT(input_size=X.shape[1],  # 输入维度
    #                     num_classes=n_clusters, embedding_size = X.shape[1]).to(device)  # 输出维度
    model = DECT(input_size=X.shape[1],  # 输入维度
                        num_classes=n_clusters, embedding_size = X.shape[1], n_head = n_head ).to(device)  # 输出维度
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])  # 学习率
    # 训练模型
    num_epochs = n_epoch 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_epochs)
    
    # y_pred = y
    # centers =cluster_center(X = X, labels = y_pred)
    # model.cluster_layer.data = torch.tensor(centers).to(device)
    with torch.no_grad():
        #torch.load('./model1.pth')
        reconstruction, h, logvar, predicted_train, Q = model(X)
        #adata.obsm['h_tem'] = h.squeeze(1).cpu().detach().numpy()
        # kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        # y_pred = kmeans.fit_predict(h.squeeze(1).data.cpu().numpy())
        # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
        
        #y_pred = np.array(adata.obs['label_tem'])
        #y_pred = torch.tensor(y_pred).long().to(device)
        y_pred = adata.obs['label_tem']
        centers =cluster_center(h.squeeze(1).data.cpu(), labels = y_pred)
        model.cluster_layer.data = torch.tensor(centers).to(device)

    #model.train()
    t1 = time.time()
    print("start training transformer")
    # for epoch in range(num_epochs):
    #     t = time.time()
    #     # 正向传播：将训练数据放到模型中，得到模型的输出
    #     reconstruction, h, logvar, predicted_train, Q = model(X_train)
    #     p = target_distribution(Q.detach())
    #     q = Q.detach().data.cpu().numpy().argmax(1)
    #     cost, KLD, robust_loss, pre_cost = loss_function(preds=reconstruction, labels=X_train,
    #                                                      mu=h, logvar=logvar,
    #                                                      n_nodes=torch.as_tensor(X_train.shape[0]).cuda(),
    #                                                      pos_weight=torch.as_tensor(0).cuda(),
    #                                                      lamda=None, robust_rep=None,
    #                                                      prediction=predicted_train, true_class=y_train1,
    #                                                      loc=0)
    #     kl_loss = F.kl_div(Q.log(), p, reduction='batchmean')
    #     #print("cost:", cost, "KLD:", KLD, "robust_loss:", robust_loss, "pre_cost:", pre_cost)
    #     loss = cost + KLD + 0.0005 * robust_loss + 8 * pre_cost + 10*kl_loss
    #     predicted_train = torch.max(predicted_train, 1)[1]
    #     ari = adjusted_rand_score(labels_true=y_train1.cpu(), labels_pred=predicted_train.cpu())
    #     #loss = torch.nn.CrossEntropyLoss(predicted1, y_test)
    #     # 反向传播和优化：清零梯度、反向传播计算梯度，并根据梯度更新模型参数
    #     optimizer.zero_grad()  # 清零梯度
    #     loss.backward()  # 反向传播计算梯度
    #     optimizer.step()  # 根据梯度更新模型参数
    #     scheduler.step()
    #     print('Epoch :', epoch + 1, 'loss_trian:', loss.item(),'times:', (time.time()) - t,'ari:',ari)
    #     #print('loss_trian:', loss.item())
    #     #print('times:', (time.time()) - t)


    for epoch in range(num_epochs):

        model.train()
        t = time.time()
        if epoch % args['update_interval'] == 0:
            # update_interval
            reconstruction, h, logvar, predicted_train, Q = model(X)
            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
        
        # 正向传播：将训练数据放到模型中，得到模型的输出
        #reconstruction, h, logvar, predicted_train, Q = model(X)
        reconstruction, h, logvar, predicted_train, Q = model(X)
        p = target_distribution(Q.detach())
        #q = Q.detach().data.cpu().numpy().argmax(1)
        cost, KLD, robust_loss, pre_cost = loss_function(preds=reconstruction, labels=X,
                                                         mu=h, logvar=logvar,
                                                         n_nodes=torch.as_tensor(X.shape[0]).cuda(),
                                                         pos_weight=torch.as_tensor(0).cuda(),
                                                         lamda=None, robust_rep=None,
                                                         prediction=predicted_train, true_class=y,
                                                         loc=0)
        kl_loss = F.kl_div(Q.log(), p, reduction='batchmean')
        #print("cost:", cost, "KLD:", KLD, "robust_loss:", robust_loss, "pre_cost:", pre_cost)
        # loss = cost + KLD + 0.0005 * robust_loss + 20 * pre_cost + kl_loss#########################################################gai0518
        loss = cost + KLD + 0.0005 * robust_loss + 8 * pre_cost + kl_loss
        #loss = cost + KLD + 0.0005 * robust_loss + 8 * pre_cost
        #loss = pre_cost + 10*kl_loss #+ 0.5 * cost
        predicted = torch.max(predicted_train, 1)[1]
        ari_q = adjusted_rand_score(labels_true=label_true, labels_pred = q)
        ari_predicted = adjusted_rand_score(labels_true=label_true, labels_pred = predicted.cpu())
        #loss = torch.nn.CrossEntropyLoss(predicted1, y_test)
        # 反向传播和优化：清零梯度、反向传播计算梯度，并根据梯度更新模型参数
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 根据梯度更新模型参数
        scheduler.step()
        print('Epoch :', epoch + 1, 'loss_trian:', loss.item(),'times:', (time.time()) - t,'ari_q:',ari_q,'ari_predicted:',ari_predicted)
        #print('loss_trian:', loss.item())
        #print('times:', (time.time()) - t)

    reconstruction, h_all, logvar, predicted_all_t, Q = model(X)
    predicted_T = torch.max(predicted_all_t, 1)[1]
    predicted_Q = Q.detach().data.cpu().numpy().argmax(1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50)
    preds_k = kmeans.fit_predict(h_all.squeeze(1).cpu().detach().numpy())

    kk = metrics.adjusted_rand_score(label_true, preds_k)  # ARI
    kk1 = metrics.normalized_mutual_info_score(label_true, preds_k)  # NMI
    ari_all_Q_P = adjusted_rand_score(y.cpu().detach().numpy(), labels_pred=predicted_Q)
    ari_all_Q_TRUE = adjusted_rand_score(label_true, labels_pred=predicted_Q)

    ari_all_T_P = adjusted_rand_score(y.cpu().detach().numpy(), labels_pred=predicted_T.cpu())
    ari_all_T_TRUE = adjusted_rand_score(label_true, labels_pred=predicted_T.cpu())

    
    print("ARI_P_all_fenlei_Q:",ari_all_Q_P,"ARI_all_fenlei_Q:",ari_all_Q_TRUE,"ARI_all_jvlei_k:", kk, "NMI_all_jvlei_k:", kk1)
    print("ARI_P_all_fenlei_T:",ari_all_T_P,"ARI_all_fenlei_T:",ari_all_T_TRUE,"ARI_all_jvlei_k:", kk, "NMI_all_jvlei_k:", kk1)
    torch.save(model.state_dict(), './model_concat.pkl')
    model.eval()
    print("Optimization Finished!")
    print("total time elapsed: {:.4f}s".format(time.time() - t1))
###compute_test()
#     with torch.no_grad():
#         torch.load('./model.pth')
#         reconstruction, h, logvar, predicted_test, Q_test = model(X_test)
#         predicted_test = torch.max(predicted_test, 1)[1]
#         ari = adjusted_rand_score(labels_true=y_test1.cpu(), labels_pred=predicted_test.cpu())
#         print("Test ARI:", ari)
# ###all###
    
#     with torch.no_grad():
#         reconstruction_all, h_all, logvar, predicted_all, Q_all = model(X)
#         predicted_all = torch.max(predicted_all, 1)[1]
#         kmeans = KMeans(n_clusters=args['ncluster'],n_init=50)
#         preds_k = kmeans.fit_predict(h_all.squeeze(1).cpu().detach().numpy())
#         kk = metrics.adjusted_rand_score(label_true, preds_k)  # ARI
#         kk1 = metrics.normalized_mutual_info_score(label_true, preds_k)  # NMI
#         ari_all_P = adjusted_rand_score(y, labels_pred=predicted_all.cpu())
#         ari_all = adjusted_rand_score(label_true, labels_pred=predicted_all.cpu())
#         print("ARI_P_all_fenlei:",ari_all,"ARI_all_fenlei:",ari_all_P,"ARI_all_jvlei:", kk, "NMI_all_jvlei:", kk1)
#     #print("ARI_all_jvlei:", kk)
#     #print("NMI_all_jvlei:", kk1)
#         np.savetxt("./output/precluster/h_gene_precluster.csv",h_all.squeeze(1).cpu().detach().numpy(),delimiter=',')
#         np.savetxt("./output/precluster/pred_precluster.csv", preds_k,delimiter=',')

    adata.obs['label_pred'] = preds_k
    #adata.obs['label_tem'] = predicted_Q
    adata.obs['label_tem'] = predicted_T.cpu().detach().numpy()#######################xiugai0519
    #adata.obs['label_tem'] = predicted_all.squeeze(1)
    adata.obsm['h_tem'] = h_all.squeeze(1).cpu().detach().numpy()
    #adata.obsm['X_tem'] = h_all.squeeze(1).cpu().detach().numpy()
    adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
    adata.obs['label_pred'] = adata.obs['label_pred'].astype('category')
    np.savetxt("./output/precluster/h_gene_precluster.csv",h_all.squeeze(1).cpu().detach().numpy(),delimiter=',')
    np.savetxt("./output/precluster/pred_precluster.csv", predicted_Q,delimiter=',')

    torch.cuda.empty_cache()


def precluster_1(adata, label_true, n_clusters, n_epoch, lr, n_head):
    args={
    'seed':1,
    #'lr':0.000001,151673
    'lr':lr,
    #'epochs':1000,
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
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    #gene2000=adata.to_df()#提取2000维高变基因基因表达信息
    X = adata.obsm['X_tem']
    label_pre = np.array(adata.obs['label_tem']).reshape(X.shape[0],1)
    #label_pre = np.array(adata.obs['label']).reshape(3639,1)

    label_pre = pd.DataFrame(label_pre).squeeze()
    y = pd.to_numeric(label_pre, errors='coerce').fillna('0').astype('int32')
    y = np.array(y)

    print(y)
    print(label_true)
    # 3.7 缩放数据
    scaler = StandardScaler()  # 创建一个标准化转换器的实例
    X = scaler.fit_transform(X)  # 对训练集进行拟合（计算平均值和标准差）
    X = np.array(X)

    # 3.8 转化成pytorch张量
    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).long().to(device)
    model = DECT(input_size=X.shape[1],  # 输入维度
                        num_classes=n_clusters, embedding_size = X.shape[1], n_head = n_head ).to(device)  # 输出维度
    # model = DECT_1(input_size=X.shape[1],  # 输入维度
    #                     num_classes=n_clusters, embedding_size = 500, n_head = n_head ).to(device)  # 输出维度
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])  # 学习率
    # 训练模型
    num_epochs = n_epoch 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_epochs)
    
    with torch.no_grad():
        reconstruction, h, logvar, predicted_train, Q = model(X)
        y_pred = adata.obs['label_tem']
        centers =cluster_center(h.squeeze(1).data.cpu(), labels = y_pred)
        model.cluster_layer.data = torch.tensor(centers).to(device)

    #model.train()
    t1 = time.time()
    print("start training transformer")


    for epoch in range(num_epochs):

        model.train()
        t = time.time()
        if epoch % args['update_interval'] == 0:
            # update_interval
            reconstruction, h, logvar, predicted_train, Q = model(X)
            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
        
        # 正向传播：将训练数据放到模型中，得到模型的输出
        #reconstruction, h, logvar, predicted_train, Q = model(X)
        reconstruction, h, logvar, predicted_train, Q = model(X)
        p = target_distribution(Q.detach())
        cost, KLD, robust_loss, pre_cost = loss_function(preds=reconstruction, labels=X,
                                                         mu=h, logvar=logvar,
                                                         n_nodes=torch.as_tensor(X.shape[0]).cuda(),
                                                         pos_weight=torch.as_tensor(0).cuda(),
                                                         lamda=None, robust_rep=None,
                                                         prediction=predicted_train, true_class=y,
                                                         loc=0)
        kl_loss = F.kl_div(Q.log(), p, reduction='batchmean')
        # loss = cost + KLD + 0.0005 * robust_loss + 20 * pre_cost + kl_loss#########################################################gai0518
        loss = cost + KLD + 0.0005 * robust_loss + 8 * pre_cost + kl_loss
        predicted = torch.max(predicted_train, 1)[1]
        ari_q = adjusted_rand_score(labels_true=label_true, labels_pred = q)
        ari_predicted = adjusted_rand_score(labels_true=label_true, labels_pred = predicted.cpu())
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 根据梯度更新模型参数
        scheduler.step()
        print('Epoch :', epoch + 1, 'loss_trian:', loss.item(),'times:', (time.time()) - t,'ari_q:',ari_q,'ari_predicted:',ari_predicted)


    reconstruction, h_all, logvar, predicted_all_t, Q = model(X)
    predicted_T = torch.max(predicted_all_t, 1)[1]
    predicted_Q = Q.detach().data.cpu().numpy().argmax(1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50)
    preds_k = kmeans.fit_predict(h_all.squeeze(1).cpu().detach().numpy())

    kk = metrics.adjusted_rand_score(label_true, preds_k)  # ARI
    kk1 = metrics.normalized_mutual_info_score(label_true, preds_k)  # NMI
    ari_all_Q_P = adjusted_rand_score(y.cpu().detach().numpy(), labels_pred=predicted_Q)
    ari_all_Q_TRUE = adjusted_rand_score(label_true, labels_pred=predicted_Q)

    ari_all_T_P = adjusted_rand_score(y.cpu().detach().numpy(), labels_pred=predicted_T.cpu())
    ari_all_T_TRUE = adjusted_rand_score(label_true, labels_pred=predicted_T.cpu())

    
    print("ARI_P_all_fenlei_Q:",ari_all_Q_P,"ARI_all_fenlei_Q:",ari_all_Q_TRUE,"ARI_all_jvlei_k:", kk, "NMI_all_jvlei_k:", kk1)
    print("ARI_P_all_fenlei_T:",ari_all_T_P,"ARI_all_fenlei_T:",ari_all_T_TRUE,"ARI_all_jvlei_k:", kk, "NMI_all_jvlei_k:", kk1)
    torch.save(model.state_dict(), './model_concat.pkl')
    model.eval()
    print("Optimization Finished!")

    # adata.obs['label_pred'] = preds_k
    adata.obs['label_tem'] = predicted_T.cpu().detach().numpy()#######################xiugai0519
    adata.obsm['h_tem'] = h_all.squeeze(1).cpu().detach().numpy()
    adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
    adata.obs['label_pred'] = adata.obs['label_pred'].astype('category')
    np.savetxt("./output/precluster/h_gene_precluster.csv",h_all.squeeze(1).cpu().detach().numpy(),delimiter=',')
    np.savetxt("./output/precluster/pred_precluster.csv", predicted_Q,delimiter=',')

    torch.cuda.empty_cache()


