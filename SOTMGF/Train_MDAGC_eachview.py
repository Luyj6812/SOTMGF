import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from B_code.utils import load_data, load_graph, normalize_adj, numpy_to_torch
from B_code.GNN_previous import GNNLayer
from B_code.eva_previous import eva
from datetime import datetime
import time
import random
from model_eachview import SDCN,target_distribution,num_net_parameter, fusion_eachview, AE
import pandas as pd
import scanpy as sc

# import scipy.io as scio

# class cross_view(nn.Module):
#     def __init__(self, adata, args):
#         super(cross_view).__init__()
#         self.view1 = SDCN(500, 500, 2000, 2000, 500, 500,
#                  n_input=adata.obsm['h_tem'].shape[1],
#                  n_z=args.n_z,
#                  n_clusters=args.n_clusters,
#                  v=1.0).cuda() 
#         self.view2 = SDCN(500, 500, 2000, 2000, 500, 500,
#                  n_input=adata.obsm['h_tem'].shape[1],
#                  n_z=args.n_z,
#                  n_clusters=args.n_clusters,
#                  v=1.0).cuda() 
#         self.args = args

#     def fit_cross_view(self, dataset, adj1, adj2, adj3, adj4):

class train_multiview(nn.Module):
    def __init__(self, adata, n_z, n_clusters, n_feature):
        super(train_multiview, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.view1 = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=adata.obsm['h_tem'].shape[1],
                 n_z=n_z, n_feature=n_feature,
                 n_clusters=n_clusters,
                 v=1.0).to(self.device)
        self.view2 = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=adata.obsm['h_tem'].shape[1],
                 n_z=n_z, n_feature=n_feature,
                 n_clusters=n_clusters,
                 v=1.0).to(self.device)   
        self.view3 = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=adata.obsm['h_tem'].shape[1],
                 n_z=n_z, n_feature=n_feature,
                 n_clusters=n_clusters,
                 v=1.0).to(self.device) 
        self.view4 = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=adata.obsm['h_tem'].shape[1],
                 n_z=n_z, n_feature=n_feature,
                 n_clusters=n_clusters,
                 v=1.0).to(self.device)     
        self.ae = AE(
                n_enc_1=500,
                n_enc_2=500,
                n_enc_3=2000,
                n_dec_1=2000,
                n_dec_2=500,
                n_dec_3=500,
                n_input=adata.obsm['h_tem'].shape[1],
                n_z=n_z).to(self.device)    
        self.n_clusters = n_clusters
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z ))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    
    def train_sdcn_singleview(self, args, dataset, adata):
        # seed_value = 1
        # random.seed(seed_value)  # 设置 random 模块的随机种子
        # np.random.seed(seed_value)  # 设置 numpy 模块的随机种子
        # torch.manual_seed(seed_value) 
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # if torch.cuda.is_available():  # 如果可以使用 CUDA，设置随机种子
        #     torch.cuda.manual_seed(seed_value)  # 设置 PyTorch 中 GPU 的随机种子
        #     torch.backends.cudnn.deterministic = True  # 使用确定性算法，使每次运行结果一样
        #     torch.backends.cudnn.benchmark = False  # 不使用自动寻找最优算法加速运算
        # from utils import seed_everything
        # seed_everything(seed=1)
        adj1, adj2, adj3, adj4 = load_graph(adata)
        adj1 = adj1.cuda()  # .to(device)
        adj2 = adj2.cuda()
        adj3 = adj3.cuda()
        adj4 = adj4.cuda()
    #data = torch.Tensor(dataset.x).cuda()  # .to(device)
    #y = dataset.y
        print('#############Start train view 1################')
        # x_bar1, net_output1_l, res1, z1, net_output1 = self.view1.fit_SDCN(args, adata, dataset, adj1, ld1=0.05, ld2=0.01, ld3=0.02, ld4=0, lr=5e-5, epoch = 100)
        x_bar1, net_output1_l, res1, z1, net_output1 = self.view1.fit_SDCN(args, adata, dataset, adj1, ld1=0.1, ld2=0.01, ld3=0.02, ld4=0, lr=1e-5, epoch = 300)
        adata.obs['label_1'] = res1
        adata.obs['label_1'] = adata.obs['label_1'].astype('category')
        #sc.pl.spatial(adata, img_key = "hires", color=['label_1'], size=1.5)
        # print (x_bar1)
        # print(net_output1)
        # print(res1)
        # print (x_bar2)
        print(net_output1)
        # print(res2)
        print('#############Start train view 2################')
        # x_bar2, net_output2_l, res2, z2, net_output2 = self.view2.fit_SDCN(args, adata, dataset, adj2, ld1=0.11, ld2=0.035, ld3=0.025, ld4=0, lr=1e-5, epoch = 300)
        x_bar2, net_output2_l, res2, z2, net_output2 = self.view2.fit_SDCN(args, adata, dataset, adj2, ld1=0.11, ld2=0.035, ld3=0.025, ld4=0, lr=1e-5, epoch = 100)
        # print (x_bar2)
        print(net_output2.shape)
        # print(res2)
        print('#############Start train view 3################')
        # x_bar3, net_output3_l, res3, z3, net_output3 = self.view3.fit_SDCN(args, adata, dataset, adj3, ld1=0.1, ld2=0.025, ld3=0.015, ld4=0, lr=1e-5,epoch = 300)
        x_bar3, net_output3_l, res3, z3, net_output3 = self.view3.fit_SDCN(args, adata, dataset, adj3, ld1=0.1, ld2=0.025, ld3=0.015, ld4=0, lr=1e-5,epoch = 300)
        # print (x_bar3)
        print(net_output3)
        # print(res3)
        print('#############Start train view 4################')
        # x_bar4, net_output4_l, res4, z4, net_output4 = self.view4.fit_SDCN(args, adata, dataset, adj4, ld1=0.1, ld2=0.03, ld3=0.015, ld4=0, lr=1e-5, epoch = 300)
        x_bar4, net_output4_l, res4, z4, net_output4 = self.view4.fit_SDCN(args, adata, dataset, adj4, ld1=0.1, ld2=0.03, ld3=0.015, ld4=0, lr=1e-5, epoch = 300)
        # print (x_bar4)
        print(net_output4)
        # print(res4)
   

    def train_sdcn( self, args, dataset, adata, n_epoch, n_z, n_feature):
            seed_value = 1
            random.seed(seed_value)  # 设置 random 模块的随机种子
            np.random.seed(seed_value)  # 设置 numpy 模块的随机种子
            torch.manual_seed(seed_value)  # 设置 PyTorch 中 CPU 的随机种子
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():  # 如果可以使用 CUDA，设置随机种子
                torch.cuda.manual_seed(seed_value)  # 设置 PyTorch 中 GPU 的随机种子
                torch.backends.cudnn.deterministic = True  # 使用确定性算法，使每次运行结果一样
                torch.backends.cudnn.benchmark = False  # 不使用自动寻找最优算法加速运算
            ld1 = args.ld1
            ld2 = args.ld2
            ld3 = args.ld3
            ld4 = args.ld4
            #dataname = args.name
            #file_out = open('./output/' + dataname + '_results.out', 'a')
            print("lambda_1: ", ld1, "lambda_2: ", ld2, "lambda_3: ", ld3)
            adj1, adj2, adj3, adj4 = load_graph(adata)
            adj1 = adj1.to(self.device)  # .to(device)
            adj2 = adj2.to(self.device)
            adj3 = adj3.to(self.device)
            adj4 = adj4.to(self.device)

            x = torch.Tensor(dataset.x).to(self.device)
            y = dataset.y

            x_bar1, q1, pred1, z1, net_output1_l, pl_loss1, z_F1, net_output1 = self.view1.inference(x, adj1)
            x_bar2, q2, pred2, z2, net_output2_l, pl_loss2, z_F2, net_output2 = self.view2.inference(x, adj2) 
            x_bar3, q3, pred3, z3, net_output3_l, pl_loss3, z_F3, net_output3 = self.view3.inference(x, adj3)
            x_bar4, q4, pred4, z4, net_output4_l, pl_loss4, z_F4, net_output4 = self.view4.inference(x, adj4)
            res4_1 = z_F1.data.cpu().numpy().argmax(1)
            acc_1, nmi_1, ari_1, f1_1 = eva(y, res4_1)
            print("view1 result:","acc:", acc_1, "nmi:", nmi_1, "ari:",ari_1, "f1:",f1_1, 'res4_1:', res4_1)
            adata.obs['label_1'] = res4_1
            adata.obs['label_1'] = adata.obs['label_1'].astype('category')

            res4_2 = z_F2.data.cpu().numpy().argmax(1)
            acc_2, nmi_2, ari_2, f1_2 = eva(y, res4_2)
            print("view2 result:","acc:", acc_2, "nmi:", nmi_2, "ari:",ari_2, "f1:",f1_2, 'res4_2:', res4_2)
            adata.obs['label_2'] = res4_2
            adata.obs['label_2'] = adata.obs['label_2'].astype('category')


            res4_3= z_F3.data.cpu().numpy().argmax(1)
            acc_3, nmi_3, ari_3, f1_3 = eva(y, res4_3)
            print("view3 result:","acc:", acc_3, "nmi:", nmi_3, "ari:",ari_3, "f1:",f1_3, 'res4_3:', res4_3)
            adata.obs['label_3'] = res4_3
            adata.obs['label_3'] = adata.obs['label_3'].astype('category')


            res4_4 = z_F4.data.cpu().numpy().argmax(1)
            acc_4, nmi_4, ari_4, f1_4 = eva(y, res4_4)
            print("view4 result:","acc:", acc_4, "nmi:", nmi_4, "ari:",ari_4, "f1:",f1_4, 'res4_1:', res4_4)
            adata.obs['label_4'] = res4_4
            adata.obs['label_4'] = adata.obs['label_4'].astype('category')
            # sc.pl.spatial(adata, img_key = "hires", color=['label_1','label_2', 'label_3', 'label_4'], size=1.5)
            # sc.pl.scatter(adata, basis='xy_loc', color=['label_1','label_2', 'label_3', 'label_4'])
            # x_bar3, net_output3, res3, z3 = self.view3.inference(x, adj3)
            # x_bar4, net_output4, res4, z4 = self.view4.inference(x, adj4)
            #z = torch.tensor(adata.obsm['h_tem']).to(self.device)######>???????netoutput1
    
            #torch.cuda.empty_cache()
    
            model_fusion = fusion_eachview(n_clusters = args.n_clusters, n_z=n_z, n_feature=n_feature).to(self.device)
            optimizer = Adam(model_fusion.parameters(), lr=args.lr)

            data = torch.Tensor(dataset.x).cuda()  # .to(device)
            y = dataset.y
            with torch.no_grad():
                _, _, _, _, z = self.ae(data)


            z_1st = z
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=args.n_clusters, random_state=1)
            y_pred = kmeans.fit_predict(z_1st.data.cpu().numpy())
        #y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()  # .to(device)

            kmeans_iter_F = []
            NMI_iter_F = []
            ARI_iter_F = []
            F1_iter_F = []
            print("########################################start fusion###########################################")
            for epoch in range(n_epoch):
                q, pred, z, net_output_l, pl_loss, z_F, net_output_f = model_fusion(x, z, net_output1, net_output2, net_output3, net_output4, adj2)
                p = target_distribution(pred.data)
                res4 = z_F.data.cpu().numpy().argmax(1)
                acc, nmi, ari, f1 = eva(y, res4, str(epoch) + 'F')
                kmeans_iter_F.append(acc)
                NMI_iter_F.append(nmi)
                ARI_iter_F.append(ari)
                F1_iter_F.append(f1)

        #x_bar, q, pred, _, net_output, pl_loss, z_F= model(data, adj1, adj2, adj3)

                KL_QP = F.kl_div(q.log(), p, reduction='batchmean')
                KL_ZP = F.kl_div(pred.log(), p, reduction='batchmean')
        #re_loss = F.mse_loss(x_bar, data)
        #pseudo_label = torch.tensor(adata.obs['label']).cuda()
                # pseudo_label = torch.tensor(adata.obs['label_tem']).cuda()
        #print(adata)
                # pse_loss = F.cross_entropy(z_F, pseudo_label)
                loss =  ld1 * (KL_QP + KL_ZP) \
                      + ld2 * (F.kl_div(q.log(), pred, reduction='batchmean')) \
                      + ld3 * pl_loss #+ ld4 * pse_loss
                adata.obs['label_fusion'] = res4
                adata.obs['label_fusion'] = adata.obs['label_fusion'].astype('category')
                adata.obsm['z_f'] = np.array(z_F.data.cpu())


                optimizer.zero_grad()  # 清零梯度
                loss.backward(retain_graph=True)  # 反向传播计算梯度
                optimizer.step()  # 根据梯度更新模型参数
        
                print('Epoch: {:04d}'.format(epoch + 1),
                    'ARI: {:.4f}'.format(ari),
                    'loss: {:.4f}'.format(loss))
                

            y_pred = kmeans.fit_predict(net_output_f.data.cpu().numpy())   
            adata.obsm['h_tem'] = net_output_f.cpu().detach().numpy()
            #adata.obs['label_tem'] = y_pred
            adata.obs['label_tem'] = y_pred
            #adata.obs['label_tem'] = adata.obs['label_1']
            acc, nmi, ari, f1 = eva(y, y_pred,'fusion')
            adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
            #adata.obs['label_1'] = adata.obs['label_1']
            adata.obs['label_1'] = adata.obs['label_1'].astype('category')

            print("##################FUSION#######:","ARI:",ari,"NMI:",nmi, "f1:",f1)
            # np.savetxt("./output/MDAGC/h_fusion_151673_net.csv", net_output_f.cpu().detach().numpy(), delimiter=',')
            # np.savetxt("./output/MDAGC/pred_fusion_res4_151673.csv", res4, delimiter=',')
            # np.savetxt("./output/MDAGC/pred_fusion_km_151673.csv", y_pred, delimiter=',')




def train_MDGAC(args, adata, n_feature, n_epoch, n_z):
    tic = time.time()
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    seed_MDAGC = 2023
    random.seed(seed_MDAGC)
    np.random.seed(seed_MDAGC)
    torch.manual_seed(seed_MDAGC)
    torch.cuda.manual_seed(seed_MDAGC)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():  # 如果可以使用 CUDA，设置随机种子
        torch.cuda.manual_seed(seed_MDAGC)  # 设置 PyTorch 中 GPU 的随机种子
        torch.backends.cudnn.deterministic = True  # 使用确定性算法，使每次运行结果一样
        torch.backends.cudnn.benchmark = False  # 不使用自动寻找最优算法加速运算
    print('seed:', seed_MDAGC)
        # 🎏 iters
    iters = 1  
    for iter_num in range(iters):
        print('iter_num:', iter_num)
        dataset = load_data(adata)
        n_input = adata.obsm['h_tem'].shape[1]
        print("n_input:",n_input)
        #if args.name == '151673':
        #args.lr = 5e-5#####0519xiugai
        args.lr = 1e-5
        args.ld1 = 0.1
        args.ld2 = 0.01
        args.ld3 = 0.0015
        args.ld4 = 0
        args.n_input = n_input
        #args.epoch = 1000
        print(args)
        #train_sdcn(args, dataset, adata)
        model = train_multiview(adata, n_z=10, n_clusters=args.n_clusters, n_feature=n_feature)
        model.train_sdcn_singleview(args, dataset, adata)
        model.train_sdcn( args, dataset, adata,  n_epoch = n_epoch, n_z=n_z, n_feature=n_feature)
    toc = time.time()
    print("Time:", (toc - tic))

#train_MDGAC()