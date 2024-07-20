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
from model import SDCN,target_distribution,num_net_parameter
import pandas as pd
from utils import  cluster_center
# import scipy.io as scio



def train_sdcn(args, dataset, adata):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 🎏
    dataname = args.name
    file_out = open('./output/' + dataname + '_results.out', 'a')
    #print("The experimental results", file=file_out)

    ld1 = args.ld1
    ld2 = args.ld2
    ld3 = args.ld3
    ld4 = args.ld4
    print("lambda_1: ", ld1, "lambda_2: ", ld2, "lambda_3: ", ld3, file=file_out)
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=adata.obsm['h_tem'].shape[1],
                 n_z=128,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)

    # model = SDCN(500, 500, 2000, 2000, 500, 500,
    #              n_input=adata.obsm['h_tem'].shape[1],
    #              n_z=args.n_z,
    #              n_clusters=args.n_clusters,
    #              v=1.0).to(device)

    print(num_net_parameter(model))

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph

    adj1, adj2, adj3, adj4 = load_graph(adata)
    adj1 = adj1.cuda()  # .to(device)
    adj2 = adj2.cuda()
    adj3 = adj3.cuda()
    adj4 = adj4.cuda()

    data = torch.Tensor(dataset.x).cuda()  # .to(device)
    y = dataset.y
    with torch.no_grad():
        #torch.load('./model_MDAGC.pth')
        _, _, _, _, z = model.ae(data)

    iters10_kmeans_iter_F = []
    iters10_NMI_iter_F = []
    iters10_ARI_iter_F = []
    iters10_F1_iter_F = []

    z_1st = z
    # kmeans = KMeans(n_clusters=args.n_clusters, n_init=7)
    # y_pred = kmeans.fit_predict(z_1st.data.cpu().numpy())
    #y_pred = kmeans.fit_predict(data.data.cpu().numpy())
    #model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()  # .to(device)

    y_pred = np.array(adata.obs['label_tem'])
    y_pred = torch.tensor(y_pred).long().to(device)
    centers =cluster_center(z_1st, labels = y_pred)
    model.cluster_layer.data = torch.tensor(centers).to(device)
    
    #acc, nmi, ari, f1 = eva(y, y_pred.cpu().detach().numpy(), 'pae')
    acc, nmi, ari, f1 = eva(y, y_pred, 'pae')
    print("acc:", acc, "nmi:", nmi, "ari:", ari, "f1:", f1)

    kmeans_iter_F = []
    NMI_iter_F = []
    ARI_iter_F = []
    F1_iter_F = []

    for epoch in range(args.epoch):

        if epoch % 1 == 0:
            x_bar, q, pred, _, net_output, pl_loss, z_F = model(data, adj1, adj2, adj3, adj4)
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
            re_loss = F.mse_loss(x_bar, data)
            #pseudo_label = torch.tensor(adata.obs['label']).cuda()
            pseudo_label = torch.tensor(adata.obs['label_tem']).cuda()
            #print(adata)
            pse_loss = F.cross_entropy(z_F, pseudo_label)
            loss = re_loss \
                   + ld1 * (KL_QP + KL_ZP) \
                   + ld2 * (F.kl_div(q.log(), pred, reduction='batchmean')) \
                   + ld3 * pl_loss + ld4 * pse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: {:04d}'.format(epoch + 1),
                  'ARI: {:.4f}'.format(ari),
                  'loss: {:.4f}'.format(loss))
    # _F
    kmeans_max = np.max(kmeans_iter_F)
    nmi_max = np.max(NMI_iter_F)
    ari_max = np.max(ARI_iter_F)
    F1_max = np.max(F1_iter_F)
    iters10_kmeans_iter_F.append(round(kmeans_max, 5))
    iters10_NMI_iter_F.append(round(nmi_max, 5))
    iters10_ARI_iter_F.append(round(ari_max, 5))
    iters10_F1_iter_F.append(round(F1_max, 5))
    print("#################" + dataname + "####################")
    print("kmeans F mean", round(np.mean(iters10_kmeans_iter_F), 5), "max", np.max(iters10_kmeans_iter_F), "\n",
          iters10_kmeans_iter_F)
    print("NMI mean", round(np.mean(iters10_NMI_iter_F), 5), "max", np.max(iters10_NMI_iter_F), "\n",
          iters10_NMI_iter_F)
    print("ARI mean", round(np.mean(iters10_ARI_iter_F), 5), "max", np.max(iters10_ARI_iter_F), "\n",
          iters10_ARI_iter_F)
    print("F1  mean", round(np.mean(iters10_F1_iter_F), 5), "max", np.max(iters10_F1_iter_F), "\n", iters10_F1_iter_F)
    print(':acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(round(np.mean(iters10_kmeans_iter_F), 5),
                                                                        round(np.mean(iters10_NMI_iter_F), 5),
                                                                        round(np.mean(iters10_ARI_iter_F), 5),
                                                                        round(np.mean(iters10_F1_iter_F), 5)))
    
    torch.save(model.state_dict(), './model_MDAGC.pth')
    ##################################################################
    #adata.obsm['X_tem'] =  x_bar.cpu().detach().numpy()
    adata.obsm['h_tem'] = net_output.cpu().detach().numpy()
    adata.obs['label_tem'] = res4
    adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
    np.savetxt("./output/MDAGC/h_scc_151673_net.csv", net_output.cpu().detach().numpy(), delimiter=',')
    np.savetxt("./output/MDAGC/pred_scc_151673.csv", res4, delimiter=',')

'''
if __name__ == "__main__":
    # 🎏 iters
    iters = 10  #
    for iter_num in range(iters):
        print('iter_num：', iter_num)
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='151673')
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--n_clusters', default=7, type=int)
        parser.add_argument('--n_z', default=10, type=int)class fusion_eachview(nn.Module):
    def __init__(self, n_clusters) -> None:
        super(fusion_eachview, self).__init__()

        self.agnn_n = GNNLayer(4 * n_clusters, n_clusters)
        self.mlp_n = MLP_n(4 * n_clusters)
        self.mlp_ZQ = MLP_ZQ(2 * n_clusters)

    def forward(self, x, z, net_output_1, net_output_2, net_output_3, net_output_4, adj2):
        x_array = list(np.shape(x))
        n_x = x_array[0]
        wn = self.mlp_n(torch.cat((net_output_1, net_output_2, net_output_3, net_output_4), 1))
        wn = F.normalize(wn, p=2)

        wn0 = torch.reshape(wn[:, 0], [n_x, 1])
        wn1 = torch.reshape(wn[:, 1], [n_x, 1])
        wn2 = torch.reshape(wn[:, 2], [n_x, 1])
        wn3 = torch.reshape(wn[:, 3], [n_x, 1])

        # 2️⃣ [Z+H]
        tile_wn0 = wn0.repeat(1, self.NFeature)
        tile_wn1 = wn1.repeat(1, self.NFeature)
        tile_wn2 = wn2.repeat(1, self.NFeature)
        tile_wn3 = wn3.repeat(1, self.NFeature)

        net_output = torch.cat((tile_wn0 * net_output_1, tile_wn1 * net_output_2, tile_wn2 * net_output_3, tile_wn3 * net_output_4), 1)
        net_output = self.agnn_n(net_output, adj2 , active=False)
        predict = torch.softmax(net_output, dim=1)
        ###z_F = alpha * z_F_1 + beta * z_F_2 + gamma * z_F_3

        # Dual Self-supervision(都可用？？以下3行）
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        #q = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        #####融合####

        p_ZH = self.mlp_ZQ(torch.cat((predict, q), 1))

        p_ZH = F.normalize(p_ZH, p=2)

        p_ZH1 = torch.reshape(p_ZH[:, 0], [n_x, 1])
        p_ZH2 = torch.reshape(p_ZH[:, 1], [n_x, 1])
        p_ZH1_broadcast = p_ZH1.repeat(1, self.n_clusters)
        p_ZH2_broadcast = p_ZH2.repeat(1, self.n_clusters)
        z_F = p_ZH1_broadcast.mul(predict) + p_ZH2_broadcast.mul(q)
        z_F = F.softmax(z_F, dim=1)

        # # 🟡pseudo_label_loss
        clu_assignment = torch.argmax(z_F, -1)
        clu_assignment_onehot = F.one_hot(clu_assignment, self.n_clusters)
        thres = 0.7
        thres_matrix = torch.zeros_like(z_F) + thres
        weight_label = torch.ge(F.normalize(z_F, p=2), thres_matrix).type(torch.cuda.FloatTensor)
        pseudo_label_loss = BCE(z_F, clu_assignment_onehot, weight_label)
        return  q, predict, z, net_output, pseudo_label_loss, z_F

        args.pretrain_path = 'data/{}.pkl'.format(args.name)
        dataset = load_data(args.name)
        if args.name == '151673':
            args.lr = 1e-5
            args.k = None
            args.ld1 = 0.09
            args.ld2 = 0.005
            args.ld3 = 0.005
            args.n_clusters = 7
            args.n_input = 2000
            args.epoch = 1000
        print(args)
        train_sdcn(dataset)

    toc = time.time()
    print("Time:", (toc - tic))
'''
#def train_MDGAC(args,adata):
def train_MDGAC(args,adata):
    tic = time.time()
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    seed_MDAGC = 2023
    random.seed(seed_MDAGC)
    np.random.seed(seed_MDAGC)
    torch.manual_seed(seed_MDAGC)
    torch.cuda.manual_seed(seed_MDAGC)
    print('seed:', seed_MDAGC)
        # 🎏 iters
    iters = 1  
    for iter_num in range(iters):
        print('iter_num:', iter_num)
        dataset = load_data(adata)
        n_input = adata.obsm['h_tem'].shape[1]
        print("n_input:",n_input)
        
        args.lr = 5e-6
        args.ld1 = 0.11 #(20)
        args.ld2 = 0.01 #0.01(1)
        args.ld3 = 0.015#0.005(0.5)
        args.ld4 = 0.04#0.06(1)
        args.n_clusters = 7
        args.n_input = n_input
        args.epoch = 200#200
        print(args)
        train_sdcn(args, dataset, adata)

    toc = time.time()
    print("Time:", (toc - tic))

#train_MDGAC()