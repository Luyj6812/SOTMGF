from torch import nn
import torch
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
from utils import  cluster_center


###Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size,  # 输入维度
                                                        nhead=8,dim_feedforward=1024)             # 注意力头数
        self.encoder = nn.TransformerEncoder(self.encoder_layer,             # 编码层
                                             num_layers=6)                   # 层数
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_size,nhead=8,dim_feedforward=1024)
        self.dc = nn.TransformerDecoder(decoder_layer=self.decoder_layer ,num_layers=6)
        self.fc = nn.Linear(input_size,
                             num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)    # 增加一个维度，变成(batch_size, 1, input_size)的形状
        x = self.encoder(x)   # 输入Transformer编码器进行编码
        return self.dc(torch.rand(x.shape).cuda(),x).squeeze(1),   x,   None,    self.fc(x.squeeze(1))
    
class AE_P(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_class):
        super(AE_P, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)
        self.fc = nn.Linear(n_z,n_class)

    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))
        z = self.z_layer(enc_z4)

        dec_z2 = F.relu(self.dec_1(z))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar, enc_z2, enc_z3, enc_z4, z, None, self.fc(z.squeeze(1))


###SDCN

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))
        z = self.z_layer(enc_z4)

        dec_z2 = F.relu(self.dec_1(z))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar, enc_z2, enc_z3, enc_z4, z


class MLP_n(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_n, self).__init__()
        self.wl = Linear(n_mlp, 4)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        return weight_output


class MLP_L(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        return weight_output


class MLP_1(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1)
        return weight_output


class MLP_2(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)
        return weight_output


class MLP_3(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)
        return weight_output


class MLP_ZQ(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_ZQ, self).__init__()
        self.w_ZQ = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w_ZQ(mlp_in)), dim=1)
        return weight_output


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, n_feature = 500):
        super(SDCN, self).__init__()

        # autoencoder
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        ## 3️⃣
        self.agnn_0 = GNNLayer(n_input, n_enc_1)
        self.agnn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.agnn_2 = GNNLayer(n_enc_2, n_enc_3)
        self.agnn_3 = GNNLayer(n_enc_3, n_z)
        self.agnn_z = GNNLayer(3020, n_feature)
        self.agnn_n = GNNLayer(n_feature, n_clusters)

        self.mlp = MLP_L(3020)
        self.mlp_n = MLP_n(n_clusters)
        # attention on [z_i, h_i]
        self.mlp1 = MLP_1(2 * n_enc_1)
        self.mlp2 = MLP_2(2 * n_enc_2)
        self.mlp3 = MLP_3(2 * n_enc_3)

        self.mlp_ZQ = MLP_ZQ(2 * n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

        self.n_clusters = n_clusters
        self.n_z = 10


    def inference(self, x, adj):
        # DAE Module
        x_bar, h1, h2, h3, z = self.ae(x)

        x_array = list(np.shape(x))
        n_x = x_array[0]
        ##########################################################adj1###########################################################
        # HWF
        # 5️⃣ GCN[p_1*Z + p_2*H]
        # z1
        z1_1 = self.agnn_0(x, adj)
        # z2
        p1_1 = self.mlp1(torch.cat((h1, z1_1), 1))
        p1_1 = F.normalize(p1_1, p=2)
        p11_1 = torch.reshape(p1_1[:, 0], [n_x, 1])
        p12_1 = torch.reshape(p1_1[:, 1], [n_x, 1])
        p11_broadcast_1 = p11_1.repeat(1, 500)
        p12_broadcast_1 = p12_1.repeat(1, 500)
        z2_1 = self.agnn_1(p11_broadcast_1.mul(z1_1) + p12_broadcast_1.mul(h1), adj)
        # z3
        p2_1 = self.mlp2(torch.cat((h2, z2_1), 1))
        p2_1 = F.normalize(p2_1, p=2)
        p21_1 = torch.reshape(p2_1[:, 0], [n_x, 1])
        p22_1 = torch.reshape(p2_1[:, 1], [n_x, 1])
        p21_broadcast_1 = p21_1.repeat(1, 500)
        p22_broadcast_1 = p22_1.repeat(1, 500)
        z3_1 = self.agnn_2(p21_broadcast_1.mul(z2_1) + p22_broadcast_1.mul(h2), adj)
        # z4
        p3_1 = self.mlp3(torch.cat((h3, z3_1), 1))  # self.mlp3(h2)
        p3_1 = F.normalize(p3_1, p=2)
        p31_1 = torch.reshape(p3_1[:, 0], [n_x, 1])
        p32_1 = torch.reshape(p3_1[:, 1], [n_x, 1])
        p31_broadcast_1 = p31_1.repeat(1, 2000)
        p32_broadcast_1 = p32_1.repeat(1, 2000)
        z4_1 = self.agnn_3(p31_broadcast_1.mul(z3_1) + p32_broadcast_1.mul(h3), adj)
        # SWF
        w_1 = self.mlp(torch.cat((z1_1, z2_1, z3_1, z4_1, z), 1))
        w_1 = F.normalize(w_1, p=2)

        w0_1 = torch.reshape(w_1[:, 0], [n_x, 1])
        w1_1 = torch.reshape(w_1[:, 1], [n_x, 1])
        w2_1 = torch.reshape(w_1[:, 2], [n_x, 1])
        w3_1 = torch.reshape(w_1[:, 3], [n_x, 1])
        w4_1 = torch.reshape(w_1[:, 4], [n_x, 1])

        # 2️⃣ [Z+H]
        tile_w0_1 = w0_1.repeat(1, 500)
        tile_w1_1 = w1_1.repeat(1, 500)
        tile_w2_1 = w2_1.repeat(1, 2000)
        tile_w3_1 = w3_1.repeat(1, 10)
        tile_w4_1 = w4_1.repeat(1, 10)

        # 2️⃣ concat
        net_output = torch.cat(
            (tile_w0_1.mul(z1_1), tile_w1_1.mul(z2_1), tile_w2_1.mul(z3_1), tile_w3_1.mul(z4_1), tile_w4_1.mul(z)), 1)
        
        ########jiayigetuzhuyiwangluo######
        net_output = self.agnn_z(net_output, adj, active=False)
        ########################################################################################################
        net_output_l = self.agnn_n(net_output, adj , active=False)
        predict = torch.softmax(net_output_l, dim=1)
        ###z_F = alpha * z_F_1 + beta * z_F_2 + gamma * z_F_3

        # Dual Self-supervision(都可用？？以下3行）
        #q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

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

        return x_bar, q, predict, z, net_output_l, pseudo_label_loss, z_F,  net_output
    
    def forward(self, x, adj):

        x_bar, q, pred, z, net_output_l, pl_loss, z_F, net_output = self.inference(x, adj)

        return x_bar, q, pred, z, net_output_l, pl_loss, z_F, net_output


    def fit_SDCN(self, args, adata, dataset, adj, ld1, ld2, ld3, ld4, lr, epoch):

        # dataname = args.name
        # file_out = open('./output/' + dataname + '_results.out', 'a')
        #print("The experimental results", file=file_out)
        seed_value = 1
        random.seed(seed_value)  # 设置 random 模块的随机种子
        np.random.seed(seed_value)  # 设置 numpy 模块的随机种子
        torch.manual_seed(seed_value)  # 设置 PyTorch 中 CPU 的随机种子
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():  # 如果可以使用 CUDA，设置随机种子
            torch.cuda.manual_seed(seed_value)  # 设置 PyTorch 中 GPU 的随机种子
            torch.backends.cudnn.deterministic = True  # 使用确定性算法，使每次运行结果一样
            torch.backends.cudnn.benchmark = False  # 不使用自动寻找最优算法加速运算

        ld1 = ld1
        ld2 = ld2
        ld3 = ld3
        ld4 = ld4
       # print("lambda_1: ", ld1, "lambda_2: ", ld2, "lambda_3: ", ld3, "lambda_4: ", ld4, file=file_out)
        # model = SDCN(500, 500, 2000, 2000, 500, 500,
        #              n_input=adata.obsm['h_tem'].shape[1],
        #              n_z=args.n_z,
        #              n_clusters=args.n_clusters,
        #              v=1.0).cuda()  # .to(device)

        #print(num_net_parameter(model))
        optimizer = Adam(self.parameters(), lr=lr)
        data = torch.Tensor(dataset.x).cuda()  # .to(device)
        y = dataset.y
        print("######################data############################",data)
        with torch.no_grad():
            _, _, _, _, z = self.ae(data)

        iters10_kmeans_iter_F = []
        iters10_NMI_iter_F = []
        iters10_ARI_iter_F = []
        iters10_F1_iter_F = []
        print("######################z############################",z)
        z_1st = z
        # kmeans = KMeans(n_clusters=self.n_clusters, n_init=7, random_state=1)
        # y_pred = kmeans.fit_predict(z_1st.data.cpu().numpy())

        # #y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        # self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()  # .to(device)
        
        # y_pred = np.array(adata.obs['label_tem'])
        # y_pred = torch.tensor(y_pred).long().to(device)
       
        y_pred = adata.obs['label_tem']
        centers = cluster_center(z_1st, labels = y_pred)
        self.cluster_layer.data = torch.tensor(centers).to(device)

        acc, nmi, ari, f1 = eva(y, y_pred, 'pae')
        print("acc:", acc, "nmi:", nmi, "ari:", ari, "f1:", f1)

        # y_pred = adata.obs['label_tem']
        # acc, nmi, ari, f1 = eva(y, y_pred, 'pae')
        # #acc, nmi, ari, f1, new_predict = eva_2(y, y_pred, 'pae')
        # #y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        # #self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()  # .to(device)
        # from utils import cluster_center
        # #new_predict = torch.tensor(np.array(new_predict))
        # #cluster_center = cluster_center(z_1st, new_predict)
        # cluster_center = cluster_center(z_1st, y_pred)
        # self.cluster_layer.data = torch.tensor(cluster_center).to(device)

        kmeans_iter_F = []
        NMI_iter_F = []
        ARI_iter_F = []
        F1_iter_F = []

        for epoch in range(epoch):

            if epoch % 1 == 0:
                x_bar, q, pred, _, net_output_l, pl_loss, z_F, net_output = self.inference(data, adj)
                p = target_distribution(pred.data)
                res4 = z_F.data.cpu().numpy().argmax(1)
                acc, nmi, ari, f1 = eva(y, res4, str(epoch) + 'F')
                kmeans_iter_F.append(acc)
                NMI_iter_F.append(nmi)
                ARI_iter_F.append(ari)
                F1_iter_F.append(f1)

                KL_QP = F.kl_div(q.log(), p, reduction='batchmean')
                KL_ZP = F.kl_div(pred.log(), p, reduction='batchmean')
                re_loss = F.mse_loss(x_bar, data)
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
        print("#################result_eachview####################")
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
        return x_bar, net_output_l, res4, z, net_output

    # #adata.obsm['X_tem'] =  x_bar.cpu().detach().numpy()
    #     adata.obsm['h_tem'] = net_output.cpu().detach().numpy()
    # #adata.obsm['X_tem'] = net_output.cpu().detach().numpy()
    #     adata.obs['label_tem'] = res4
    #     adata.obs['label_tem'] = adata.obs['label_tem'].astype('category')
    #     np.savetxt("./output/MDAGC/h_scc_151673_net.csv", net_output.cpu().detach().numpy(), delimiter=',')
    # #np.savetxt("h_scc_151673_z_F.csv", z_F.cpu().detach().numpy(), delimiter=',')
    #     np.savetxt("./output/MDAGC/pred_scc_151673.csv", res4, delimiter=',')





class fusion_eachview(nn.Module):
    def __init__(self, n_clusters, n_z, n_feature) -> None:
        super(fusion_eachview, self).__init__()
        self.agnn_f = GNNLayer(4 * n_feature, n_feature)
        self.agnn_n = GNNLayer(n_feature, n_clusters)
        self.mlp_n = MLP_n(4 * n_clusters)
        self.mlp_f = MLP_n(4 * n_feature)
        self.mlp_ZQ = MLP_ZQ(2 * n_clusters)
        self.n_clusters = n_clusters
        self.n_feature = n_feature
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z ))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = 1

    def forward(self, x, z, net_output_1, net_output_2, net_output_3, net_output_4, adj2):
        x_array = list(np.shape(x))
        n_x = x_array[0]
        #print("net_output1:",net_output_1.shape,"net_output2:",net_output_2.shape,"net_output3:",net_output_3.shape,"net_output4:",net_output_4.shape)
        wn = self.mlp_f(torch.cat((net_output_1, net_output_2, net_output_3, net_output_4), 1))
        wn = F.normalize(wn, p=2)

        wn0 = torch.reshape(wn[:, 0], [n_x, 1])
        wn1 = torch.reshape(wn[:, 1], [n_x, 1])
        wn2 = torch.reshape(wn[:, 2], [n_x, 1])
        wn3 = torch.reshape(wn[:, 3], [n_x, 1])

        # 2️⃣ [Z+H]
        tile_wn0 = wn0.repeat(1, self.n_feature)
        tile_wn1 = wn1.repeat(1, self.n_feature)
        tile_wn2 = wn2.repeat(1, self.n_feature)
        tile_wn3 = wn3.repeat(1, self.n_feature)
        # tile_wn0 = wn0.repeat(1, self.n_clusters)
        # tile_wn1 = wn1.repeat(1, self.n_clusters)
        # tile_wn2 = wn2.repeat(1, self.n_clusters)
        # tile_wn3 = wn3.repeat(1, self.n_clusters)
        net_output_f = torch.cat((tile_wn0 * net_output_1, tile_wn1 * net_output_2, tile_wn2 * net_output_3, tile_wn3 * net_output_4), 1)
        net_output_f = self.agnn_f(net_output_f, adj2 , active=False)
        net_output_l = self.agnn_n(net_output_f, adj2,active = False )
        predict = torch.softmax(net_output_l, dim=1)
        ###z_F = alpha * z_F_1 + beta * z_F_2 + gamma * z_F_3

        # Dual Self-supervision(都可用？？以下3行）
       # q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # print("z:",z)
        # print("z:",z.shape)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

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
        return  q, predict, z, net_output_l, pseudo_label_loss, z_F, net_output_f

def BCE(out, tar, weight):
    eps = 1e-12  # The case without eps could lead to the `nan' situation
    l_n = weight * (tar * (torch.log(out + eps)) + (1 - tar) * (torch.log(1 - out + eps)))
    l = -torch.sum(l_n) / torch.numel(out)
    return l


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def num_net_parameter(net):
    all_num = sum(i.numel() for i in net.parameters())
    print('[The network parameters]', all_num)


