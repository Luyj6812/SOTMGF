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
from layer import GATLayer


###Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, n_head):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size,  # 输入维度
                                                        nhead=n_head, dim_feedforward=1024)             # 注意力头数
        self.encoder = nn.TransformerEncoder(self.encoder_layer,             # 编码层
                                             num_layers=3)                   # 层数
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_size,nhead=n_head, dim_feedforward=1024)
        self.dc = nn.TransformerDecoder(decoder_layer=self.decoder_layer ,num_layers=3)
        self.fc = nn.Linear(input_size,
                             num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)    # 增加一个维度，变成(batch_size, 1, input_size)的形状
        x = self.encoder(x)   # 输入Transformer编码器进行编码
        return self.dc(torch.rand(x.shape).cuda(),x).squeeze(1),   x,   None,    self.fc(x.squeeze(1))

class TransformerModel_1(nn.Module):
    def __init__(self, input_size, num_classes, n_head):
        super(TransformerModel_1, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size,  # 输入维度
                                                        nhead=n_head, dim_feedforward=48)             # 注意力头数
        self.encoder = nn.TransformerEncoder(self.encoder_layer,             # 编码层
                                             num_layers=3)                   # 层数
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_size,nhead=n_head, dim_feedforward=48)
        self.dc = nn.TransformerDecoder(decoder_layer=self.decoder_layer ,num_layers=3)
        self.fc = nn.Linear(input_size,
                             num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)    # 增加一个维度，变成(batch_size, 1, input_size)的形状
        x = self.encoder(x)   # 输入Transformer编码器进行编码
        return self.dc(torch.rand(x.shape).cuda(),x).squeeze(1),   x,   None,    self.fc(x.squeeze(1))

class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        #self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))## pretrain_path

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


class DECT(nn.Module):
    def __init__(self, input_size, num_classes, embedding_size, n_head, v = 1):
        super(DECT, self).__init__()
        self.transformer = TransformerModel(input_size, num_classes, n_head)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_classes, embedding_size)) ####2000daiding!!!!!!!
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v
    
    def forward(self, x):
        reconstruction, h, logvar, predicted_train = self.transformer(x)
        q = self.get_Q(h)
        return reconstruction, h, logvar, predicted_train, q

    def get_Q(self, h):
        #q = 1.0 / (1.0 + torch.sum(torch.pow(h.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = 1.0 / (1.0 + torch.sum(torch.pow(h - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


class DECT_1(nn.Module):
    def __init__(self, input_size, num_classes, embedding_size, n_head, v = 1):
        super(DECT_1, self).__init__()
        self.transformer = TransformerModel_1(input_size, num_classes, n_head)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_classes, embedding_size)) ####2000daiding!!!!!!!
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v
    
    def forward(self, x):
        reconstruction, h, logvar, predicted_train = self.transformer(x)
        q = self.get_Q(h)
        return reconstruction, h, logvar, predicted_train, q

    def get_Q(self, h):
        #q = 1.0 / (1.0 + torch.sum(torch.pow(h.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = 1.0 / (1.0 + torch.sum(torch.pow(h - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
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
                 n_input, n_z, n_clusters, v=1):
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
        #self.agnn_z = GNNLayer(3020, n_clusters)
        self.agnn_z = GNNLayer(3000+ 2*n_z, n_clusters)
        self.agnn_n = GNNLayer(4 * n_clusters, n_clusters)

        #self.mlp = MLP_L(3020)
        self.mlp = MLP_L(3000+ 2*n_z)
        self.mlp_n = MLP_n(4 * n_clusters)
        # attention on [z_i, h_i]
        self.mlp1 = MLP_1(2 * n_enc_1)
        self.mlp2 = MLP_2(2 * n_enc_2)
        self.mlp3 = MLP_3(2 * n_enc_3)

        self.mlp_ZQ = MLP_ZQ(2 * n_clusters)
        self.n_z = n_z
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

        self.n_clusters = n_clusters
        self.NFeature = n_clusters

    def forward(self, x, adj1, adj2, adj3, adj4):
        # DAE Module
        x_bar, h1, h2, h3, z = self.ae(x)

        x_array = list(np.shape(x))
        n_x = x_array[0]
        ##########################################################adj1###########################################################
        # HWF
        # 5️⃣ GCN[p_1*Z + p_2*H]
        # z1
        z1_1 = self.agnn_0(x, adj1)
        # z2
        p1_1 = self.mlp1(torch.cat((h1, z1_1), 1))
        p1_1 = F.normalize(p1_1, p=2)
        p11_1 = torch.reshape(p1_1[:, 0], [n_x, 1])
        p12_1 = torch.reshape(p1_1[:, 1], [n_x, 1])
        p11_broadcast_1 = p11_1.repeat(1, 500)
        p12_broadcast_1 = p12_1.repeat(1, 500)
        z2_1 = self.agnn_1(p11_broadcast_1.mul(z1_1) + p12_broadcast_1.mul(h1), adj1)
        # z3
        p2_1 = self.mlp2(torch.cat((h2, z2_1), 1))
        p2_1 = F.normalize(p2_1, p=2)
        p21_1 = torch.reshape(p2_1[:, 0], [n_x, 1])
        p22_1 = torch.reshape(p2_1[:, 1], [n_x, 1])
        p21_broadcast_1 = p21_1.repeat(1, 500)
        p22_broadcast_1 = p22_1.repeat(1, 500)
        z3_1 = self.agnn_2(p21_broadcast_1.mul(z2_1) + p22_broadcast_1.mul(h2), adj1)
        # z4
        p3_1 = self.mlp3(torch.cat((h3, z3_1), 1))  # self.mlp3(h2)
        p3_1 = F.normalize(p3_1, p=2)
        p31_1 = torch.reshape(p3_1[:, 0], [n_x, 1])
        p32_1 = torch.reshape(p3_1[:, 1], [n_x, 1])
        p31_broadcast_1 = p31_1.repeat(1, 2000)
        p32_broadcast_1 = p32_1.repeat(1, 2000)
        z4_1 = self.agnn_3(p31_broadcast_1.mul(z3_1) + p32_broadcast_1.mul(h3), adj1)
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
        # tile_w3_1 = w3_1.repeat(1, 10)
        # tile_w4_1 = w4_1.repeat(1, 10)
        tile_w3_1 = w3_1.repeat(1, self.n_z)
        tile_w4_1 = w4_1.repeat(1, self.n_z)

        # 2️⃣ concat
        net_output_1 = torch.cat(
            (tile_w0_1.mul(z1_1), tile_w1_1.mul(z2_1), tile_w2_1.mul(z3_1), tile_w3_1.mul(z4_1), tile_w4_1.mul(z)), 1)
        net_output_1 = self.agnn_z(net_output_1, adj1, active=False)

        ##########################################################adj2###########################################################
        # HWF
        # 5️⃣ GCN[p_1*Z + p_2*H]
        # z1
        z1_2 = self.agnn_0(x, adj2)
        # z2
        p1_2 = self.mlp1(torch.cat((h1, z1_2), 1))
        p1_2 = F.normalize(p1_2, p=2)
        p11_2 = torch.reshape(p1_2[:, 0], [n_x, 1])
        p12_2 = torch.reshape(p1_2[:, 1], [n_x, 1])
        p11_broadcast_2 = p11_2.repeat(1, 500)
        p12_broadcast_2 = p12_2.repeat(1, 500)
        z2_2 = self.agnn_1(p11_broadcast_2.mul(z1_2) + p12_broadcast_2.mul(h1), adj2)
        # z3
        p2_2 = self.mlp2(torch.cat((h2, z2_2), 1))
        p2_2 = F.normalize(p2_2, p=2)
        p21_2 = torch.reshape(p2_2[:, 0], [n_x, 1])
        p22_2 = torch.reshape(p2_2[:, 1], [n_x, 1])
        p21_broadcast_2 = p21_2.repeat(1, 500)
        p22_broadcast_2 = p22_2.repeat(1, 500)
        z3_2 = self.agnn_2(p21_broadcast_2.mul(z2_2) + p22_broadcast_2.mul(h2), adj2)
        # z4
        p3_2 = self.mlp3(torch.cat((h3, z3_2), 1))  # self.mlp3(h2)
        p3_2 = F.normalize(p3_2, p=2)
        p31_2 = torch.reshape(p3_2[:, 0], [n_x, 1])
        p32_2 = torch.reshape(p3_2[:, 1], [n_x, 1])
        p31_broadcast_2 = p31_2.repeat(1, 2000)
        p32_broadcast_2 = p32_2.repeat(1, 2000)
        z4_2 = self.agnn_3(p31_broadcast_2.mul(z3_2) + p32_broadcast_2.mul(h3), adj2)
        # SWF
        w_2 = self.mlp(torch.cat((z1_2, z2_2, z3_2, z4_2, z), 1))
        w_2 = F.normalize(w_2, p=2)

        w0_2 = torch.reshape(w_2[:, 0], [n_x, 1])
        w1_2 = torch.reshape(w_2[:, 1], [n_x, 1])
        w2_2 = torch.reshape(w_2[:, 2], [n_x, 1])
        w3_2 = torch.reshape(w_2[:, 3], [n_x, 1])
        w4_2 = torch.reshape(w_2[:, 4], [n_x, 1])

        # 2️⃣ [Z+H]
        tile_w0_2 = w0_2.repeat(1, 500)
        tile_w1_2 = w1_2.repeat(1, 500)
        tile_w2_2 = w2_2.repeat(1, 2000)
        # tile_w3_2 = w3_2.repeat(1, 10)
        # tile_w4_2 = w4_2.repeat(1, 10)
        tile_w3_2 = w3_2.repeat(1, self.n_z)
        tile_w4_2 = w4_2.repeat(1, self.n_z)

        # 2️⃣ concat
        net_output_2 = torch.cat(
            (tile_w0_2.mul(z1_2), tile_w1_2.mul(z2_2), tile_w2_2.mul(z3_2), tile_w3_2.mul(z4_2), tile_w4_2.mul(z)), 1)
        net_output_2 = self.agnn_z(net_output_2, adj2, active=False)

        ##########################################################adj3###########################################################
        # HWF
        # 5️⃣ GCN[p_1*Z + p_2*H]
        # z1
        z1_3 = self.agnn_0(x, adj3)
        # z2
        p1_3 = self.mlp1(torch.cat((h1, z1_3), 1))
        p1_3 = F.normalize(p1_3, p=2)
        p11_3 = torch.reshape(p1_3[:, 0], [n_x, 1])
        p12_3 = torch.reshape(p1_3[:, 1], [n_x, 1])
        p11_broadcast_3 = p11_3.repeat(1, 500)
        p12_broadcast_3 = p12_3.repeat(1, 500)
        z2_3 = self.agnn_1(p11_broadcast_3.mul(z1_3) + p12_broadcast_3.mul(h1), adj3)
        # z3
        p2_3 = self.mlp2(torch.cat((h2, z2_3), 1))
        p2_3 = F.normalize(p2_3, p=2)
        p21_3 = torch.reshape(p2_3[:, 0], [n_x, 1])
        p22_3 = torch.reshape(p2_3[:, 1], [n_x, 1])
        p21_broadcast_3 = p21_3.repeat(1, 500)
        p22_broadcast_3 = p22_3.repeat(1, 500)
        z3_3 = self.agnn_2(p21_broadcast_3.mul(z2_3) + p22_broadcast_3.mul(h2), adj3)
        # z4
        p3_3 = self.mlp3(torch.cat((h3, z3_3), 1))  # self.mlp3(h2)
        p3_3 = F.normalize(p3_3, p=2)
        p31_3 = torch.reshape(p3_3[:, 0], [n_x, 1])
        p32_3 = torch.reshape(p3_3[:, 1], [n_x, 1])
        p31_broadcast_3 = p31_3.repeat(1, 2000)
        p32_broadcast_3 = p32_3.repeat(1, 2000)
        z4_3 = self.agnn_3(p31_broadcast_3.mul(z3_3) + p32_broadcast_3.mul(h3), adj3)
        # SWF
        w_3 = self.mlp(torch.cat((z1_3, z2_3, z3_3, z4_3, z), 1))
        w_3 = F.normalize(w_3, p=2)

        w0_3 = torch.reshape(w_3[:, 0], [n_x, 1])
        w1_3 = torch.reshape(w_3[:, 1], [n_x, 1])
        w2_3 = torch.reshape(w_3[:, 2], [n_x, 1])
        w3_3 = torch.reshape(w_3[:, 3], [n_x, 1])
        w4_3 = torch.reshape(w_3[:, 4], [n_x, 1])

        # 2️⃣ [Z+H]
        tile_w0_3 = w0_3.repeat(1, 500)
        tile_w1_3 = w1_3.repeat(1, 500)
        tile_w2_3 = w2_3.repeat(1, 2000)
        # tile_w3_3 = w3_3.repeat(1, 10)
        # tile_w4_3 = w4_3.repeat(1, 10)
        tile_w3_3 = w3_3.repeat(1, self.n_z)
        tile_w4_3 = w4_3.repeat(1, self.n_z)

        # 2️⃣ concat
        net_output_3 = torch.cat(
            (tile_w0_3.mul(z1_3), tile_w1_3.mul(z2_3), tile_w2_3.mul(z3_3), tile_w3_3.mul(z4_3), tile_w4_3.mul(z)), 1)
        net_output_3 = self.agnn_z(net_output_3, adj3, active=False)

        ##########################################################adj4###########################################################
        # HWF
        # 5️⃣ GCN[p_1*Z + p_2*H]
        # z1
        z1_4 = self.agnn_0(x, adj4)
        # z2
        p1_4 = self.mlp1(torch.cat((h1, z1_4), 1))
        p1_4 = F.normalize(p1_4, p=2)
        p11_4 = torch.reshape(p1_4[:, 0], [n_x, 1])
        p12_4 = torch.reshape(p1_4[:, 1], [n_x, 1])
        p11_broadcast_4 = p11_4.repeat(1, 500)
        p12_broadcast_4 = p12_4.repeat(1, 500)
        z2_4 = self.agnn_1(p11_broadcast_4.mul(z1_4) + p12_broadcast_4.mul(h1), adj4)
        # z3
        p2_4 = self.mlp2(torch.cat((h2, z2_4), 1))
        p2_4 = F.normalize(p2_4, p=2)
        p21_4 = torch.reshape(p2_4[:, 0], [n_x, 1])
        p22_4 = torch.reshape(p2_4[:, 1], [n_x, 1])
        p21_broadcast_4 = p21_4.repeat(1, 500)
        p22_broadcast_4 = p22_4.repeat(1, 500)
        z3_4 = self.agnn_2(p21_broadcast_4.mul(z2_4) + p22_broadcast_4.mul(h2), adj4)
        # z4
        p3_4 = self.mlp3(torch.cat((h3, z3_4), 1))  # self.mlp3(h2)
        p3_4 = F.normalize(p3_4, p=2)
        p31_4 = torch.reshape(p3_4[:, 0], [n_x, 1])
        p32_4 = torch.reshape(p3_4[:, 1], [n_x, 1])
        p31_broadcast_4 = p31_4.repeat(1, 2000)
        p32_broadcast_4 = p32_4.repeat(1, 2000)
        z4_4 = self.agnn_3(p31_broadcast_4.mul(z3_4) + p32_broadcast_4.mul(h3), adj4)
        # SWF
        w_4 = self.mlp(torch.cat((z1_4, z2_4, z3_4, z4_4, z), 1))
        w_4 = F.normalize(w_4, p=2)

        w0_4 = torch.reshape(w_4[:, 0], [n_x, 1])
        w1_4 = torch.reshape(w_4[:, 1], [n_x, 1])
        w2_4 = torch.reshape(w_4[:, 2], [n_x, 1])
        w3_4 = torch.reshape(w_4[:, 3], [n_x, 1])
        w4_4 = torch.reshape(w_4[:, 4], [n_x, 1])

        # 2️⃣ [Z+H]
        tile_w0_4 = w0_4.repeat(1, 500)
        tile_w1_4 = w1_4.repeat(1, 500)
        tile_w2_4 = w2_4.repeat(1, 2000)
        # tile_w3_4 = w3_4.repeat(1, 10)
        # tile_w4_4 = w4_4.repeat(1, 10)
        tile_w3_4 = w3_4.repeat(1, self.n_z)
        tile_w4_4 = w4_4.repeat(1, self.n_z)

        # 2️⃣ concat
        net_output_4 = torch.cat(
            (tile_w0_4.mul(z1_3), tile_w1_4.mul(z2_3), tile_w2_4.mul(z3_3), tile_w3_4.mul(z4_3), tile_w4_4.mul(z)), 1)
        net_output_4 = self.agnn_z(net_output_4, adj4, active=False)
        ########################################################################################################
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

        return x_bar, q, predict, z, net_output, pseudo_label_loss, z_F

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