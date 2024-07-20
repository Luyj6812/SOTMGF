import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_EDC import do_clustering, seed_everything, RunKmeans, AverageMeter, normalized_mutual_info_score, accuracy, save_dict
#from datasets import MNISTDataset, BDGPDataset, CCVDataset, HandWriteDataset
#from models import Net4Mnist, Net4BDGP, Net4HW
from torch.utils.data import Dataset
import scipy
import os
import math
from itertools import chain
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F



class BaseNet(nn.Module):

    def __init__(self):
        super().__init__()

    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        return init_fun


class Netfusion(BaseNet):

    def __init__(self, input_A, input_B, input_C, input_D, nz, n_view, n_clusters):
        super().__init__()
        self.input_A = input_A
        self.input_B = input_B
        self.input_C = input_C
        self.input_D = input_D
        self.nz = nz
        self.encoder1 = nn.Sequential(
            nn.Linear(input_A, 500, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(500, 300, bias=True),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(300, self.nz//n_view, bias=True),
            nn.ReLU(True),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_B, 300, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(300, self.nz//n_view, bias=True),
            nn.ReLU(True),
        )
        self.encoder3 = nn.Sequential(
            nn.Linear(input_C, 300, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(300, self.nz//n_view, bias=True),
            nn.ReLU(True),
        )
        self.encoder4 = nn.Sequential(
            nn.Linear(input_D, 300, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(300, self.nz//n_view, bias=True),
            nn.ReLU(True),
        )

        # Add Transformer.
        self.trans_enc = nn.TransformerEncoderLayer(d_model=self.nz, nhead=1, dim_feedforward=256)
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1)

        self.cls_layer = nn.Sequential(
            nn.Linear(self.nz, n_clusters),
            nn.Sigmoid()
        )

        self.layer4 = nn.Linear(self.nz, 300)
        self.layer5_1 = nn.Linear(300, 500)
        self.layer6_1 = nn.Linear(500, input_A)
        self.layer6_2 = nn.Linear(300, input_B)
        self.layer6_3 = nn.Linear(300, input_C)
        self.layer6_4 = nn.Linear(300, input_D)
        self.drop = 0.5

        self.sigmoid = nn.Sigmoid()

        self.apply(self.weights_init('xavier'))
        self.flatten = nn.Flatten()
        self.recon_criterion = nn.BCELoss(reduction='none')
        self.cls_criterion = nn.CrossEntropyLoss()


    def forward(self, Xs):
        x1, x2, x3, x4 = Xs
        x1 = self.encoder1(x1.view(-1, self.input_A)).unsqueeze(1)
        x2 = self.encoder2(x2.view(-1, self.input_B)).unsqueeze(1)
        x3 = self.encoder3(x3.view(-1, self.input_C)).unsqueeze(1)
        x4 = self.encoder4(x4.view(-1, self.input_D)).unsqueeze(1)

        x = self.extract_layers(torch.cat((x1, x2, x3, x4), 2))
        # out: (batch size, length, d_model)
        x = x.transpose(0, 1)
        # mean pooling
        x = x.mean(dim=0)
        y = self.cls_layer(x)

        return y

    def decoder(self, latent):
        x = F.dropout(F.relu(self.layer4(latent)), self.drop)

        out1 = F.relu(self.layer5_1(x))
        out1 = self.sigmoid(self.layer6_1(out1))
        out2 = self.sigmoid(self.layer6_2(x))
        out3 = self.sigmoid(self.layer6_3(x))
        out4 = self.sigmoid(self.layer6_4(x))

        return out1.view(-1, self.input_A), out2.view(-1, self.input_B), out3.view(-1, self.input_C), out4.view(-1, self.input_D)

    def get_loss(self, Xs, labels=None):
        if labels is not None:
            y = self(Xs)
            cls_loss = self.cls_criterion(y, labels)
            return cls_loss
        else:
            latent = self.test_commonZ(Xs)
            recon1, recon2, recon3, recon4 = self.decoder(latent)
            recon_loss = 0.3 * self.recon_criterion(recon1, Xs[0]).mean(0).sum() + \
                         0.3 * self.recon_criterion(recon2, Xs[1]).mean(0).sum() + \
                         0.2 * self.recon_criterion(recon3, Xs[2]).mean(0).sum() + \
                         0.2 * self.recon_criterion(recon4, Xs[3]).mean(0).sum()                         

            return recon_loss

    def test_commonZ(self, Xs):
        x1, x2, x3, x4 = Xs
        x1 = self.encoder1(x1.view(-1, self.input_A)).unsqueeze(1)
        x2 = self.encoder2(x2.view(-1, self.input_B)).unsqueeze(1)
        x3 = self.encoder3(x3.view(-1, self.input_C)).unsqueeze(1)
        x4 = self.encoder4(x4.view(-1, self.input_D)).unsqueeze(1)

        x = self.extract_layers(torch.cat((x1, x2, x3, x4), 2))
        # out: (batch size, length, d_model)
        x = x.transpose(0, 1)
        # mean pooling
        latent = x.mean(dim=0)
        return latent

    def get_cls_optimizer(self):
        self.cls_optimizer = torch.optim.SGD(chain(self.encoder1.parameters(), self.encoder2.parameters(), 
                                                   self.encoder3.parameters(), self.encoder4.parameters(),
                                                   self.extract_layers.parameters(), self.cls_layer.parameters()),
                                             lr=1e-3,
                                             momentum=0.9,
                                             weight_decay=5e-4)
        return self.cls_optimizer

    def get_recon_optimizer(self):
        self.recon_optimizer = torch.optim.SGD(self.parameters(),
                                               lr=1e-3,
                                               momentum=0.9,
                                               weight_decay=5e-4,
                                               )
        return self.recon_optimizer
    

class Dataset(Dataset):
    """
    BDGP dataset
    Refer by:
    Xiao Cai, Hua Wang, Heng Huang, and Chris Ding. Joint stage recognition and anatomical annotation of drosophila gene expression patterns. Bioinformatics, 28(12):i16– i24, 2012.
    """
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 need_target=False):
        self.root = root
        self.transform = transform
        self.train = train
        self.need_target = need_target
        self.paried_num = int(2500 * 1)
        data_0 = scipy.io.loadmat(os.path.join(self.root, 'paired_a2500all.mat'))
        data_dict = dict(data_0)
        self.data_a = torch.Tensor(np.array(data_dict['xpaired']))
        data_2 = scipy.io.loadmat(os.path.join(self.root, 'paired_b2500all.mat'))
        data_dict = dict(data_2)
        self.data_b = torch.Tensor(np.array(data_dict['ypaired']))

        data_3 = scipy.io.loadmat(os.path.join(self.root, 'paired_a2500all.mat'))
        data_dict = dict(data_3)
        self.data_c = torch.Tensor(np.array(data_dict['xpaired']))


        data_4 = scipy.io.loadmat(os.path.join(self.root, 'paired_b2500all.mat'))
        data_dict = dict(data_4)
        self.data_d = torch.Tensor(np.array(data_dict['ypaired']))


        labels = scipy.io.loadmat(os.path.join(self.root, 'label.mat'))
        labels = dict(labels)
        labels = np.array(labels['label'])
        self.labels = torch.LongTensor(labels).reshape(-1, )


    def __getitem__(self, index):
        img_a, img_b, img_c, img_d = self.data_a[index], self.data_b[index], self.data_c[index], self.data_d[index]
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
            img_c = self.transform(img_c)
            img_d = self.transform(img_d)
        if self.need_target:
            return img_a, img_b, img_c, img_d, self.labels[index]
        else:
            return img_a, img_b, img_c, img_d

    def __len__(self):
        return self.paried_num
    
class Dataset_EMC(Dataset):
    """
    BDGP dataset
    Refer by:
    Xiao Cai, Hua Wang, Heng Huang, and Chris Ding. Joint stage recognition and anatomical annotation of drosophila gene expression patterns. Bioinformatics, 28(12):i16– i24, 2012.
    """
    def __init__(self, adata, net_output1, net_output2, net_output3, net_output4,
                 train=True,
                 transform=None,
                 need_target=False):

        self.transform = transform
        self.train = train
        self.need_target = need_target
        self.paried_num = int(adata.obsm['h_tem'].shape[0] * 1)

        # self.data_a = torch.Tensor(np.array(net_output1))
        # self.data_b = torch.Tensor(np.array(net_output2))
        # self.data_c = torch.Tensor(np.array(net_output3))
        # self.data_d = torch.Tensor(np.array(net_output4))
        self.data_a = net_output1.cpu().detach()
        self.data_b = net_output2.cpu().detach()
        self.data_c = net_output3.cpu().detach()
        self.data_d = net_output4.cpu().detach()
        labels = np.array(adata.obs['label']).reshape(adata.obsm['h_tem'].shape[0],1)

        self.labels = torch.LongTensor(labels).reshape(-1, )


    def __getitem__(self, index):
        img_a, img_b, img_c, img_d = self.data_a[index], self.data_b[index], self.data_c[index], self.data_d[index]
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
            img_c = self.transform(img_c)
            img_d = self.transform(img_d)
        if self.need_target:
            return img_a, img_b, img_c, img_d, self.labels[index]
        else:
            return img_a, img_b, img_c, img_d

    def __len__(self):
        return self.paried_num

def create_data_loader(datasets, batch_size, num_workers, init=False, labels=None):
    if init:
        return DataLoader(datasets, batch_size=batch_size, num_workers=num_workers)
    if labels is not None:
        datasets.labels = labels
        datasets.need_target = True
        return DataLoader(datasets, batch_size=batch_size, num_workers=num_workers)
    else:
        datasets.need_target = False
        return DataLoader(datasets, batch_size=batch_size, num_workers=num_workers)


def extract_features(train_loader, model, device):
    model.eval()
    commonZ = []
    with torch.no_grad():
        for data in train_loader:
            Xs, y = [d.to(device) for d in data[:-1]], data[-1].to(device)
            common = model.test_commonZ(Xs)
            commonZ.extend(common.detach().cpu().numpy().tolist())

    commonZ = np.array(commonZ)
    return commonZ


def validate(data_loader, model, labels_holder, n_clusters, device):
    commonZ = extract_features(data_loader, model, device)
    acc, nmi, pur, ari = RunKmeans(commonZ, labels_holder['labels_gt'], K=n_clusters, cv=1)
    return acc, nmi, pur, ari


def unsupervised_clustering_step(model, train_loader, num_workers, labels_holder, n_clusters, device):
    print('[Pesudo labels]...')
    features = extract_features(train_loader, model, device)

    if 'labels' in labels_holder:
        labels_holder['labels_prev_step'] = labels_holder['labels']

    if 'score' not in labels_holder:
        labels_holder['score'] = -1

    labels = do_clustering(features, n_clusters)
    labels_holder['labels'] = labels
    nmi = 0
    # score = unsupervised_measures(features, labels)
    # print(labels.shape, labels_holder['labels_gt'].shape)

    nmi_gt = normalized_mutual_info_score(labels_holder['labels_gt'], labels)
    print('NMI t / GT = {:.4f}'.format(nmi_gt))

    if 'labels_prev_step' in labels_holder:
        nmi = normalized_mutual_info_score(labels_holder['labels_prev_step'], labels)
        print('NMI t / t-1 = {:.4f}'.format(nmi))

    train_loader = create_data_loader(train_loader.dataset, train_loader.batch_size, num_workers, labels=labels)

    return train_loader, nmi_gt, nmi


def train_unsupervised(train_loader, model, optimizer, epoch, max_steps, device, tag='unsupervised', verbose=1):
    losses = AverageMeter()

    model.train()
    if verbose == 1:
        pbar = tqdm(total=len(train_loader),
                    ncols=0, desc=f'[{tag.upper()}]', unit=" batch")
    for data in train_loader:
        # measure data loading time
        Xs = [d.to(device) for d in data[:-1]]

        loss = model.get_loss(Xs)
        losses.update(loss.item(), Xs[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if verbose == 1:
            pbar.update()
            pbar.set_postfix(
                loss=f"{losses.avg:.4f}",
                epoch=epoch + 1,
                max_steps=max_steps
            )
    if verbose == 1:
        pbar.close()

    return losses.avg


def train(train_loader, model, optimizer, epoch, max_steps, device, tag='train', verbose=1):
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    if verbose == 1:
        pbar = tqdm(total=len(train_loader), ncols=0, desc=f'[{tag.upper()}]', unit=" batch")
    for data in train_loader:
        # measure data loading time
        Xs, target = [d.to(device) for d in data[:-1]], data[-1].to(device)

        # compute output
        outputs = model(Xs)
        loss = model.get_loss(Xs, target)

        # measure accuracy and record loss
        prec1 = accuracy(outputs, target, topk=(1,))[0]  # returns tensors!
        losses.update(loss.item(), Xs[0].size(0))
        acc.update(prec1.item(), Xs[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if verbose == 1:
            pbar.update()
            pbar.set_postfix(
                loss=f"{losses.avg:.4f}",
                Acc=f"{acc.avg:.4f}",
                epoch=epoch + 1,
                max_steps=max_steps
            )

    if verbose == 1:
        pbar.close()

    return acc.avg, losses.avg


def main(Net, mparams, datasets, batch_size=128,
         n_clusters=10, seed=10,
         max_steps=1000, recluster_epoch=1,
         validate_epoch=1, max_unsupervised_steps=3, max_supervised_steps=2, model_path='./', verbose=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('CudNN:', torch.backends.cudnn.version())
    print('Run on {} GPUs'.format(torch.cuda.device_count()))

    start_epoch = 0
    best_nmi = 0
    ### Data loading ###
    num_workers = 4

    print('[TRAIN]...')
    seed_everything(seed)
    model = Net(*mparams).to(device)
    cls_optimizer = model.get_cls_optimizer()
    recon_optimizer = model.get_recon_optimizer()
    ###############################################################################

    labels_holder = {}  # utility container to save labels from the previous clustering step
    train_loader = create_data_loader(datasets, batch_size, num_workers, init=True, labels=None)
    labels_holder['labels_gt'] = train_loader.dataset.labels.numpy()
    history = {}
    # Training Start

    history['best_acc'] = 0
    history['nmi_gt'] = []
    history['nmi_t_1'] = []
    history['recon_loss'] = []
    history['cls_loss'] = []
    history['cluster_result'] = []
    best_score = 0

    for epoch in range(max_steps):
        nmi_gt = None

        for u_epoch in range(max_unsupervised_steps):
            loss_avg = train_unsupervised(train_loader, model, recon_optimizer, u_epoch, max_unsupervised_steps,
                                          device, verbose=verbose)
            history['recon_loss'].append(loss_avg)

        if epoch == start_epoch or epoch % recluster_epoch == 0:
            train_loader, nmi_gt, nmi_t_1 = \
                unsupervised_clustering_step(model, train_loader, num_workers, labels_holder, n_clusters, device)
            history['nmi_gt'].append(nmi_gt)
            history['nmi_t_1'].append(nmi_t_1)

        for u_epoch in range(max_supervised_steps):
            acc_avg, loss_avg = train(train_loader, model, cls_optimizer, u_epoch, max_supervised_steps, device, verbose=verbose)
            history['cls_loss'].append(loss_avg)

        if (epoch + 1) % validate_epoch == 0:
            acc, nmi, pur, ari = validate(train_loader, model, labels_holder, n_clusters, device)
            history['cluster_result'].append((acc, nmi, pur, ari))
            if acc > best_score:
                best_score = acc
                history['best_acc'] = best_score
                torch.save(model.state_dict(), model_path)
        print(f"{'-' * 20} {seed}: best score: {best_score} {'-' * 20}")
    return history


