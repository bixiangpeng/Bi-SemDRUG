import torch
import torch.nn as nn
from .KGGNN import NHGNN, HHGNN
from torch_geometric.utils import to_dense_batch
from .contrast import Contrast

class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats),
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)
        return x

class LSEblock(nn.Module):
    def __init__(self, hidden_dim = 128, num_layers = 3, num_heads = 8, num_node_type = 13):
        super(LSEblock, self).__init__()
        self.num_layers = num_layers
        self.NHGNNs = nn.ModuleList([NHGNN(hidden_dim = hidden_dim, num_heads = num_heads, num_node_type = num_node_type) for i in range(num_layers)])
        self.transform = LinearBlock(hidden_dim)
    def forward(self, x, edge_index, edge_attr, node_type_index):
        for i in range(self.num_layers):
            x, edge_attr = self.NHGNNs[i](x, edge_index, edge_attr, node_type_index)
        return self.transform(x)


class HSEblock(nn.Module):
    def __init__(self, hidden_dim = 128, num_layers = 3, hyperNum = 8, num_node_type = 13):
        super(HSEblock, self).__init__()
        self.num_layers = num_layers
        self.HHGNNs = nn.ModuleList([HHGNN(hidden_dim=hidden_dim, num_node_type = num_node_type) for i in range(num_layers)])
        self.dHyper = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim, hyperNum)))
        self.transform = LinearBlock(hidden_dim)
    def forward(self, x, batch, node_type_index):
        hyper_x, mask = to_dense_batch(x, batch)
        hyper_edge = hyper_x @ self.dHyper
        for i in range(self.num_layers):
            hyper_x = self.HHGNNs[i](hyper_edge, hyper_x, node_type_index, mask)
        return self.transform(hyper_x[mask])


class BiSemEncoder(nn.Module):
    def __init__(self, hidden_dim = 128, num_layers = 3, hyperNum = 8, num_node_type = 13, drop=0.2):
        super(BiSemEncoder, self).__init__()

        self.relu = nn.ReLU()
        self.num_layers = num_layers
        self.KG_node_transform = nn.Linear(400, hidden_dim)
        self.KG_edge_embedding = nn.Embedding(18, hidden_dim)
        self.NH_REP = LSEblock(hidden_dim = hidden_dim, num_layers = num_layers, num_heads = 8, num_node_type = num_node_type)
        self.HH_REP = HSEblock(hidden_dim=hidden_dim, num_layers=num_layers, hyperNum = hyperNum,num_node_type = num_node_type)
        self.constrast = Contrast(hidden_dim, 0.5, ['KG', 'MOL'])

    def forward(self, SUBKG_data1, SUBKG_data2, drug_index1, drug_index2, Trans_emb):

        KG_node_index1, KG_edge_index1, KG_edge_type_index1, KG_node_type_index1, KG_node_mask1, KG_batch1 = SUBKG_data1.node_type_, SUBKG_data1.edge_index, SUBKG_data1.edge_type, SUBKG_data1.atom_type, SUBKG_data1.node_mask, SUBKG_data1.batch
        KG_node_index2, KG_edge_index2, KG_edge_type_index2, KG_node_type_index2, KG_node_mask2, KG_batch2 = SUBKG_data2.node_type_, SUBKG_data2.edge_index, SUBKG_data2.edge_type, SUBKG_data2.atom_type, SUBKG_data2.node_mask, SUBKG_data2.batch
        Trans_emb = self.KG_node_transform(Trans_emb)
        KG_x1 = Trans_emb[KG_node_index1]
        KG_x2 = Trans_emb[KG_node_index2]
        KG_edge_feat1 = self.KG_edge_embedding(KG_edge_type_index1)
        KG_edge_feat2 = self.KG_edge_embedding(KG_edge_type_index2)

        KG_x1_HG = self.NH_REP(KG_x1, KG_edge_index1, KG_edge_feat1, KG_node_type_index1)
        KG_x1_HP = self.HH_REP(KG_x1, KG_batch1, KG_node_type_index1)
        KG_x2_HG = self.NH_REP(KG_x2, KG_edge_index2, KG_edge_feat2, KG_node_type_index2)
        KG_x2_HP = self.HH_REP(KG_x2, KG_batch2, KG_node_type_index2)

        KG_d1_feat_HG = KG_x1_HG[KG_node_mask1]
        KG_d2_feat_HG = KG_x2_HG[KG_node_mask2]
        KG_d1_feat_HP = KG_x1_HP[KG_node_mask1]
        KG_d2_feat_HP = KG_x2_HP[KG_node_mask2]
        if self.training:
            cl_loss = self.constrast(KG_d1_feat_HG, KG_d2_feat_HG, KG_d1_feat_HP, KG_d2_feat_HP, drug_index1, drug_index2)
        else:
            cl_loss = None
        KG_d1_feat = KG_d1_feat_HG + KG_d1_feat_HP
        KG_d2_feat = KG_d2_feat_HG + KG_d2_feat_HP
        return KG_d1_feat, KG_d2_feat, cl_loss


class BiSemDRUG(nn.Module):
    def __init__(self, hidden_dim = 128, ddi_type = 86, num_layers = 3, hyperNum = 8, drop = 0.2):
        super(BiSemDRUG, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.num_layers = num_layers

        self.hyperDDI_page = BiSemEncoder(hidden_dim = hidden_dim, num_layers=num_layers, hyperNum = hyperNum, num_node_type = 13, drop = drop)
        self.hyperDDI_RW = BiSemEncoder(hidden_dim=hidden_dim, num_layers=num_layers, hyperNum =  hyperNum, num_node_type = 13, drop = drop)
        self.constrast = Contrast(hidden_dim, 0.5, ['KG', 'MOL'])

        self.type_embedding = nn.Embedding(ddi_type, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, drug_index1, drug_index2, DDI_type_index, Trans_emb_HG, Trans_emb_HY, Page_SUBKG_data1, Page_SUBKG_data2, RW_SUBKG_data1, RW_SUBKG_data2):


        page_KG_d1_feat, page_KG_d2_feat, inter_cl_loss1 = self.hyperDDI_page(Page_SUBKG_data1, Page_SUBKG_data2, drug_index1, drug_index2, Trans_emb_HG)
        RW_KG_d1_feat, RW_KG_d2_feat, inter_cl_loss2 = self.hyperDDI_RW(RW_SUBKG_data1, RW_SUBKG_data2, drug_index1, drug_index2, Trans_emb_HY)

        if self.training:
            intra_cl_loss = self.constrast(page_KG_d1_feat, page_KG_d2_feat, RW_KG_d1_feat, RW_KG_d2_feat, drug_index1, drug_index2)
        else:
            intra_cl_loss = None
        KG_d1_feat = page_KG_d1_feat + RW_KG_d1_feat
        KG_d2_feat = page_KG_d2_feat + RW_KG_d2_feat

        DDI_type_emb = self.type_embedding(DDI_type_index)
        hc = torch.cat([KG_d1_feat, KG_d2_feat, DDI_type_emb], dim=1)

        hc = self.fc1(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        hc = self.fc2(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        output = self.out(hc)

        return output, inter_cl_loss1, inter_cl_loss2, intra_cl_loss


