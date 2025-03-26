import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix

class Contrast(nn.Module):
    def __init__(self, out_dim, tau, keys):
        super(Contrast, self).__init__()
        self.proj = nn.ModuleDict({k: nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim)
        ) for k in keys})
        self.tau = tau
        for k, v in self.proj.items():
            for model in v:
                if isinstance(model, nn.Linear):
                    nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def compute_loss(self, MOL1, MOL2, KG1, KG2, p1_index, p2_index):
        MOL = torch.cat((MOL1,MOL2), dim = 0)
        p_index = torch.cat((p1_index,p2_index), dim=0).cpu().numpy()

        KG = torch.cat((KG1, KG2), dim=0)
        z_MOL = self.proj['MOL'](MOL)
        z_KG = self.proj['KG'](KG)

        matrix_str2seq = self.sim(z_MOL, z_KG)
        matrix_seq2str = matrix_str2seq.t()

        index_ = []
        for i, index in enumerate(p_index):
            temp = np.argwhere(p_index == index)
            for j in temp:
                index_.append([i, j[0]])
        index_ = np.transpose(np.array(index_))
        src_, target_ = index_[0], index_[1]
        temp_data2 = np.ones((src_.shape[0]), dtype=int)
        pos_ = torch.LongTensor(coo_matrix((temp_data2, (src_, target_)), shape=(matrix_seq2str.shape[0], matrix_seq2str.shape[1])).toarray()).to(MOL1.device)
        matrix_str2seq = matrix_str2seq / (torch.sum(matrix_str2seq, dim=1).view(-1, 1) + 1e-8)
        lori_str = -torch.log(matrix_str2seq.mul(pos_).sum(dim=-1)).mean()
        matrix_seq2str = matrix_seq2str / (torch.sum(matrix_seq2str, dim=1).view(-1, 1) + 1e-8)
        lori_seq = -torch.log(matrix_seq2str.mul(pos_).sum(dim=-1)).mean()
        return lori_str + lori_seq

    def forward(self, str1, str2, seq1, seq2, p1_index, p2_index):
        return self.compute_loss(str1, str2, seq1, seq2, p1_index, p2_index)
