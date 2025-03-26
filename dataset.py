from torch.utils.data.dataset import Dataset
from torch_geometric.data import Batch
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric import data as DATA



def MKG_collate(items):
    index1 = [item[0] for item in items]
    index2 = [item[1] for item in items]
    ddi_type = [item[2] for item in items]
    label = [item[3] for item in items]
    return torch.LongTensor(index1), torch.LongTensor(index2), torch.LongTensor(ddi_type), torch.LongTensor(label)

class MKG_dataset(Dataset):
    def __init__(self, samples=None, index_map = None):
        super(MKG_dataset, self).__init__()
        self.samples = samples
        self.index_map = index_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        drug_ID1, drug_ID2, type, label = self.samples[index]
        index1, index2 = self.index_map[drug_ID1], self.index_map[drug_ID2]

        return index1, index2, type, label



def MC_collate(items):
    SUBKG_Data1 = Batch.from_data_list([item[0] for item in items])
    SUBKG_Data2 = Batch.from_data_list([item[1] for item in items])
    RWSUBKG_Data1 = Batch.from_data_list([item[2] for item in items])
    RWSUBKG_Data2 = Batch.from_data_list([item[3] for item in items])
    index1 = [item[4] for item in items]
    index2 = [item[5] for item in items]
    ddi_type = [item[6] for item in items]
    label = [item[7] for item in items]

    return torch.LongTensor(index1), torch.LongTensor(index2), torch.LongTensor(ddi_type), torch.LongTensor(label), SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2

class MC_dataset(Dataset):
    def __init__(self, samples=None, page_subgraph = None, index_map = None, rw_subgraph = None):
        super(MC_dataset, self).__init__()
        self.samples = samples
        self.page_subgraph = page_subgraph
        self.rw_subgraph = rw_subgraph
        self.index_map = index_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        drug_ID1, drug_ID2, type, label = self.samples[index]
        index1, index2 = self.index_map[drug_ID1], self.index_map[drug_ID2]
        subgraph_node_index1, subgraph_node_type_index1, subgraph_edge_index1, subgraph_edge_type_index1, subgraph_node_mask1 = self.page_subgraph[drug_ID1]
        subgraph_node_index2, subgraph_node_type_index2, subgraph_edge_index2, subgraph_edge_type_index2, subgraph_node_mask2 = self.page_subgraph[drug_ID2]
        SUBKG_Data1 = DATA.Data(x=torch.zeros((subgraph_node_index1.shape[0], 1)), node_type_ = subgraph_node_index1, edge_index=subgraph_edge_index1, edge_type = subgraph_edge_type_index1, atom_type = subgraph_node_type_index1, node_mask = subgraph_node_mask1)
        SUBKG_Data2 = DATA.Data(x=torch.zeros((subgraph_node_index2.shape[0], 1)), node_type_ = subgraph_node_index2, edge_index=subgraph_edge_index2, edge_type = subgraph_edge_type_index2, atom_type = subgraph_node_type_index2, node_mask = subgraph_node_mask2)

        rw_subgraph_node_index1, rw_subgraph_node_type_index1, rw_subgraph_edge_index1, rw_subgraph_edge_type_index1, rw_subgraph_node_mask1 = self.rw_subgraph[drug_ID1]
        rw_subgraph_node_index2, rw_subgraph_node_type_index2, rw_subgraph_edge_index2, rw_subgraph_edge_type_index2, rw_subgraph_node_mask2 = self.rw_subgraph[drug_ID2]
        RWSUBKG_Data1 = DATA.Data(x=torch.zeros((rw_subgraph_node_index1.shape[0], 1)), node_type_=rw_subgraph_node_index1, edge_index=rw_subgraph_edge_index1, edge_type=rw_subgraph_edge_type_index1, atom_type=rw_subgraph_node_type_index1, node_mask=rw_subgraph_node_mask1)
        RWSUBKG_Data2 = DATA.Data(x=torch.zeros((rw_subgraph_node_index2.shape[0], 1)), node_type_=rw_subgraph_node_index2, edge_index=rw_subgraph_edge_index2, edge_type=rw_subgraph_edge_type_index2, atom_type=rw_subgraph_node_type_index2, node_mask=rw_subgraph_node_mask2)
        return SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2, index1, index2, type, label



def Binary_collate(items):
    SUBKG_Data1 = Batch.from_data_list([item[0] for item in items])
    SUBKG_Data2 = Batch.from_data_list([item[1] for item in items])
    RWSUBKG_Data1 = Batch.from_data_list([item[2] for item in items])
    RWSUBKG_Data2 = Batch.from_data_list([item[3] for item in items])
    index1 = [item[4] for item in items]
    index2 = [item[5] for item in items]
    label = [item[6] for item in items]

    return torch.LongTensor(index1), torch.LongTensor(index2), torch.LongTensor(label), SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2

class Binary_dataset(Dataset):
    def __init__(self, samples=None, page_subgraph = None, index_map = None, rw_subgraph = None):
        super(Binary_dataset, self).__init__()
        self.samples = samples
        self.page_subgraph = page_subgraph
        self.rw_subgraph = rw_subgraph
        self.index_map = index_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        drug_ID1, drug_ID2, label = self.samples[index]
        index1, index2 = self.index_map[drug_ID1], self.index_map[drug_ID2]
        subgraph_node_index1, subgraph_node_type_index1, subgraph_edge_index1, subgraph_edge_type_index1, subgraph_node_mask1 = self.page_subgraph[drug_ID1]
        subgraph_node_index2, subgraph_node_type_index2, subgraph_edge_index2, subgraph_edge_type_index2, subgraph_node_mask2 = self.page_subgraph[drug_ID2]
        SUBKG_Data1 = DATA.Data(x=torch.zeros((subgraph_node_index1.shape[0], 1)), node_type_ = subgraph_node_index1, edge_index=subgraph_edge_index1, edge_type = subgraph_edge_type_index1, atom_type = subgraph_node_type_index1, node_mask = subgraph_node_mask1)
        SUBKG_Data2 = DATA.Data(x=torch.zeros((subgraph_node_index2.shape[0], 1)), node_type_ = subgraph_node_index2, edge_index=subgraph_edge_index2, edge_type = subgraph_edge_type_index2, atom_type = subgraph_node_type_index2, node_mask = subgraph_node_mask2)

        rw_subgraph_node_index1, rw_subgraph_node_type_index1, rw_subgraph_edge_index1, rw_subgraph_edge_type_index1, rw_subgraph_node_mask1 = self.rw_subgraph[drug_ID1]
        rw_subgraph_node_index2, rw_subgraph_node_type_index2, rw_subgraph_edge_index2, rw_subgraph_edge_type_index2, rw_subgraph_node_mask2 = self.rw_subgraph[drug_ID2]
        RWSUBKG_Data1 = DATA.Data(x=torch.zeros((rw_subgraph_node_index1.shape[0], 1)), node_type_=rw_subgraph_node_index1, edge_index=rw_subgraph_edge_index1, edge_type=rw_subgraph_edge_type_index1, atom_type=rw_subgraph_node_type_index1, node_mask=rw_subgraph_node_mask1)
        RWSUBKG_Data2 = DATA.Data(x=torch.zeros((rw_subgraph_node_index2.shape[0], 1)), node_type_=rw_subgraph_node_index2, edge_index=rw_subgraph_edge_index2, edge_type=rw_subgraph_edge_type_index2, atom_type=rw_subgraph_node_type_index2, node_mask=rw_subgraph_node_mask2)
        return SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2, index1, index2, label


