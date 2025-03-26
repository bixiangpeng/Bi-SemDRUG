import torch
import os
import networkx as nx
from randomWalk.node2vec import Node2vec
from torch_geometric.utils import subgraph
import pickle as pkl
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import remove_self_loops, add_self_loops


def google_matrix(G, alpha=0.85, personalization=None, nodelist=None, weight="weight", dangling=None):
    if nodelist is None:
        nodelist = list(G)
    M = np.asmatrix(nx.to_numpy_array(G, nodelist=nodelist, weight=weight), dtype='float32')
    N = len(G)
    if N == 0:
        return M
    if personalization is None:
        p = np.repeat(1.0 / N, N).astype('float32')
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype="float32")
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
    if dangling is None:
        dangling_weights = p
    else:
        dangling_weights = np.array([dangling.get(n, 0) for n in tqdm(nodelist)], dtype='float32')
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]
    for node in tqdm(dangling_nodes):
        M[node] = dangling_weights
    M /= M.sum(axis=1).astype('float32')
    return np.multiply(alpha, M, dtype='float32') + np.multiply(1 - alpha, p, dtype='float32')

def probExtractor(drug_id, edge_index, node_index, rel_index, fixed_num, pagerank_path, subgraph_path):
    if os.path.exists(subgraph_path):
        with open(subgraph_path, 'rb') as f:
            subgraphs = pkl.load(f)
        return subgraphs
    g = nx.DiGraph()
    g.add_nodes_from(node_index)
    g.add_edges_from(edge_index.transpose(1, 0).tolist())
    if not os.path.exists(pagerank_path):
        pagerank = np.array(google_matrix(g), dtype='float32')
        page_dict = {}
        for d in tqdm(drug_id):
            page_dict[d] = list(pagerank[list(g.nodes()).index(index_map[d])])
        with open(pagerank_path, 'wb') as f:
            pkl.dump(page_dict, f)
    else:
        with open(pagerank_path, 'rb') as f:
            page_dict = pkl.load(f)
        f.close()

    edge_index = torch.from_numpy(edge_index)
    subgraphs = {}
    max_rel_type = 0
    for d in tqdm(drug_id):
        try:
            subsets = [index_map[d]]
            neighbors = np.random.choice(a=list(g.nodes()), size=fixed_num, replace=False, p=np.array(page_dict[d]) / np.sum(page_dict[d]))
            subsets.extend(neighbors)
            subsets = list(set(subsets))
            mapping_list = [False for _ in subsets]
            mapping_idx = subsets.index(index_map[d])
            mapping_list[mapping_idx] = True
            sub_edge_index, sub_rel_index =  subgraph(subsets, edge_index, rel_index, relabel_nodes=True)
            if max_rel_type < np.max(sub_rel_index):
                max_rel_type = np.max(sub_rel_index)
            new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
            new_s_rel = sub_rel_index.tolist()
            for i in range(len(subsets)):
                new_s_edge_index.append([i,i])
                new_s_rel.append(17)
            subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list
        except:
            print('hit')
            subsets = [index_map[d]]
            for i in range(fixed_num):
                subsets.append(index_map[d])
            assert len(subsets) == fixed_num +1
            mapping_list = [False for _ in subsets]
            mapping_list[0] = True
            new_s_edge_index = [[0,0]]
            new_s_rel = [17]
            subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list

    with open(subgraph_path, 'wb') as f:
        pkl.dump((subgraphs, max_rel_type), f)

    return subgraphs, max_rel_type

def rwExtractor(drug_id, edge_index, node_index, rel_index, sub_num, length, subgraph_path):
    if os.path.exists(subgraph_path):
        with open(subgraph_path, 'rb') as f:
            subgraphs = pkl.load(f)
        return subgraphs

    my_graph = nx.Graph()
    my_graph.add_nodes_from(node_index)
    my_graph.add_edges_from(edge_index.transpose(1,0).tolist())

    edge_index = torch.from_numpy(edge_index)
    subgraphs = {}
    max_rel_type = 0
    for d in tqdm(drug_id):
        try:
            subsets = Node2vec(start_nodes=[int(index_map[d])], graph=my_graph, path_length=length, num_paths=sub_num, workers=6, dw=True).get_walks() ##返回一个list
            mapping_id = subsets.index(int(index_map[d]))
            mapping_list = [False for _ in range(len((subsets)))]
            mapping_list[mapping_id] = True
            sub_edge_index, sub_rel_index = subgraph(subsets, edge_index, rel_index, relabel_nodes=True)
            if max_rel_type < np.max(sub_rel_index):
                max_rel_type = np.max(sub_rel_index)
            new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
            new_s_rel = sub_rel_index.tolist()
            subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list
        except:
            print('hit')
            subsets = [index_map[d]]
            for i in range(length-1):
                subsets.append(index_map[d])
            assert len(subsets) == length
            mapping_list = [False for _ in subsets]
            mapping_list[0] = True
            new_s_edge_index = [[0, 0]]
            new_s_rel = [17]
            subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list
    with open(subgraph_path, 'wb') as f:
        pkl.dump((subgraphs, max_rel_type), f)

    return subgraphs, max_rel_type


divide_methods = ['PAGERANK', 'RW']
datasets = ['DrugBank','TwoSides', 'DeepDDI']
parmeters = [32]

for dataset in datasets:
    print(f'{dataset} is being processed!')
    with open(f'./{dataset}/edge_index.pkl','rb') as file:
        edge_index = pkl.load(file)
    with open(f'./{dataset}/edge_type_index.pkl','rb') as file:
        edge_type_index = pkl.load(file)
    with open(f'./{dataset}/index_map.pkl', 'rb') as file:
        index_map = pkl.load(file)
    with open(f'./{dataset}/node_type_index.pkl','rb') as file:
        node_type_index = pkl.load(file)
    node_index_ = [i for i in range(len(node_type_index))]
    drug_ids = list(index_map.keys())
    for divide_method in divide_methods:
        for parmeter in parmeters:
            print(f'{divide_method}_subgraph_{str(parmeter)} is being processed!')
            if divide_method == 'RW':
                subgraph_path = f'./{dataset}/RW_subgraph/subgraph_{parmeter}.pkl'
                subgraph_, new_rel_type = rwExtractor(drug_ids, edge_index, node_index_, edge_type_index, 1, parmeter, subgraph_path)
            else:
                subgraph_path = f'./{dataset}/page_subgraph/subgraph_{parmeter}.pkl'
                pagerank_path = f'./{dataset}/page_subgraph/pagerank.pkl'
                subgraph_, new_rel_type = probExtractor(drug_ids, edge_index, node_index_, edge_type_index, parmeter, pagerank_path, subgraph_path)

            subgraph_to_Tensor = {}
            max_ = 0
            for ID, item in tqdm(subgraph_.items()):
                if max_ < max(item[0]):
                    max_ = max(item[0])
                subgraph_node_index = torch.LongTensor(item[0])
                subgraph_node_type_index = torch.LongTensor([node_type_index[index] for index in item[0]])
                subgraph_edge_index = torch.from_numpy(np.array(item[1]).transpose(1,0))
                subgraph_edge_type_index = torch.LongTensor(item[2])
                subgraph_edge_index, subgraph_edge_type_index = remove_self_loops(subgraph_edge_index, edge_attr=subgraph_edge_type_index)
                subgraph_edge_index, subgraph_edge_type_index = add_self_loops(subgraph_edge_index, edge_attr=subgraph_edge_type_index, fill_value=17, num_nodes=subgraph_node_index.shape[0])
                subgraph_node_mask = torch.tensor(item[3], dtype=torch.bool)
                subgraph_to_Tensor[ID] = ( subgraph_node_index, subgraph_node_type_index, subgraph_edge_index, subgraph_edge_type_index, subgraph_node_mask )

            if divide_method == 'RW':
                with open(f'./{dataset}/RW_subgraph/subgraph_to_Tensor_{parmeter}_.pkl', 'wb') as file:
                    pkl.dump(subgraph_to_Tensor, file)
            else:
                with open(f'./{dataset}/page_subgraph/subgraph_to_Tensor_{parmeter}_.pkl', 'wb') as file:
                    pkl.dump(subgraph_to_Tensor, file)
            print(f'{dataset} have been processed successfully!')











