import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle as pkl
from dataset import MC_dataset, MC_collate
from torch.utils.data.dataloader import DataLoader
from model.BiSemDRUG import BiSemDRUG
from train_and_test import do_compute_metrics
import numpy as np
from tqdm import tqdm


def main(args):
    device = torch.device('cuda:' + str(args.device_id if torch.cuda.is_available() else 'cpu'))
    with open(f'./data/{args.dataset}/Drkg_TransE_Emb.pkl', 'rb') as file:
        Trans_emb = pkl.load(file)
    with open(f'./data/{args.dataset}/page_subgraph/subgraph_to_Tensor_32.pkl','rb') as file:
        page_subgraph = pkl.load(file)
    with open(f'./data/{args.dataset}/RW_subgraph/subgraph_to_Tensor_32.pkl','rb') as file:
        rw_subgraph = pkl.load(file)
    with open(f'./data/{args.dataset}/index_map.pkl', 'rb') as file:
        index_map = pkl.load(file)
    Trans_emb_HG = Trans_emb.to(device)
    Trans_emb_HY = Trans_emb.to(device)


    test_samples = []
    with open(f'./data/{args.dataset}/test_fold2.csv', 'r') as file: #### for DrugBank
        for index, line in enumerate(file.readlines()):
            if index == 0:
                pass
            else:
                line = line.strip().split(',')
                test_samples.append((line[0], line[1], int(line[2]), int(line[3])))


    test_dataset = MC_dataset(samples=test_samples, index_map=index_map, page_subgraph = page_subgraph, rw_subgraph = rw_subgraph)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=MC_collate)

    model = BiSemDRUG(hidden_dim=args.hidden_dim, hyperNum=args.hyper_num, ddi_type=args.ddi_types).to(device)
    path = f'./model_pkl/{args.dataset}/pretrained_model.pkl'
    model.load_state_dict(torch.load(path, map_location=device))

    model.eval()
    pred_list = []
    label_list = []
    for data in tqdm(test_loader, total=len(test_loader)):
        drug_index1, drug_index2, type, label, SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2 = [d.to(device) for d in data]
        with torch.no_grad():
            pred, _, _, _ = model(drug_index1, drug_index2, type, Trans_emb_HG, Trans_emb_HY, SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2)
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)
    msg = "acc-%.4f, auroc-%.4f, f1_score-%.4f, prec-%.4f, rec-%.4f, ap-%.4f" % (acc, auroc, f1_score, precision, recall, ap)
    print(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'TwoSides')
    parser.add_argument('--ddi_types', type = int, default = 963, help='86 for drugbank, 963 for twosides')

    parser.add_argument('--device_id', type = int, default = 1)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--hyper_num', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=1e-2, choices=[1e-3, 1e-2, 1e-1, 1, 10])
    parser.add_argument('--beta', type=float, default=1, choices=[1e-3, 1e-2, 1e-1, 1, 10])

    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    args = parser.parse_args()
    main(args)