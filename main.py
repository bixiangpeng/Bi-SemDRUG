import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle as pkl
from dataset import MC_dataset, MC_collate
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
from model.BiSemDRUG import BiSemDRUG
from train_and_test import train_MC, test_MC
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

    for fold in range(args.folds):
        print(f'fold{fold} is being running !!!')
        train_samples = []
        with open(f'./data/{args.dataset}/train_fold{fold}.csv','r') as file:
            for index, line in enumerate(file.readlines()):
                if index == 0:
                    pass
                else:
                    line = line.strip().split(',')
                    train_samples.append((line[0], line[1], int(line[2]), int(line[3])))

        test_samples = []
        with open(f'./data/{args.dataset}/test_fold{fold}.csv', 'r') as file:
            for index, line in enumerate(file.readlines()):
                if index == 0:
                    pass
                else:
                    line = line.strip().split(',')
                    test_samples.append((line[0], line[1], int(line[2]), int(line[3])))

        train_dataset = MC_dataset(samples=train_samples, index_map=index_map, page_subgraph = page_subgraph, rw_subgraph = rw_subgraph)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=MC_collate)
        test_dataset = MC_dataset(samples=test_samples, index_map=index_map, page_subgraph = page_subgraph, rw_subgraph = rw_subgraph)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=MC_collate)

        model = BiSemDRUG(hidden_dim=args.hidden_dim, hyperNum=args.hyper_num, ddi_type=args.ddi_types, drop = 0.2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(args.epochs):
            print("Running EPOCH", epoch + 1)
            train_loss, train_acc = train_MC(model, train_loader, criterion, optimizer, Trans_emb_HG, Trans_emb_HY,  device, args)
            if (args.do_Save):
                torch.save(model.state_dict(), args.rst_file + f'fold{fold}/epoch' + f'{epoch}.pkl')
            test_loss, test_acc, auroc, f1_score, precision, recall, ap = test_MC(model, criterion, test_loader, Trans_emb_HG, Trans_emb_HY, device, args)
            msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (epoch, train_loss, train_acc, test_loss, test_acc, auroc, f1_score, precision, recall, ap)
            print(msg)
            with open(args.rst_file + f'fold{fold}/results.txt', 'a+') as fp:
                fp.write(msg + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'DrugBank')
    parser.add_argument('--ddi_types', type = int, default = 86, help='86 for drugbank, 963 for twosides')

    parser.add_argument('--device_id', type = int, default = 1)
    parser.add_argument('--num_workers', type = int, default = 2)

    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--hyper_num', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=1e-2, choices=[1e-3, 1e-2, 1e-1, 1, 10])
    parser.add_argument('--beta', type=float, default=1, choices=[1e-3, 1e-2, 1e-1, 1, 10])

    parser.add_argument('--folds', type = int, default = 3)
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--do_Save', type = bool, default = True)
    parser.add_argument("--rst_file", type = str, default = './model_pkl/DrugBank/')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    args = parser.parse_args()
    main(args)