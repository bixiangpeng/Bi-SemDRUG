import torch
from sklearn import metrics
import numpy as np
from tqdm import tqdm

def accuracy(y_true, y_pred):
    sample_count = 1.
    for s in y_true.shape:
        sample_count *= s

    return np.sum((y_true == y_pred)) / sample_count

def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(np.int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    ap= metrics.average_precision_score(target, probas_pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    return acc, auroc, f1_score, precision, recall, ap



def test_MC(model, criterion, dataloader, Trans_emb_HG, Trans_emb_HY, device, args):
    model.eval()
    pred_list = []
    label_list = []
    total_loss = 0
    n_batches = 0
    for data in tqdm(dataloader, total=len(dataloader)):
        drug_index1, drug_index2, type, label, SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2 = [d.to(device) for d in data]
        with torch.no_grad():
            pred, _, _, _ = model(drug_index1, drug_index2, type, Trans_emb_HG, Trans_emb_HY, SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2)
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            n_batches += 1
    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)
    return 0, acc, auroc, f1_score, precision, recall, ap

def train_MC(model, dataloader,criterion, optimizer, Trans_emb_HG, Trans_emb_HY, device, args):
    model.train()
    total_loss = 0
    n_batches = 0
    correct = 0
    for data in tqdm(dataloader, total=len(dataloader)):
        drug_index1, drug_index2, type, label, SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2 = [d.to(device) for d in data]
        optimizer.zero_grad()
        pred, inter_cl_loss1, inter_cl_loss2, intra_cl_loss = model(drug_index1, drug_index2, type, Trans_emb_HG, Trans_emb_HY, SUBKG_Data1, SUBKG_Data2, RWSUBKG_Data1, RWSUBKG_Data2)
        loss = criterion(pred, label.view(-1,1).float())
        loss += (args.alpha*((inter_cl_loss1 + inter_cl_loss2)/2) + args.beta* intra_cl_loss)
        y_pred = torch.round(torch.sigmoid(pred).squeeze(1))
        correct += torch.eq(y_pred, label).data.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        n_batches += 1
    avg_loss = total_loss / n_batches
    avg_acc = correct / (len(dataloader.dataset))
    return  avg_loss, avg_acc



