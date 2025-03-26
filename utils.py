from sklearn.model_selection import StratifiedShuffleSplit
import torch

def split_train_valid(data_df, fold, val_ratio=0.2):
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y=data_df['Y'])))

    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]

    return train_df, val_df


def load_data(fold = 0, mode='S1'):
    train_samples = []
    test_samples = []
    with open(f'/media/ST-18T/xiangpeng/MyDDI/data/DrugBank/cold_start/fold{fold}/train.csv', 'r') as file:
        for index, line in enumerate(file.readlines()):
            if index == 0:
                pass
            else:
                line = line.strip().split(',')
                train_samples.append((line[0], line[1], int(line[2]), int(line[3])))
    with open(f'/media/ST-18T/xiangpeng/MyDDI/data/DrugBank/cold_start/fold{fold}/{mode}.csv', 'r') as file:
        for index, line in enumerate(file.readlines()):
            if index == 0:
                pass
            else:
                line = line.strip().split(',')
                test_samples.append((line[0], line[1], int(line[2]), int(line[3])))
    return train_samples, test_samples


