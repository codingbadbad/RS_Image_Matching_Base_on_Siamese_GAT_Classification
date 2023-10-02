import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
from torch_geometric.loader import DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filepath, num=31, row=31 , start = 21):
        super(DiabetesDataset, self).__init__()

        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # print(xy.shape[0])
        self.len = xy.shape[0] // row
        self.num_nodes = num
        self.data = []
        num_rows = xy.shape[0]

        for i in range(self.len):

            # x_data = torch.from_numpy(
            #     xy[i * row:  i * row + num, [3, 7, 4, 8, 5, 9, 6, 10, 11, 13, 12, 14, 15, 16, 17, 18,19,20]])
            x_data = torch.from_numpy(
                xy[i * row:  i * row + num, [3, 7, 4, 8, 5, 9, 6, 10, 11, 13, 12, 14, 15, 16, 17, 19,20]])


            weight = torch.from_numpy(xy[i * row:   i * row + num, start:start + num]) - torch.eye(num, dtype=torch.float32)
            weight[:, 1:] = 0

            edge_index = torch.where(weight > 0.3)
            edge_index = torch.stack(edge_index)
            edge_attr = weight[edge_index[0], edge_index[1]]

            y_data = torch.tensor([xy[i * row, 0]], dtype=torch.float)

            self.data.append(Data(x=x_data, edge_index=edge_index, edge_attr=edge_attr, y=y_data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


import os
from concurrent.futures import ThreadPoolExecutor


def doubleload(path):
    # files = os.listdir(path)
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)) )

    datalens = []
    all_data_lr, all_data_rl = [], []

    for file in files:
        file_path = os.path.join(path, file)
        dataset = DiabetesDataset(file_path)
        #print(file, len(dataset))
        datalens.append(int(len(dataset) / 2))

        for index, item in enumerate(dataset.data):
            if index % 2 == 0:
                all_data_lr.append(item)
            else:

                all_data_rl.append(item)
    print(f"read {path} ok")
    return all_data_lr, all_data_rl, datalens


def list2loader(dataset_lr, dataset_rl, batch_size):
    train_list_lr = [Data(x=datatemp.x,
                          edge_index=datatemp.edge_index, batch=torch.full((31,), i)) for i, datatemp in
                     enumerate(dataset_lr)]
    train_list_rl = [Data(x=datatemp.x,
                          edge_index=datatemp.edge_index, batch=torch.full((31,), i)) for i, datatemp in
                     enumerate(dataset_rl)]
    train_labels = [datatemp.y for i, datatemp in enumerate(dataset_lr)]

    # 创建DataLoader
    train_loader = DataLoader(list(zip(train_list_lr, train_list_rl, train_labels)), batch_size=batch_size,
                              shuffle=True)
    return train_loader


def cut(data_lr, data_rl, datalens, test_len):
    graphnum = sum(datalens[:test_len])
    # print(sum(datalens[:test_len]))
    train_dataset_lr = data_lr[graphnum:]
    train_dataset_rl = data_rl[graphnum:]
    test_dataset3_lr = data_lr[:graphnum]
    test_dataset3_rl = data_rl[:graphnum]

    train_list_lr = [Data(x=datatemp.x,
                          edge_index=datatemp.edge_index, batch=torch.full((31,), i)) for i, datatemp in
                     enumerate(train_dataset_lr)]
    train_list_rl = [Data(x=datatemp.x,
                          edge_index=datatemp.edge_index, batch=torch.full((31,), i)) for i, datatemp in
                     enumerate(train_dataset_rl)]
    train_labels = [datatemp.y for i, datatemp in enumerate(train_dataset_lr)]

    test_list1 = [Data(x=datatemp.x,
                       edge_index=datatemp.edge_index, batch=torch.full((31,), i)) for i, datatemp in
                  enumerate(test_dataset3_lr)]
    test_list2 = [Data(x=datatemp.x,
                       edge_index=datatemp.edge_index, batch=torch.full((31,), i)) for i, datatemp in
                  enumerate(test_dataset3_rl)]
    test_labels = [datatemp.y for i, datatemp in enumerate(test_dataset3_lr)]

    return train_list_lr, train_list_rl, train_labels, test_list1, test_list2, test_labels



def load_data(path):
    return doubleload(path)
