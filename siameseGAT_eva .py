import cv2
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader
import random
import os
from torch_geometric.utils import softmax
import numpy as np
import torch
import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch_geometric.nn import GATConv
from torch.nn import Linear


class SiameseGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, fc_chanel, head=4):
        super(SiameseGAT, self).__init__()
        self.num_classes = num_classes
        self.gat = GATConv(num_features, num_classes, heads=head)
        self.gat1 = GATConv(num_classes * head, num_classes, heads=head)
        self.gat2 = GATConv(num_classes * head, num_classes, heads=head)

        self.bn1 = torch.nn.BatchNorm1d(num_classes * head)
        self.bn2 = torch.nn.BatchNorm1d(num_classes * head)
        self.bn3 = torch.nn.BatchNorm1d(num_classes * head)

        self.fc_individual = Linear(num_classes * head, fc_chanel)
        self.bn_individual = torch.nn.BatchNorm1d(fc_chanel)

        self.fc = Linear(fc_chanel, fc_chanel // 4)
        self.bn_fc = torch.nn.BatchNorm1d(fc_chanel // 4)

        self.fc1 = Linear((fc_chanel // 4) * 2, fc_chanel // 4)
        self.bn_fc1 = torch.nn.BatchNorm1d(fc_chanel // 4)

        self.fc2 = Linear(fc_chanel // 4, 1)

        self.attention = torch.nn.Linear(num_classes * head, 1)

    def normalize_tensor(self, tensor):
        min_val = tensor.min(dim=1, keepdim=True)[0]
        max_val = tensor.max(dim=1, keepdim=True)[0]
        epsilon = 1e-10
        normalized_tensor = (tensor - min_val) / (max_val - min_val + epsilon)
        return normalized_tensor

    def graph_attention_pool(self, x, batch, attention):
        attention_scores = attention(x)
        attention_scores = softmax(attention_scores, batch)

        pooled_features = scatter_add(x * attention_scores, batch, dim=0)
        return pooled_features

    def forward(self, data1, data2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # Apply GAT layers with Batch Norm
        x1 = self.gat(x1, edge_index1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.gat(x2, edge_index2)
        x2 = self.bn1(x2)
        x2 = F.relu(x2)

        x1 = self.gat1(x1, edge_index1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)

        x2 = self.gat1(x2, edge_index2)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x1 = self.gat2(x1, edge_index1)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)

        x2 = self.gat2(x2, edge_index2)
        x2 = self.bn3(x2)
        x2 = F.relu(x2)

        # Apply Set Transformer Pooling
        x1 = self.graph_attention_pool(x1, batch1, self.attention)
        x2 = self.graph_attention_pool(x2, batch2, self.attention)

        x1 = self.fc_individual(x1)
        x1 = self.bn_individual(x1)
        x1 = F.relu(x1)
        x1 = self.fc(x1)

        x2 = self.fc_individual(x2)
        x2 = self.bn_individual(x2)
        x2 = F.relu(x2)
        x2 = self.fc(x2)

        # Concatenate and apply final classifier with Batch Norm
        out = torch.cat([x1, x2], dim=-1)
        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        return torch.sigmoid(out).squeeze(-1)


input_feature = 17
hiddne_chanel = 48
fc_chanel = 48

model = SiameseGAT(input_feature, hiddne_chanel, fc_chanel)
model = model.to(device)

# model.load_state_dict(torch.load('./saved_models/model_epoch_34 +score 199.27833817864698.pt'))
model.load_state_dict(torch.load('./saved_models/model_epoch_79 +score 996.4472422164132.pt'))

model.eval()
bias = 0.08


def evaluate(model, train_loader, filename="model_outputs_labels.xlsx"):
    model.eval()

    outputs = []
    labels_list = []

    with torch.no_grad():
        TP = 0
        TPFP = 0
        TPFN = 0
        start_time = time.time()
        for data1, data2, label in train_loader:
            data1 = data1.to(device)
            data2 = data2.to(device)
            label = label.float().to(device)

            out = model(data1, data2)

            pred = torch.round(out + bias)

            TPFP = int((torch.eq(pred.view(-1), 1)).sum())

            TP = int((torch.logical_and(torch.eq(pred.view(-1), 1), torch.eq(label.view(-1), 1))).sum())
            TPFN = int((torch.eq(label.view(-1), 1)).sum())

    precision = TP / TPFP if TPFP != 0 else 0.0
    recall = TP / TPFN if TPFN != 0 else 0.0
    print(f"Precision: {precision:.6f}, Recall: {recall:.6f}")

    return precision, recall, pred


path = './tgrs/RS/both/'

files = [f for f in os.listdir(path) if f.endswith('.csv')]

import csv
import time
import re


def extract_png_names(filename, endwith='.jpg'):
    pattern = r"(\w+\.jpg)"

    if endwith == '.png': pattern = r"(\w+\.png)"
    # pattern = r"(\w+\.jpg)"
    png_names = re.findall(pattern, filename)
    png_names = [name.replace('from', '') for name in png_names]
    return png_names


def readname(th, file_name, endswith):
    if th == 1:
        match = re.search(r"from(\d+)_", file_name)
        k = match.group(1)
        print(k)
        image_path1 = k + 'l' + '.png'
        image_path2 = k + 'r' + '.png'
    else:
        [image_path1, image_path2] = extract_png_names(file_name, endwith=endswith)
        # = extract_png_names(file_name)[1]
    return image_path1, image_path2


def draw(path, filename, pred):
    lens = 62
    image_path1, image_path2 = readname(1, filename, endswith='.png')
    # image_path1, image_path2 = readname(0,filename , endswith = '.png')
    image1 = cv2.imread(path + image_path1)
    image2 = cv2.imread(path + image_path2)
    # cv2.imshow("",image1)
    # cv2.waitKey(0)
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    # print(height1,width1,height2,width2)

    hmax = max(height1, height2)
    wmax = max(width1, width2)

    merged_width = width1 + width2
    merged_height = max(height1, height2)
    shift = 10

    merged_image = np.zeros((merged_height, merged_width + shift, 3), dtype=np.uint8) + 255
    merged_image[:height1, :width1] = image1
    merged_image[:height2, width1 + shift:] = image2

    whiteimage = np.zeros((hmax, wmax, 3), dtype=np.uint8) + 255

    xy = np.loadtxt(path + filename, delimiter=',', dtype=np.float32)
    labels = xy[::lens, 0]
    pred = pred.cpu().numpy()

    labels = labels.astype(int)
    pred = pred.astype(int)

    results = []

    for p, l in zip(pred, labels):
        if p == 1 and l == 1:
            results.append('TP')
        elif p == 1 and l == 0:
            results.append('FP')
        elif p == 0 and l == 1:
            results.append('FN')
        else:  # p == 0 and l == 0
            results.append('TN')

    x = xy[::lens, [3, 4]]
    y = xy[::lens, [5, 6]]
    width1 += shift
    for i in range(len(results)):

        if results[i] == 'TN':
            pt1 = (int(x[i][0] * wmax), int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_ = (int(y[i][0] * wmax), int(y[i][1] * hmax))
            # cv2.line(merged_image, pt1, pt2,  (0, 0, 0), 1)
            cv2.line(whiteimage, pt1, pt2_, (0, 0, 0), 1)

    for i in range(len(results)):
        if results[i] == 'TP':
            pt1 = (int(x[i][0] * wmax), int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_ = (int(y[i][0] * wmax), int(y[i][1] * hmax))
            cv2.line(merged_image, pt1, pt2, (255, 0, 0), 1)
            cv2.line(whiteimage, pt1, pt2_, (255, 0, 0), 1)
    for i in range(len(results)):
        if results[i] == 'FP':
            pt1 = (int(x[i][0] * wmax), int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_ = (int(y[i][0] * wmax), int(y[i][1] * hmax))
            cv2.line(merged_image, pt1, pt2, (0, 0, 255), 2)
            cv2.line(whiteimage, pt1, pt2_, (0, 0, 255), 2)
        if results[i] == 'FN':
            pt1 = (int(x[i][0] * wmax), int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_ = (int(y[i][0] * wmax), int(y[i][1] * hmax))
            cv2.line(merged_image, pt1, pt2, (0, 255, 0), 2)
            cv2.line(whiteimage, pt1, pt2_, (0, 255, 0), 2)
    #
    cv2.imshow('whiteimage', whiteimage)
    cv2.imshow('merged_image', merged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./view/' + image_path1 + image_path2 + 'whiteimage.png', whiteimage)
    cv2.imwrite('./view/' + image_path1 + image_path2 + 'merged_image.png', merged_image)


def evaluate_and_write_to_csv(model, train_loader, csv_writer, file_name):
    precision, recall, pred = evaluate(model, train_loader)
    draw(path, file_name, pred)

    csv_writer.writerow([file_name, precision, recall])


with open('./evaluation_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'Precision', 'Recall', 'Runtime'])
    for file_name in files:
        data = dataloader.DiabetesDataset(path + file_name)
        train_dataset_lr, train_dataset_rl = [], []
        for index, item in enumerate(data.data):
            if index % 2 == 0:
                train_dataset_lr.append(item)
            else:
                train_dataset_rl.append(item)

        train_list_lr = [Data(x=datatemp.x,
                              edge_index=datatemp.edge_index, batch=torch.full((31,), i)) for i, datatemp in
                         enumerate(train_dataset_lr)]
        train_list_rl = [Data(x=datatemp.x,
                              edge_index=datatemp.edge_index, batch=torch.full((31,), i)) for i, datatemp in
                         enumerate(train_dataset_rl)]
        train_labels = [datatemp.y for i, datatemp in enumerate(train_dataset_lr)]
        print(len(train_labels))
        train_loader = DataLoader(list(zip(train_list_lr, train_list_rl, train_labels)),
                                  batch_size=len(train_labels))
        print(len(train_labels))
        evaluate_and_write_to_csv(model, train_loader, writer, file_name)
