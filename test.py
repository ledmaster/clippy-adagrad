import torch
import numpy as np
from clippyadagrad import ClippyAdagrad
from torch.optim import Adagrad

import tqdm

import sys
sys.path = ['Multitask-Recommendation-Library'] + sys.path
from models.sharedbottom import SharedBottomModel
from aliexpress import AliExpressDataset
from torch.utils.data import DataLoader

import pathlib

from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
    path = pathlib.Path("path_to_data")
    
    train_dataset = AliExpressDataset(path / "train.csv")
    test_dataset = AliExpressDataset(path / "test.csv")

    batch_size = 2048
    embed_dim = 128
    task_num = 2
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=7, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=7, shuffle=False)
    
    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num

    model = SharedBottomModel(field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=2, dropout=0.2).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = ClippyAdagrad(model.parameters(), lr=1e-1)
    
    
    log_interval = 100

    model.train()
    total_loss = 0
    loader = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
        y = model(categorical_fields, numerical_fields)
        loss_list = [criterion(y[i], labels[:, i].float()) for i in range(labels.size(1))]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)
        model.zero_grad()
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
            

    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(test_data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
            y = model(categorical_fields, numerical_fields)
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].extend(torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    print('AUC:', auc_results, 'Loss:', loss_results)
    