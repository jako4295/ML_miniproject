from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from torch import nn


class NumbersDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.from_numpy(samples).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def load_data(dim_type: str, smoking: bool, as_numpy: bool) -> list[np.array]:
    if smoking:
        path = "data_Smoker/"
    else:
        path = "data_Diabetes_binary_5050_norm/"
    match dim_type.lower():
        case "full":
            file_names = [
                path+"trn_all.csv",
                path+"tst_all.csv",
            ]
        case "pca2":
            file_names = [
                path+"trn_pca2.csv",
                path+"tst_pca2.csv",
            ]
        case "pca10":
            file_names = [
                path+"trn_pca10.csv",
                path+"tst_pca10.csv",
            ]
        case other:
            raise KeyError("dim_type must be: 'full', 'pca2', or 'pca10'")
    
    file_names += [path+"trn_labs.csv", path+"tst_labs.csv"]
    
    if as_numpy:
        return (pd.read_csv(f).to_numpy() for f in file_names)
    else:
        return (pd.read_csv(f) for f in file_names)


class Network(nn.Module):
    def __init__(self, hidden_layer_dims: list[int], input_size: int, device: torch.device) -> None:
        super(Network, self).__init__()
        self.device = device
        
        self.relu = nn.ReLU().to(device)
        self.l1 = nn.Linear(input_size, hidden_layer_dims[0]).to(device)
        self.ln = nn.Linear(hidden_layer_dims[-1], 2).to(device)
        self.ls = [self.l1]
        
        for i in range(len(hidden_layer_dims) - 1):
            self.ls.append(
                nn.Linear(hidden_layer_dims[i], hidden_layer_dims[i+1]).to(device)
            )
        self.ls.append(self.ln)
        
    def forward(self, x: np.array) -> np.array:
        out = self.relu(self.ls[0](x))
        for l in self.ls[1:-1]:
            out = self.relu(l(out))
        out = self.ls[-1](out)
        
        return out