# %%
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader
import methods_NN
import torch
from time import sleep
from tqdm import tqdm

# %%
def load_data(dim_type: str) -> list[np.array]:
    match dim_type.lower():
        case "full":
            file_names = [
                "data/trn_all.csv",
                "data/tst_all.csv",
            ]
        case "pca2":
            file_names = [
                "data/trn_pca2.csv",
                "data/tst_pca2.csv",
            ]
        case "pca10":
            file_names = [
                "data/trn_pca10.csv",
                "data/tst_pca10.csv",
            ]
        case other:
            raise KeyError("dim_type must be: 'full', 'pca2', or 'pca10'")
    
    file_names += ["data/trn_labs.csv", "data/tst_labs.csv"]
    
    return (pd.read_csv(f).to_numpy() for f in file_names)

# %%
# Data prep
# data_train.__len__() -> 2, 2, 2, 3, 3, 3, 7, 151
# data_test.__len__() -> 2, 2, 2, 3, 7, 151
batch_size = 7

data_train, data_test, label_train, label_test = load_data("full")
data_train = methods_NN.NumbersDataset(data_train, label_train)
data_test = methods_NN.NumbersDataset(data_test, label_train)

loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True, pin_memory=True)

# %%
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

# %%
# Hyperparams and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
learning_rate = 0.01

# %%
# Model prep
model = Network(
    hidden_layer_dims=[100, 50, 25, 10],
    input_size=21,
    device=device
).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %%
# Train the data
num_steps = len(loader_train)
loss_list = list()

for epoch in tqdm(range(epochs)):
    for i, (value, label) in enumerate(loader_train):
        sample = value.to(device)
        label = label.view(label.shape[0], 1).to(device)
        
        # forward
        output = model(sample)
        loss = criterion(output, label)
        loss_list.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# %%
# Test the data

# %%
# Summary


