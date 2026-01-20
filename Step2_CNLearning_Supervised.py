import os
import shutil
import datetime
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv
from torch_geometric.data import InMemoryDataset
from sparse_mincut_pool import sparse_mincut_pool_batch


## Hyperparameters
Num_TCN = 10
Num_Run = 10
Num_Epoch = 1000
Num_Class = 3
Embedding_Dimension = 128
LearningRate = 0.001
MiniBatchSize = 2
beta = 0.9


## Load dataset from Step1 
LastStep_OutputFolderName = "./Step1_Output/"
ThisStep_OutputFolderName = "./Step2_Output/"
if os.path.exists(ThisStep_OutputFolderName):
    shutil.rmtree(ThisStep_OutputFolderName)
os.makedirs(ThisStep_OutputFolderName)

class SpatialOmicsImageDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SpatialOmicsImageDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SpatialOmicsImageDataset.pt']

    def download(self):
        pass

    def process(self):
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
dataset = SpatialOmicsImageDataset(LastStep_OutputFolderName)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=Embedding_Dimension):
        super(Net, self).__init__()
        self.conv1  = GraphConv(in_channels, hidden_channels)
        self.pool1  = Linear(hidden_channels, Num_TCN)
        self.conv3  = GraphConv(hidden_channels, hidden_channels)
        self.lin1   = Linear(hidden_channels, hidden_channels)
        self.lin2   = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        s = self.pool1(x)  
        x, adj, mc_loss, o_loss = sparse_mincut_pool_batch(
            x, edge_index, s, batch,
            edge_weight=edge_weight
        )

        # Change pooled adj to sparse edge_index
        B, C, F_dim = x.size()
        x_flat = x.view(B * C, F_dim)     

        edge_index_list = []
        edge_weight_list = []

        for b in range(B):
            adj_b = adj[b]   # [C, C]
            # Adding the batch offset: The index range of the C nodes of the b-th graph is [b*C, (b+1)*C-1]
            row, col = (adj_b != 0).nonzero(as_tuple=True)
            edge_index_list.append(torch.stack([row + b * C, col + b * C], dim=0))  # [2, E_b]
            edge_weight_list.append(adj_b[row, col])

        pooled_edge_index  = torch.cat(edge_index_list, dim=1).long()             # [2, E_tot]
        pooled_edge_weight = torch.cat(edge_weight_list).to(x_flat.dtype)         # [E_tot]

        x_flat = self.conv3(x_flat, pooled_edge_index, pooled_edge_weight)

        # reshape [B, C, F']
        x = x_flat.view(B, C, -1)
        
        x = x.mean(dim=1)   # [B, F']
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), mc_loss, o_loss, s, adj


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_mincut_loss = 0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, mc_loss, o_loss, _, _ = model(
            data.x, data.edge_index, data.batch, data.edge_weight
        )
        cross_entropy_loss = F.nll_loss(out, data.y.view(-1))
        mincut_loss = mc_loss + o_loss
        total_loss_value = cross_entropy_loss * (1 - beta) + mincut_loss * beta

        total_loss_value.backward()
        optimizer.step()

        n = data.num_graphs
        total_loss += total_loss_value.item() * n
        total_ce_loss += cross_entropy_loss.item() * n
        total_mincut_loss += mincut_loss.item() * n
        total_samples += n

    return total_loss / total_samples, total_ce_loss / total_samples, total_mincut_loss / total_samples


print("Start:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
for run_ind in range(1, Num_Run+1):    
    print(f"\n=== This is Run {run_ind:02d} ===")
    RunFolderName = os.path.join(ThisStep_OutputFolderName, f"Run{run_ind}")
    if os.path.exists(RunFolderName):
        shutil.rmtree(RunFolderName)
    os.makedirs(RunFolderName)

    train_loader = DataLoader(
        dataset, batch_size=MiniBatchSize,
        shuffle=True, pin_memory=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, Num_Class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate)
    
    # save loss
    loss_csv = os.path.join(RunFolderName, "Epoch_TrainLoss.csv")
    with open(loss_csv, "w", newline='') as f0:
        writer = csv.writer(f0)
        writer.writerow(["Epoch", "TotalLoss", "CrossEntropyLoss", "MinCutLoss"])

    for epoch in range(1, Num_Epoch+1):
        total_loss, ce_loss, mincut_loss = train_epoch(model, train_loader, optimizer, device)
        with open(loss_csv, "a", newline='') as f0:
            csv.writer(f0).writerow([epoch, total_loss, ce_loss, mincut_loss])
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}  TotalLoss={total_loss:.4f}  CrossEntropyLoss={ce_loss:.4f}  MinCutLoss={mincut_loss:.4f}")

    # Save the clustering results
    model.eval()
    sample_loader = DataLoader(dataset, batch_size=1)
    for idx, data in enumerate(sample_loader):
        data = data.to(device)
        with torch.no_grad():
            _, _, _, s, pooled_adj = model(
                data.x, data.edge_index, data.batch, data.edge_weight
            )

        # Save the node allocation matrix
        assign_np = torch.softmax(s, dim=-1).cpu().numpy()      
        np.savetxt(
            os.path.join(RunFolderName, f"ClusterAssignMatrix1_{idx}.csv"),
            assign_np, delimiter=','
        )

        # Save the inter-cluster adjacency after pooling
        adj_np = pooled_adj[0].cpu().numpy()                   
        np.savetxt(
            os.path.join(RunFolderName, f"ClusterAdjMatrix1_{idx}.csv"),
            adj_np, delimiter=','
        )

    print(f"Run {run_ind:02d} done at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("All runs finished at", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
