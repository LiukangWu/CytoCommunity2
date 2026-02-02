import os
import shutil
import datetime
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv,DenseGraphConv
from torch_geometric.data import InMemoryDataset
from sparse_mincut_pool import sparse_mincut_pool_batch


## Hyperparameters
Num_TCN = 4
Num_Run = 10
Num_Epoch = 1000
Num_Class = 2
Embedding_Dimension = 128
LearningRate = 0.0005
MiniBatchSize = 2
beta = 0.9


Step0_OutputFolderName = "./Step0_Output/"
use_pseudo_samples = os.path.exists(Step0_OutputFolderName)

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
        self.conv3  = DenseGraphConv(hidden_channels, hidden_channels)
        self.lin1   = Linear(hidden_channels, hidden_channels)
        self.lin2   = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, graph_mask=None):
        x = F.relu(self.conv1(x, edge_index))
        s = self.pool1(x)

        x, adj, mc_loss, o_loss = sparse_mincut_pool_batch(
            x, edge_index, s, batch, graph_mask=graph_mask
        )
        x = self.conv3(x, adj) 
        x = x.mean(dim=1)   
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), mc_loss, o_loss, s, adj


def train_epoch(model, loader, optimizer, device, use_pseudo_samples):
    model.train()
    
    total_ce_loss = 0
    total_mincut_loss = 0
    total_samples = 0
    total_real_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # graph_mask: True means "participate in MinCut/Ortho"
        if use_pseudo_samples:
            graph_mask = data.graph_mask.view(-1).to(device)  # [B]
            num_real = int(graph_mask.sum().item())
        else:
            graph_mask = None  # all graphs are real
            num_real = data.num_graphs

        out, mc_loss, o_loss, _, _ = model(
            data.x, data.edge_index, data.batch, graph_mask=graph_mask
        )

        cross_entropy_loss = F.nll_loss(out, data.y.view(-1))
        mincut_loss = mc_loss + o_loss
        total_loss_value = cross_entropy_loss * (1 - beta) + mincut_loss * beta

        total_loss_value.backward()
        optimizer.step()

        n = data.num_graphs
        
        total_ce_loss += cross_entropy_loss.item() * n
        total_samples += n
        # MinCut is defined over REAL graphs only
        if num_real > 0:
            total_mincut_loss += mincut_loss.item() * num_real
            total_real_samples += num_real

    # compute averages with correct denominators
    avg_ce = total_ce_loss / max(total_samples, 1)
    avg_mc = total_mincut_loss / max(total_real_samples, 1)  # average over real only

    # total loss should be consistent with the two averages
    avg_total = avg_ce * (1 - beta) + avg_mc * beta

    return avg_total, avg_ce, avg_mc

def shuffle_pseudo(dataset):
    for i in range(len(dataset)):
        data = dataset[i]
        if data.is_pseudo.item():
            perm = torch.randperm(data.x.size(0))
            data.x = data.x[perm]


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
        if use_pseudo_samples:
            shuffle_pseudo(dataset)
            
        total_loss, ce_loss, mincut_loss = train_epoch(
            model, train_loader, optimizer, device, use_pseudo_samples
        )
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
                data.x, data.edge_index, data.batch
            )

        if use_pseudo_samples and data.y.item() == 0:
            continue

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
