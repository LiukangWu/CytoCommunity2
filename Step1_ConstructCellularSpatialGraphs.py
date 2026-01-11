import numpy as np
import pandas as pd
import datetime
import os
import shutil
import torch
from torch_geometric.data import Data, InMemoryDataset
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# Hyperparameters
KNN_K = 20

<<<<<<< HEAD
InputFolderName = "./TNBC_Input/"
=======
InputFolderName = "./Step0_Output/"
>>>>>>> 84c27ac81f7e5f4cfe70fe741e8810b5085da1bd
ThisStep_OutputFolderName = "./Step1_Output/"
if os.path.exists(ThisStep_OutputFolderName):
    shutil.rmtree(ThisStep_OutputFolderName)
os.makedirs(ThisStep_OutputFolderName)

# Import image name list.
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
    Region_filename,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["Image"],  # set our own names for the columns
)

## Below is for generation of topology structures (edges) of cellular spatial graphs.
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("Constructing topology structures of KNN graphs...")
for graph_index in range(len(region_name_list)):
    print(f"This is image-{graph_index}")
    region_name = region_name_list.Image[graph_index]

    GraphCoord_filename = InputFolderName + region_name + "_Coordinates.txt"
    X = np.loadtxt(GraphCoord_filename, dtype='float64', delimiter="\t")
    N = len(X)

    nbrs = NearestNeighbors(n_neighbors=KNN_K+1, algorithm='auto').fit(X)
    _, idx = nbrs.kneighbors(X)    # [N, K+1]，其中一个是自身（距离 0）

    idx_wo_self = idx[:, 1:]       # 因为第一个就是自身（距离最小为 0）

    rows = np.repeat(np.arange(N), KNN_K)
    cols = idx_wo_self.reshape(-1)
    data = np.ones_like(cols, dtype=np.float32)

    # Build adjacency
    adj = csr_matrix((data, (rows, cols)), shape=(N, N))
    adj = adj.maximum(adj.T)       
    edge_index = np.vstack(adj.nonzero()).T
    filename0 = ThisStep_OutputFolderName + region_name + "_EdgeIndex.txt"
    np.savetxt(filename0, edge_index, delimiter='\t', fmt='%i')
print("All topology structures have been generated!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


## Below is for generation of node attribute matrices of cellular spatial graphs.
print("Generating node attribute matrices of KNN graphs...")
cell_type_vec = []
for graph_index in range(0, len(region_name_list)):

    region_name = region_name_list.Image[graph_index]
    # Import cell type label.
    CellType_filename = InputFolderName + region_name + "_CellTypeLabel.txt"
    cell_type_label = pd.read_csv(
        CellType_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["cell_type"],  # set our own names for the columns
    )
    cell_type_vec.extend(cell_type_label["cell_type"].values.tolist())

cell_type_vec_uniq = sorted(set(cell_type_vec))  # generate a vector of unique cell types and store it to .txt for final illustration.
CellTypeVec_filename = ThisStep_OutputFolderName + "UniqueCellTypeList.txt"
with open(CellTypeVec_filename, 'w') as fp:
    for item in cell_type_vec_uniq:
        # write each item on a new line
        fp.write("%s\n" % item)

# generate a node attribute matrix for each image.
for graph_index in range(0, len(region_name_list)):
    print(f"This is image-{graph_index}")
    region_name = region_name_list.Image[graph_index]

    # import cell type label.
    CellType_filename = InputFolderName + region_name + "_CellTypeLabel.txt"
    cell_type_label = pd.read_csv(
        CellType_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["cell_type"],  # set our own names for the columns
    )

    # initialize a zero-valued numpy matrix.
    node_attr_matrix = np.zeros((len(cell_type_label), len(cell_type_vec_uniq)))
    for cell_ind in range(0, len(cell_type_label)):
        # get the index of the current cell.
        type_index = cell_type_vec_uniq.index(cell_type_label["cell_type"][cell_ind])
        node_attr_matrix[cell_ind, type_index] = 1  # make the one-hot vector for each cell.

    filename1 = ThisStep_OutputFolderName + region_name + "_NodeAttr.txt"
    np.savetxt(filename1, node_attr_matrix, delimiter='\t', fmt='%i')  # save as integers.

print("All node attribute matrices have been generated!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


## Below is for transforming input graphs into the data structure required by deep geometric learning. 
print("Start graph data structure transformation...")
# Construct ordinary Python list to hold all input graphs.
data_list = []
for i in range(0, len(region_name_list)):
    region_name = region_name_list.Image[i]
    
    # Import network topology.
    EdgeIndex_filename = ThisStep_OutputFolderName + region_name + "_EdgeIndex.txt"
    edge_ndarray = np.loadtxt(EdgeIndex_filename, dtype='int64', delimiter="\t")
    edge_index = torch.from_numpy(edge_ndarray).t().contiguous()

    # Import node attribute.
    NodeAttr_filename = ThisStep_OutputFolderName + region_name + "_NodeAttr.txt"
    x_ndarray = np.loadtxt(NodeAttr_filename, dtype='float32', delimiter="\t")  # should be float32 not float or float64.
    x = torch.from_numpy(x_ndarray)

    # Import graph label.
    GraphLabel_filename = InputFolderName + region_name + "_GraphLabel.txt"
    graph_label = np.loadtxt(GraphLabel_filename, dtype='int64', delimiter="\t")  # change to int64 from int due to expected torch.LongTensor.
    y = torch.from_numpy(graph_label)

    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight, name=region_name)
    data_list.append(data)

# Define "SpatialOmicsImageDataset" class based on ordinary Python list.
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
        # Read data_list into huge `Data` list.
        data, slices = self.collate(data_list)
        torch.save(
            (data, slices),
            self.processed_paths[0],
            _use_new_zipfile_serialization=False
        )

# Create an object of this "SpatialOmicsImageDataset" class.
dataset = SpatialOmicsImageDataset(ThisStep_OutputFolderName)
print("Step2 done!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
