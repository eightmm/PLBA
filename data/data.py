from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
    
import numpy as np


class PDBbindHeteroDataset(DGLDataset):
    def __init__(self, hetero_graph_path):
        super(PDBbindDataset, self).__init__(name='PDBbind v2020')
        self.g, self.label = load_graphs( hetero_graph_path )
        self.label = self.label['label']
        
    def __getitem__(self, idx):
        return self.g[idx], self.label[idx]
    
    def __len__(self):
        return len(self.label)

class PDBbindDataset(DGLDataset):
    def __init__(self, prot_graph, lig_graph, com_graph):
        super(PDBbindDataset, self).__init__(name='PDBbind v2020')
        self.gp, _  = load_graphs( prot_graph )
        self.gl, _  = load_graphs( lig_graph )
        self.gc, self.label = load_graphs( com_graph )
        
        self.label = self.label['scores']
        
    def __getitem__(self, idx):
        return self.gp[idx], self.gl[idx], self.gc[idx], self.label[idx]
    
    def __len__(self):
        return len(self.label)
    
    # def preprocessing(self, g):
        
        
class PDBbindDecoyDataset(DGLDataset):
    def __init__(self, prot_graph, lig_graph, com_graph):
        super(PDBbindDecoyDataset, self).__init__(name='PDBbind v2020')
        self.gp, _  = load_graphs( prot_graph )
        self.gl, _  = load_graphs( lig_graph )
        self.gc, self.label = load_graphs( com_graph )

        self.pkd = self.label['scores']
        self.rmsd = self.label['rmsds']
        self.binary = self.label['binary']
        
    def __getitem__(self, idx):
        return self.gp[idx], self.gl[idx], self.gc[idx], self.pkd[idx], self.rmsd[idx], self.binary[idx]
    
    def __len__(self):
        return len(self.gc)