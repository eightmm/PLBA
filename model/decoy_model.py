import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling

class DecoyPredictionModel(nn.Module):
    def __init__(self, cmodel, out_dim):
        super(DecoyPredictionModel, self).__init__()
        self.cmodel = cmodel
        
        self.pKd = nn.Sequential(nn.Linear(out_dim, out_dim), 
                                             nn.BatchNorm1d(out_dim), 
                                             nn.ELU(),
                                             nn.Linear(out_dim, 1), 
                                ) 
        
        self.RMSD = nn.Sequential(nn.Linear(out_dim, out_dim), 
                                             nn.BatchNorm1d(out_dim), 
                                             nn.ELU(),
                                             nn.Linear(out_dim, 1), 
                                ) 
        
        self.Binary = nn.Sequential(nn.Linear(out_dim, out_dim), 
                                             nn.BatchNorm1d(out_dim), 
                                             nn.ELU(),
                                             nn.Linear(out_dim, 1), 
                                             nn.Sigmoid()
                                ) 
        
        self.pooling = SumPooling()
        
    def forward(self, gp, gl, gc):
        hp, hl, hc = self.cmodel(gp, gl, gc)
        
        hc_pooling = self.pooling(gc, hc)
        hl_pooling = self.pooling(gl, hl)
        
        pKd = self.pKd( hl_pooling ).squeeze(1)
        RMSD = self.RMSD( hl_pooling ).squeeze(1)
        Binary = self.Binary( hc_pooling ).squeeze(1)

        return pKd, RMSD, Binary
    