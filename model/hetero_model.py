import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling
import torchbnn as bnn


def to_dense_batch_dgl(bg, feats, fill_value=0):
    max_num_nodes = int(bg.batch_num_nodes().max())
    batch = th.cat([th.full((1,x.type(th.int)), y) for x,y in zip(bg.batch_num_nodes(),range(bg.batch_size))],dim=1).reshape(-1).type(th.long).to(bg.device)
    cum_nodes = th.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
    idx = th.arange(bg.num_nodes(), dtype=th.long, device=bg.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
    out = feats.new_full(size, fill_value)
    out[idx] = feats
    out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])

    mask = th.zeros(bg.batch_size * max_num_nodes, dtype=th.bool, device=bg.device)
    mask[idx] = 1
    mask = mask.view(bg.batch_size, max_num_nodes)  
    return out, mask

class PredictionScore(nn.Module):
    def __init__(self, lmodel, pmodel, plmodel, lpmodel, out_dim, batch_size):
        super(PredictionScore, self).__init__()
        self.lmodel = lmodel
        self.pmodel = pmodel
        
        self.plmodel = plmodel
        self.lpmodel = lpmodel
        
        self.batch_size = batch_size
        
        self.MLP = nn.Sequential( nn.Linear(out_dim*2, out_dim), 
                                nn.BatchNorm1d(out_dim), 
                                nn.ELU(), 
                                nn.Dropout(p=0.1) ) 
        
        self.z_pi = nn.Linear(out_dim, 10)
        self.z_sigma = nn.Linear(out_dim, 10)
        self.z_mu = nn.Linear(out_dim, 10)
        
        
    def forward(self, hg):
        p_batch_num_nodes = hg.batch_num_nodes('p')
        l_batch_num_nodes = hg.batch_num_nodes('l')
        
        batch_num_nodes = p_batch_num_nodes + l_batch_num_nodes
        batch_num_edges = hg.batch_num_edges('pl')
        
        gl = hg.node_type_subgraph(['l'])
        gp = hg.node_type_subgraph(['p'])
        
        gl.set_batch_num_nodes( l_batch_num_nodes )
        gp.set_batch_num_nodes( p_batch_num_nodes )
        
        hl, cl, el = self.lmodel(gl, gl.ndata['feature'], gl.ndata['pos'], gl.edata['feature'])
        hp, cp, ep = self.pmodel(gp, gp.ndata['feature'], gp.ndata['pos'], gp.edata['feature'])
        
#         gpl_hetero = hg.edge_type_subgraph(['pl'])
#         glp_hetero = hg.edge_type_subgraph(['lp'])

#         gpl = dgl.to_homogeneous( gpl_hetero, edata=['feature'] )
#         glp = dgl.to_homogeneous( glp_hetero, edata=['feature'] )
        
#         gpl.ndata['feature'] = th.cat( [hp, hl] )
#         gpl.ndata['pos']     = th.cat( [cp, cl] )
        
#         glp.ndata['feature'] = th.cat( [hl, hp] )
#         glp.ndata['pos']     = th.cat( [cl, cp] )

#         hpl, cpl, epl = self.plmodel(gpl, gpl.ndata['feature'], gpl.ndata['pos'], gpl.edata['feature'])
#         hlp, clp, elp = self.lpmodel(glp, glp.ndata['feature'], glp.ndata['pos'], glp.edata['feature'])
                
        h_l_x, l_mask = to_dense_batch_dgl(gl, hl, fill_value=0)
        h_p_x, p_mask = to_dense_batch_dgl(gp, hp, fill_value=0)
        
        h_l_pos, _ =  to_dense_batch_dgl(gl, gl.ndata["pos"], fill_value=0)
        h_p_pos, _ =  to_dense_batch_dgl(gp, gp.ndata["pos"], fill_value=0)
        
        (B, N_l, C_out), N_p = h_l_x.size(), h_p_x.size(1)
        
        self.B = B
        self.N_l = N_l
        self.N_p = N_p

        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_p, 1) # [B, N_l, N_t, C_out]
        
        h_p_x = h_p_x.unsqueeze(-3)
        h_p_x = h_p_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]
        C = th.cat((h_l_x, h_p_x), -1)
        
        C_mask = l_mask.view(B, N_l, 1) & p_mask.view(B, 1, N_p)
        C = C[C_mask]
        
        C = self.MLP(C)
        
        C_batch = th.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_p)[C_mask]
            
        mu    = F.elu(self.z_mu(C)) + 1
        sigma = F.elu(self.z_sigma(C)) + 1.1
        pi    = F.softmax(self.z_pi(C), -1)
        
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_p_pos.view(B,-1,3))[C_mask]

        return pi, sigma, mu, dist.unsqueeze(1).detach(), C_batch
    
    def compute_euclidean_distances_matrix(self, X, Y):
        X = X.double()
        Y = Y.double()
        
        dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y**2, axis=-1).unsqueeze(1) + th.sum(X**2, axis=-1).unsqueeze(-1)
        
        return dists ** 0.5