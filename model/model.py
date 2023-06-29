import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling

def calculate_pair_distance(arr1, arr2):
    return th.linalg.norm( arr1[:, :, None, :] - arr2[:, None, :, :], axis = -1)

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

class MixtureDensityNetwork(nn.Module):
    def __init__(self, in_dim, num_gaussian=10, eps=1e-10):
        super(MixtureDensityNetwork, self).__init__()
        self.eps = eps
        
        self.pi    = nn.Linear(in_dim, num_gaussian)
        self.sigma = nn.Linear(in_dim, num_gaussian)
        self.mu    = nn.Linear(in_dim, num_gaussian)
        
    def forward(self, h):
        pi    = F.softmax( self.pi(h), -1 ) + self.eps
        sigma = F.elu( self.sigma(h) ) + 1.1 + self.eps
        mu    = F.elu( self.mu(h) ) + 1 + self.eps
        
        return pi, sigma, mu
    
# class PredictionScore(nn.Module):
#     def __init__(self, cmodel, out_dim, num_gaussian=10):
#         super(PredictionScore, self).__init__()
#         # self.lmodel = lmodel
#         # self.pmodel = pmodel
#         self.cmodel = cmodel
        
#         self.MLP = nn.Sequential(nn.Linear(out_dim*2, out_dim), 
#                                 nn.BatchNorm1d(out_dim), 
#                                 nn.ELU(),  
#                                 ) 
        
#         self.BindingAffinity = nn.Sequential(nn.Linear(out_dim*2, out_dim), 
#                                 nn.BatchNorm1d(out_dim), 
#                                 nn.ELU(),  
#                                 ) 
        
#         self.atom_types = nn.Linear(out_dim, 17)
#         self.is_rotate = nn.Linear(out_dim*2, 1)
        
#         self.MDN = MixtureDensityNetwork(out_dim, num_gaussian=num_gaussian)
        
#     def forward(self, gp, gl, gc):
#         # hl, cl, el = self.lmodel(gl, gl.ndata['feats'], gl.ndata['coord'], gl.edata['feats'])
#         # hp, cp, ep = self.pmodel(gp, gp.ndata['feats'], gp.ndata['coord'], gp.edata['feats'])
        
#         # hlp = th.cat( [hp, hl], dim=0 )
#         # clp = th.cat( [cp, cl], dim=0 )
#         # elp = th.cat( [ep, el], dim=1 )
        
#         # hlp, clp, elp = self.cmodel(gc, hlp, clp, gc.edata['feats'])
        
#         hp, hl, hc = self.cmodel(gp, gl, gc)
        
#         h_l_x, l_mask = to_dense_batch_dgl(gl, hl, fill_value=0)
#         h_p_x, p_mask = to_dense_batch_dgl(gp, hp, fill_value=0)
        
#         h_l_pos, _ =  to_dense_batch_dgl(gl, gl.ndata["coord"], fill_value=0)
#         h_p_pos, _ =  to_dense_batch_dgl(gp, gp.ndata["coord"], fill_value=0)
        
#         (B, N_l, C_out), N_p = h_l_x.size(), h_p_x.size(1)
        
#         self.B = B
#         self.N_l = N_l
#         self.N_p = N_p

#         # Combine and mask
#         h_l_x = h_l_x.unsqueeze(-2)
#         h_l_x = h_l_x.repeat(1, 1, N_p, 1) # [B, N_l, N_t, C_out]
        
#         h_p_x = h_p_x.unsqueeze(-3)
#         h_p_x = h_p_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]
#         C = th.cat((h_l_x, h_p_x), -1)
        
#         C_mask = l_mask.view(B, N_l, 1) & p_mask.view(B, 1, N_p)
#         C = C[C_mask]

#         C = self.MLP(C)
        
#         C_batch = th.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
#         C_batch = C_batch.repeat(1, N_l, N_p)[C_mask]
            
#         pi, sigma, mu = self.MDN(C)
        
#         dist = self.compute_euclidean_distances_matrix(h_l_pos, h_p_pos.view(B,-1,3))[C_mask]

#         is_rotate = self.is_rotate( th.cat( [hl[gl.edges()[0]], hl[gl.edges()[1]] ], axis=1) ).squeeze(dim=1)

#         return pi, sigma, mu, dist.unsqueeze(1).detach(), C_batch, is_rotate
    
#     def compute_euclidean_distances_matrix(self, X, Y):
#         X = X.double()
#         Y = Y.double()
#         dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y**2, axis=-1).unsqueeze(1) + th.sum(X**2, axis=-1).unsqueeze(-1)
        
#         return dists ** 0.5
    
    
class PredictionScore(nn.Module):
    def __init__(self, cmodel, out_dim, num_gaussian=10):
        super(PredictionScore, self).__init__()
        self.cmodel = cmodel
        
        self.MLP = nn.Sequential(nn.Linear(out_dim*2, out_dim), 
                                nn.BatchNorm1d(out_dim), 
                                nn.ELU(),  
                                ) 
        
        self.BindingAffinity = nn.Sequential(nn.Linear(out_dim, out_dim*2), 
                                             nn.BatchNorm1d(out_dim*2), 
                                             nn.ELU(),
                                             nn.Linear(out_dim*2, out_dim), 
                                             nn.BatchNorm1d(out_dim), 
                                             nn.ELU(),
                                             nn.Linear(out_dim, 1), 
                                ) 
        
        self.pooling = SumPooling()
        
        self.atom_types = nn.Linear(out_dim, 17)
        self.is_rotate = nn.Linear(out_dim*2, 1)
        
        self.MDN = MixtureDensityNetwork(out_dim, num_gaussian=num_gaussian)
        
    def forward(self, gp, gl, gc):
        hp, hl, hc = self.cmodel(gp, gl, gc)
        h = self.pooling(gl, hl)
        binding_affinity = self.BindingAffinity( h )
        
        h_l_x, l_mask = to_dense_batch_dgl(gl, hl, fill_value=0)
        h_p_x, p_mask = to_dense_batch_dgl(gp, hp, fill_value=0)
        
        h_l_pos, _ =  to_dense_batch_dgl(gl, gl.ndata["coord"], fill_value=0)
        h_p_pos, _ =  to_dense_batch_dgl(gp, gp.ndata["coord"], fill_value=0)
        
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
            
        pi, sigma, mu = self.MDN(C)
        
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_p_pos.view(B,-1,3))[C_mask]

        return pi, sigma, mu, dist.unsqueeze(1).detach(), C_batch, binding_affinity.squeeze(1)
    
    def compute_euclidean_distances_matrix(self, X, Y):
        X = X.double()
        Y = Y.double()
        dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y**2, axis=-1).unsqueeze(1) + th.sum(X**2, axis=-1).unsqueeze(-1)
        
        return dists ** 0.5
    
    
class PredictionPKD(nn.Module):
    def __init__(self, cmodel, out_dim):
        super(PredictionPKD, self).__init__()
        self.cmodel = cmodel
        
        self.BindingAffinity = nn.Sequential(nn.Linear(out_dim, out_dim), 
                                             nn.BatchNorm1d(out_dim), 
                                             nn.ELU(),
                                             nn.Linear(out_dim, 1), 
                                ) 
        
        self.pooling = SumPooling()
        
    def forward(self, gp, gl, gc):
        hp, hl, hc = self.cmodel(gp, gl, gc)
        h = self.pooling(gl, hl)
        binding_affinity = self.BindingAffinity( h )

        return binding_affinity.squeeze(1)
    