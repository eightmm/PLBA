import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from .GatedGCNLSPE import GatedGCNLSPELayer
from .EGNN import EGNNLayer

class GatedGCNLSPENet(nn.Module):
    def __init__(self, input_size, embedding_size, edge_size, com_edge_size, pose_size, num_layers, dropout_ratio=0.2):
        super(GatedGCNLSPENet, self).__init__()
        
        self.protein_node_encoder = nn.Linear( input_size, embedding_size )
        self.protein_edge_encoder = nn.Linear( edge_size,  embedding_size )
        self.protein_pose_encoder = nn.Linear( pose_size,  embedding_size )
        
        self.ligand_node_encoder = nn.Linear( input_size, embedding_size )
        self.ligand_edge_encoder = nn.Linear( edge_size,  embedding_size )
        self.ligand_pose_encoder = nn.Linear( pose_size,  embedding_size )
        
        self.complex_edge_encoder = nn.Linear( com_edge_size, embedding_size )
        
        self.protein_norm = nn.LayerNorm( embedding_size )
        self.ligand_norm  = nn.LayerNorm( embedding_size )
        
        protein_blocks = [ GatedGCNLSPELayer( input_dim=embedding_size, output_dim=embedding_size, dropout=0.15, batch_norm=True, use_lapeig_loss=False, residual=True )
                            for _ in range(num_layers) ]
        
        ligand_blocks  = [ GatedGCNLSPELayer( input_dim=embedding_size, output_dim=embedding_size, dropout=0.15, batch_norm=True, use_lapeig_loss=False, residual=True )
                            for _ in range(num_layers) ]
        
        complex_blocks = [ GatedGCNLSPELayer( input_dim=embedding_size, output_dim=embedding_size, dropout=0.15, batch_norm=True, use_lapeig_loss=False, residual=True )
                            for _ in range(num_layers) ]
        
        self.protein_blocks = nn.ModuleList( protein_blocks )
        self.ligand_blocks  = nn.ModuleList( ligand_blocks )
        self.complex_blocks = nn.ModuleList( complex_blocks )

    def forward(self, gp, gl, gc):
        hp = self.protein_node_encoder( gp.ndata['feats'] )
        ep = self.protein_edge_encoder( gp.edata['feats'] )
        pp = self.protein_pose_encoder( gp.ndata['pos_enc'] )
        
        hl = self.ligand_node_encoder( gl.ndata['feats'] )
        el = self.ligand_edge_encoder( gl.edata['feats'] )
        pl = self.ligand_pose_encoder( gl.ndata['pos_enc'] )
        
        ec = self.complex_edge_encoder( gc.edata['feats'] )
        
        hp = self.protein_norm( hp )
        hl = self.ligand_norm( hl )
        
        hp_raw = hp
        hl_raw = hl
        
        gp_batch_sizes = gp.batch_num_nodes()
        gl_batch_sizes = gl.batch_num_nodes()
        
        gp_start_indices = [0] + torch.cumsum(gp_batch_sizes[:-1], dim=0).tolist()
        gl_start_indices = [0] + torch.cumsum(gl_batch_sizes[:-1], dim=0).tolist()
        
        for (protein_layer, ligand_layer, complex_layer) in zip(self.protein_blocks, self.ligand_blocks, self.complex_blocks):
            hp, pp, ep = protein_layer( gp, hp, pp, ep, 1 ) #  g, h, p, e, 
            hl, pl, el = ligand_layer( gl, hl, pl, el, 1 )
            
            hc = []
            pc = []
            for gp_start, gp_size, gl_start, gl_size in zip(gp_start_indices, gp_batch_sizes, gl_start_indices, gl_batch_sizes):
                gp_slice = hp[gp_start:gp_start + gp_size]
                gl_slice = hl[gl_start:gl_start + gl_size]
                pp_slice = pp[gp_start:gp_start + gp_size]
                pl_slice = pl[gl_start:gl_start + gl_size]
                hc.append( torch.cat( [gp_slice, gl_slice] ) )
                pc.append( torch.cat( [pp_slice, pl_slice] ) )
                
            hc = torch.cat( hc )
            pc = torch.cat( pc )
            
            hc, pc, ec = complex_layer( gc, hc, pc, ec, 1 )
            
            hp_separated = []
            hl_separated = []
            start = 0
            for gp_size, gl_size in zip(gp_batch_sizes, gl_batch_sizes):
                hp_separated.append(hc[start: start + gp_size])
                start += gp_size
                hl_separated.append(hc[start: start + gl_size])
                start += gl_size
            
            hp = torch.cat(hp_separated)
            hl = torch.cat(hl_separated)

        return hp, hl, hc
    
    
class EGNNNet(nn.Module):
    def __init__(self, input_size, embedding_size, edge_size, com_edge_size, num_layers, dropout_ratio=0.2):
        super(EGNNNet, self).__init__()
        
        self.protein_node_encoder = nn.Linear( input_size, embedding_size )
        self.protein_edge_encoder = nn.Linear( edge_size,  embedding_size )
        
        self.ligand_node_encoder = nn.Linear( input_size, embedding_size )
        self.ligand_edge_encoder = nn.Linear( edge_size,  embedding_size )
        
        self.complex_edge_encoder = nn.Linear( com_edge_size, embedding_size )
        
        self.protein_norm = nn.LayerNorm( embedding_size )
        self.ligand_norm  = nn.LayerNorm( embedding_size )
        
        protein_blocks = [ EGNNLayer( embedding_size=embedding_size,
                                      dropout_ratio=dropout_ratio )
                            for _ in range(num_layers) ]
        
        ligand_blocks  = [ EGNNLayer( embedding_size=embedding_size,
                                      dropout_ratio=dropout_ratio )
                            for _ in range(num_layers) ]
        
        complex_blocks = [ EGNNLayer( embedding_size=embedding_size,
                                      dropout_ratio=dropout_ratio )
                            for _ in range(num_layers) ]
        
        self.protein_blocks = nn.ModuleList( protein_blocks )
        self.ligand_blocks  = nn.ModuleList( ligand_blocks )
        self.complex_blocks = nn.ModuleList( complex_blocks )

    def forward(self, gp, gl, gc):
        hp = self.protein_node_encoder( gp.ndata['feats'] )
        ep = self.protein_edge_encoder( gp.edata['feats'] )
        
        hl = self.ligand_node_encoder( gl.ndata['feats'] )
        el = self.ligand_edge_encoder( gl.edata['feats'] )
        
        ec = self.complex_edge_encoder( gc.edata['feats'] )
        cc = gc.ndata['coord']
        
        hp = self.protein_norm( hp )
        hl = self.ligand_norm( hl )
        
        hp_i = hp
        hl_i = hl
        
        cp = gp.ndata['coord']
        cl = gl.ndata['coord']
        
        gp_batch_sizes = gp.batch_num_nodes()
        gl_batch_sizes = gl.batch_num_nodes()
        
        gp_start_indices = [0] + torch.cumsum(gp_batch_sizes[:-1], dim=0).tolist()
        gl_start_indices = [0] + torch.cumsum(gl_batch_sizes[:-1], dim=0).tolist()
        
        for (protein_layer, ligand_layer, complex_layer) in zip(self.protein_blocks, self.ligand_blocks, self.complex_blocks):
            hp, cp = protein_layer( gp, hp, cp, ep )
            hl, cl = ligand_layer( gl, hl, cl, el )
            
            hc = []
            for gp_start, gp_size, gl_start, gl_size in zip(gp_start_indices, gp_batch_sizes, gl_start_indices, gl_batch_sizes):
                gp_slice = hp[gp_start:gp_start + gp_size]
                gl_slice = hl[gl_start:gl_start + gl_size]
                hc.append( torch.cat( [gp_slice, gl_slice] ) )
            hc = torch.cat( hc )

            hc, cc = complex_layer( gc, hc, cc, ec )
            
            hp_separated = []
            hl_separated = []
            hc_start = 0
            for gp_size, gl_size in zip(gp_batch_sizes, gl_batch_sizes):
                hp_separated.append(hc[hc_start: hc_start + gp_size])
                hc_start += gp_size
                hl_separated.append(hc[hc_start: hc_start + gl_size])
                hc_start += gl_size

            hp = torch.cat(hp_separated)
            hl = torch.cat(hl_separated)

        return hp, hl, hc
    