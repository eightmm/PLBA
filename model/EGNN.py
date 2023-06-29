import torch, dgl
import torch.nn as nn

from dgl.nn import EGNNConv

class EGNNLayer(nn.Module): # norm - activation - conolution - skip add
    def __init__(self, embedding_size, dropout_ratio=0.1):
        super(EGNNLayer, self).__init__()
        
        self.egnn_layer = EGNNConv( embedding_size, embedding_size, embedding_size, embedding_size )
        
        self.norm_layer = nn.LayerNorm( embedding_size )
        
        self.drop_out   = nn.Dropout( dropout_ratio )
        
        self.activation = nn.ELU()
        
    def forward(self, g, h, c, e):
        with g.local_scope():
            h_i = h
            
            h = self.norm_layer( h )
            
            h = self.activation( h )

            h, c = self.egnn_layer(g, h, c, e)
            
            h = h + h_i
            
            h = self.drop_out( h )
                        
            return h, c
            
            
# class EGNNNet(nn.Module):
#     def __init__(self, input_size, embedding_size, edge_size, num_layers, pos_enc_k=20, dropout_ratio=0.2):
#         super(EGNNNet, self).__init__()
        
#         # self.k = pos_enc_k
        
#         # self.node_encoder = BayesLinear( prior_mu=0, prior_sigma=0.15, in_features=input_size, out_features=embedding_size )
#         # self.edge_encoder = BayesLinear( prior_mu=0, prior_sigma=0.15, in_features=edge_size,  out_features=embedding_size )
        
#         self.node_encoder = nn.Linear( input_size, embedding_size )
#         self.edge_encoder = nn.Linear( edge_size,  embedding_size )
        
#         self.norm_layer = nn.LayerNorm( embedding_size )
        
#         egnn_blocks = [ EGNNLayer( embedding_size = embedding_size,
#                                     dropout_ratio = dropout_ratio )
#                             for _ in range(num_layers) ]
        
#         self.egnn_blocks = nn.ModuleList( egnn_blocks )

#     def forward(self, g, h, c, e):
#         # pos_enc = dgl.random_walk_pe(g, self.k)
#         # h = torch.cat( [h, pos_enc], dim=1 )
        
#         h = self.node_encoder( h )
#         e = self.edge_encoder( e )
        
#         h = self.norm_layer( h )
#         h_i = h
        
#         for layer in self.egnn_blocks:
#             h, c, e = layer(g, h, c, e)
#             h = h + h_i
        
#         return h, c, e
            
        
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
            hp, ep = protein_layer( gp, hp, cp, ep )
            hl, el = ligand_layer( gl, hl, cl, el )
            
            hc = []
            for gp_start, gp_size, gl_start, gl_size in zip(gp_start_indices, gp_batch_sizes, gl_start_indices, gl_batch_sizes):
                gp_slice = hp[gp_start:gp_start + gp_size]
                gl_slice = hl[gl_start:gl_start + gl_size]
                hc.append( torch.cat( [gp_slice, gl_slice] ) )
            hc = torch.cat( hc )
            # hc_i = hc
            hc, ec = complex_layer( gc, hc, cc, ec )
            
            # hc -= hc_i
            
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

            hp += hp_i
            hl += hl_i
            
        return hp, hl, hc
            