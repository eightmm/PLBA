import dgl, numpy
import torch as th

import torch.nn as nn
import torch.nn.functional as F

def run_train_epoch(model, data_loader, optimizer, scheduler, device='cpu'):
    model.train()
        
    loss_total  = 0
    loss_pkd    = 0
    loss_rmsd   = 0
    loss_binary = 0
    
    for batch_idx, batch_data in enumerate(data_loader):
        bgp, bgl, bgc, pkd, rmsd, binary = batch_data
        bgp, bgl, bgc, pkd, rmsd, binary = bgp.to(device), bgl.to(device), bgc.to(device), pkd.to(device), rmsd.to(device), binary.to(device)

        predicted_pkd, predicted_rmsd, predicted_binary = model( bgp, bgl ,bgc )
        
        pkd_loss  = F.mse_loss( predicted_pkd, pkd )
        rmsd_loss = F.mse_loss( predicted_rmsd, rmsd )
        binary_loss = F.binary_cross_entropy( predicted_binary, binary )
        
        loss = pkd_loss + rmsd_loss + binary_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_total  += loss.item() * bgl.batch_size
        loss_pkd    += pkd_loss.item() * bgl.batch_size
        loss_rmsd   += rmsd_loss.item() * bgl.batch_size
        loss_binary += binary_loss.item() * bgl.batch_size
        
        th.cuda.empty_cache()
        
        del bgl, bgp, bgc, pkd, rmsd, binary, predicted_pkd, predicted_rmsd, predicted_binary, pkd_loss, rmsd_loss, binary_loss, loss
    scheduler.step()
    
    divisor = len(data_loader.dataset)

    return loss_total/divisor, loss_pkd/divisor, loss_rmsd/divisor, loss_binary/divisor

# @th.no_grad()
def run_eval_epoch(model, data_loader, device='cpu'):
    model.eval()
    
    true_pkd = []
    true_rmsd = []
    true_binary = []
    
    pred_pkd = []
    pred_rmsd = []
    pred_binary = []
    
    loss_total  = 0
    loss_pkd    = 0
    loss_rmsd   = 0
    loss_binary = 0
    
    with th.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            bgp, bgl, bgc, pkd, rmsd, binary = batch_data
            bgp, bgl, bgc, pkd, rmsd, binary = bgp.to(device), bgl.to(device), bgc.to(device), pkd.to(device), rmsd.to(device), binary.to(device)
            
            true_pkd.extend( pkd )
            true_rmsd.extend( rmsd )
            true_binary.extend( binary )
            
            predicted_pkd, predicted_rmsd, predicted_binary = model( bgp, bgl ,bgc )

            pred_pkd.extend( predicted_pkd )
            pred_rmsd.extend( predicted_rmsd )
            pred_binary.extend( predicted_binary )
            
            pkd_loss  = F.mse_loss( predicted_pkd, pkd )
            rmsd_loss = F.mse_loss( predicted_rmsd, rmsd )
            binary_loss = F.binary_cross_entropy( predicted_binary, binary )

            loss = pkd_loss + rmsd_loss + binary_loss

            loss_total  += loss.item() * bgl.batch_size
            loss_pkd    += pkd_loss.item() * bgl.batch_size
            loss_rmsd   += rmsd_loss.item() * bgl.batch_size
            loss_binary += binary_loss.item() * bgl.batch_size

            th.cuda.empty_cache()

            del bgl, bgp, bgc, pkd, rmsd, binary, predicted_pkd, predicted_rmsd, predicted_binary, pkd_loss, rmsd_loss, binary_loss, loss
            
    divisor = len(data_loader.dataset)            
            
    true_pkd = th.tensor( true_pkd )
    true_rmsd = th.tensor( true_rmsd )
    true_binary = th.tensor( true_binary )
    
    pred_pkd = th.tensor( pred_pkd )
    pred_rmsd = th.tensor( pred_rmsd )
    pred_binary  = th.tensor( pred_binary )

    return loss_total/divisor, loss_pkd/divisor, loss_rmsd/divisor, loss_binary/divisor, true_pkd, true_rmsd, true_binary, pred_pkd, pred_rmsd, pred_binary
