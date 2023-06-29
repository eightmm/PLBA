import dgl, numpy
import torch as th

import torch.nn as nn
import torch.nn.functional as F
# import torchbnn as bnn

from torch.distributions import Normal
from torch_scatter import scatter_add

    
def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
    # MU=kind of predicted distance, SIGMA=distribution, PI=coeffcient, Y=true distance
    normal = Normal(mu, sigma) ## make normal distribution based on givec MU & SIGMA
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)
    return loss
    
def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += th.log(pi)
    prob = logprob.exp().sum(1)
    return prob

def run_train_epoch(model, data_loader, optimizer, scheduler, device='cpu'):
    model.train()
        
    total_loss  = 0
    mdn_loss    = 0
    ba_loss = 0
    
    for batch_idx, batch_data in enumerate(data_loader):
        bgp, bgl, bgc, score = batch_data
        
        bgp, bgl, bgc, score = bgp.to(device), bgl.to(device), bgc.to(device), score.to(device)

        pi, sigma, mu, dist, batch, binding_affinity = model(bgp, bgl, bgc)
            
        ba = F.mse_loss( binding_affinity, score )
            
        mdn = mdn_loss_fn(pi, sigma, mu, dist)
        mdn = mdn[th.where(dist <= 5)[0]]
        mdn = mdn.mean()
            
        loss = mdn + ba 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * bgl.batch_size
        mdn_loss   += mdn.item()  * bgl.batch_size
        ba_loss    += ba.item()   * bgl.batch_size
        
        th.cuda.empty_cache()
        
        del bgl, bgp, bgc, pi, sigma, mu, dist, batch, mdn, loss
    scheduler.step()
    
    divisor = len(data_loader.dataset)

    return total_loss/divisor, mdn_loss/divisor, ba_loss/divisor


def run_eval_epoch(model, data_loader, device='cpu'):
    model.eval()
    
    true = []
    mdns  = []
    pkd  = []
    
    total_loss = 0
    mdn_loss    = 0
    ba_loss = 0
    
    with th.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            bgp, bgl, bgc, score = batch_data
            bgp, bgl, bgc, score = bgp.to(device), bgl.to(device), bgc.to(device), score.to(device)
            true.extend(score)
            
            pi, sigma, mu, dist, batch, binding_affinity = model(bgp, bgl, bgc)
            
            ba = F.mse_loss( binding_affinity, score )
            
            mdn = mdn_loss_fn(pi, sigma, mu, dist)
            mdn = mdn[th.where(dist <= 5)[0]]
            mdn = mdn.mean()
            
            loss = mdn + ba 
            loss = ba

            total_loss += loss.item() * bgl.batch_size
            mdn_loss   += mdn.item()  * bgl.batch_size
            ba_loss    += ba.item()   * bgl.batch_size
            
            prob = calculate_probablity(pi, sigma, mu, dist)
            batch = batch.to(device)
            probx = scatter_add(prob, batch, dim=0, dim_size=bgl.batch_size)
            mdns.append(probx)
            pkd.append(binding_affinity)
            
            th.cuda.empty_cache()

    divisor = len(data_loader.dataset)
    mdns = th.cat(mdns)
    pkd  = th.cat(pkd)

    true = th.tensor( true ).to(device)
    
    return total_loss/divisor, mdn_loss/divisor, ba_loss/divisor, true, mdns, pkd

def run_train_epoch(model, data_loader, optimizer, scheduler, device='cpu'):
    model.train()
        
    total_loss  = 0
    for batch_idx, batch_data in enumerate(data_loader):
        bgp, bgl, bgc, score = batch_data
        
        bgp, bgl, bgc, score = bgp.to(device), bgl.to(device), bgc.to(device), score.to(device)
        
        binding_affinity = model(bgp, bgl, bgc)

        loss = F.mse_loss( binding_affinity, score )
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * bgl.batch_size
        th.cuda.empty_cache()
        
    scheduler.step()
    
    divisor = len(data_loader.dataset)

    return total_loss/divisor


def run_eval_epoch(model, data_loader, device='cpu'):
    model.eval()
    
    true = []
    pkd  = []
    
    total_loss = 0
    
    with th.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            bgp, bgl, bgc, score = batch_data
            bgp, bgl, bgc, score = bgp.to(device), bgl.to(device), bgc.to(device), score.to(device)
            true.extend(score)
            
            binding_affinity = model(bgp, bgl, bgc)

            loss = F.mse_loss( binding_affinity, score )

            total_loss += loss.item() * bgl.batch_size
            pkd.extend(binding_affinity)
            
            th.cuda.empty_cache()

    divisor = len(data_loader.dataset)
    pkd  = th.tensor( pkd )
    true = th.tensor( true )
    
    return total_loss/divisor, true, pkd



def run_eval_docking(model, data_loader, device='cpu'):
    model.eval()
    
    pkd  = []
    
    with th.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            bgp, bgl, bgc = batch_data[0][0].to(device), batch_data[1][0].to(device), batch_data[2][0].to(device)
            
            binding_affinity = model(bgp, bgl, bgc)

            pkd.extend(binding_affinity)
            
            th.cuda.empty_cache()

    pkd  = th.tensor( pkd )
    
    return pkd