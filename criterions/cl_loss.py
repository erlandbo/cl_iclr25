import torch
import torch.nn.functional as F
from torch import nn

# https://github.com/berenslab/t-simcne/blob/master/tsimcne/tsimcne.py

class InfoNCECosine(nn.Module):
    def __init__(
        self,
        temperature: float = 0.5,
        reg_coef: float = 0,
        reg_radius: float = 200,
    ):
        super().__init__()
        self.temperature = temperature
        self.reg_coef = reg_coef
        self.reg_radius = reg_radius

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        # mean deviation from the sphere with radius `reg_radius`
        vecnorms = torch.linalg.vector_norm(features, dim=1)
        target = torch.full_like(vecnorms, self.reg_radius)
        penalty = self.reg_coef * F.mse_loss(vecnorms, target)

        a = F.normalize(a)
        b = F.normalize(b)

        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        # mean of the diagonal
        tempered_alignment = cos_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float("-inf"))
        cos_bb.masked_fill_(self_mask, float("-inf"))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2) + penalty
        return loss


class InfoNCECauchy(nn.Module):
    def __init__(self, temperature: float = 1, exaggeration: float = 1):
        super().__init__()
        self.temperature = temperature
        self.exaggeration = exaggeration

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)

        tempered_alignment = torch.diagonal(sim_ab).log().mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(self.exaggeration * tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCEGaussian(InfoNCECauchy):
    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = -(torch.cdist(a, a) * self.temperature).square()
        sim_bb = -(torch.cdist(b, b) * self.temperature).square()
        sim_ab = -(torch.cdist(a, b) * self.temperature).square()

        tempered_alignment = sim_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float("-inf"))
        sim_bb.masked_fill_(self_mask, float("-inf"))

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, metric, temp):
        super().__init__()

        #metric = self.metric
        if metric == "cosine":
            self.cls = InfoNCECosine(temperature=temp)
        elif metric == "cauchy":  # actually Cauchy
            self.cls = InfoNCECauchy()
        elif metric == "gauss":
            self.cls = InfoNCEGaussian()
        else:
            raise ValueError(f"Unknown {metric = !r} for InfoNCE loss")

    def forward(self, hidden, idx):
        return self.cls(hidden)
    


###############################

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class SogCLR(nn.Module):
    def __init__(self, T=0.1, loss_type='dcl', N=50000):
        """
        T: softmax temperature (default: 1.0)
        """
        super(SogCLR, self).__init__()
        self.T = T
        self.N = N
        self.loss_type = loss_type
        
        # for DCL
        self.u = torch.zeros(N).reshape(-1, 1) #.to(self.device) 
        self.LARGE_NUM = 1e9

    def dynamic_contrastive_loss(self, hidden1, hidden2, index=None, gamma=0.9, distributed=False):
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]
        
        # Gather hidden1/hidden2 across replicas and create local labels.
        if distributed:  
           hidden1_large = torch.cat(all_gather_layer.apply(hidden1), dim=0) # why concat_all_gather()
           hidden2_large =  torch.cat(all_gather_layer.apply(hidden2), dim=0)
           enlarged_batch_size = hidden1_large.shape[0]

           labels_idx = (torch.arange(batch_size, dtype=torch.long) + batch_size  * torch.distributed.get_rank()).to(hidden1.device) 
           labels = F.one_hot(labels_idx, enlarged_batch_size*2).to(hidden1.device) 
           masks  = F.one_hot(labels_idx, enlarged_batch_size).to(hidden1.device) 
           batch_size = enlarged_batch_size
        else:
           hidden1_large = hidden1
           hidden2_large = hidden2
           labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(hidden1.device) 
           masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(hidden1.device) 

        logits_aa = torch.matmul(hidden1, hidden1_large.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T)
        logits_ba = torch.matmul(hidden2, hidden1_large.T)

        #  SogCLR
        neg_mask = 1-labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1)
        logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1)
      
        neg_logits1 = torch.exp(logits_ab_aa /self.T)*neg_mask   #(B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb /self.T)*neg_mask

        # u init    
        if self.u[index.cpu()].sum() == 0:
            gamma = 1
            
        u1 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits1, dim=1, keepdim=True)/(2*(batch_size-1))
        u2 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits2, dim=1, keepdim=True)/(2*(batch_size-1))

        # this sync on all devices (since "hidden" are gathering from all devices)
        if distributed:
           u1_large = concat_all_gather(u1)
           u2_large = concat_all_gather(u2)
           index_large = concat_all_gather(index)
           self.u[index_large.cpu()] = (u1_large.detach().cpu() + u2_large.detach().cpu())/2 
        else:
           self.u[index.cpu()] = (u1.detach().cpu() + u2.detach().cpu())/2 

        p_neg_weights1 = (neg_logits1/u1).detach()
        p_neg_weights2 = (neg_logits2/u2).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(2*(batch_size-1))
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss
    
    def forward(self, x, index, gamma=0.9):
        """
        Input:
            x1: first views of images
            x2: second views of images
            index: index of image
            gamma: moving average of sogclr 
        Output:
            loss
        """
        # compute features
        batch_size = x.size(0) // 2

        x1 = x[:batch_size]
        x2 = x[batch_size:]

        loss = self.dynamic_contrastive_loss(x1, x2, index, gamma) 
        return loss

