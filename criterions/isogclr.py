import torch
from torch import nn
from torch.nn import functional as F

"""
    follow: https://github.com/Optimization-AI/SogCLR
"""
class SogCLR_Loss(nn.Module):
    def __init__(self, N=15000000, gamma=0.8, temperature=0.1):
        super(SogCLR_Loss, self).__init__()
        self.gamma = gamma
        self.u = torch.zeros(N).cuda()
        self.LARGE_NUM = 1e9
        self.T = temperature

    def forward(self, hidden, index):
        hidden = F.normalize(hidden, dim=1)
        batch_size = hidden.shape[0] // 2
        hidden1 = hidden[:batch_size]
        hidden2 = hidden[batch_size:]

        labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).cuda() # [B, 2*B]
        masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).cuda()     # [B, B]
        #import pdb; pdb.set_trace()
        logits_aa = torch.matmul(hidden1, hidden1.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2.T)
        logits_ba = torch.matmul(hidden2, hidden1.T)

        #  SogCLR
        neg_mask = 1-labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa], 1) 
        logits_ba_bb = torch.cat([logits_ba, logits_bb], 1) 
      
        neg_logits1 = torch.exp(logits_ab_aa / self.T) * neg_mask   #(B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb / self.T) * neg_mask

        u1 = (1 - self.gamma) * self.u[index] + self.gamma * torch.sum(neg_logits1, dim=1, keepdim=False) / (2*(batch_size-1))
        u2 = (1 - self.gamma) * self.u[index] + self.gamma * torch.sum(neg_logits2, dim=1, keepdim=False) / (2*(batch_size-1))
        
        self.u[index] = u1.detach()+ u2.detach()

        p_neg_weights1 = (neg_logits1 / u1[:, None]).detach()
        p_neg_weights2 = (neg_logits2 / u2[:, None]).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(2*(batch_size-1))
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss


class SogCLR_DRO_Loss(nn.Module):
    def __init__(self, N=15000000, gamma=0.8, tau_init=0.5, tau_min=0.05, tau_max=1.0, rho=0.8, bsz=128,
                    eta_init=0.001, eta_min=1e-4, beta_u=0.9, eta_sched='const', eta_exp_gamma=0.8):
        super(SogCLR_DRO_Loss, self).__init__()
        self.gamma = gamma
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rho = rho
        self.eta_init = eta_init
        self.eta_min = eta_min
        self.beta_u = beta_u
        self.eta_sched = eta_sched
        self.eta_exp_gamma = eta_exp_gamma
        self.s = torch.zeros(N).cuda()
        self.tau = torch.ones(N).cuda() * self.tau_init
        self.u = torch.zeros(N).cuda()
        self.b = torch.zeros(N).cuda()
        self.eps = 1e-8
        self.grad_clip = 3.0

        self.mask_neg = (1.0 - torch.eye(bsz)).repeat(2,2).cuda()
        self.num_neg = 2 * bsz - 2
        self.epoch = 0
        #self.mask_neg = (1.0 - torch.eye(bsz * 2)).cuda()
        #self.num_neg = 2 * bsz - 1


    def forward(self, features, index):
        #Compute SogCLR_DRO loss
        contrast_feature = F.normalize(features, dim=1)
        epoch = self.epoch
        #Args:
        #    index: index of each sample [bsz].  e.g. [512]
        #    features: hidden vector of shape [bsz, n_views, dim].  e.g. [512, 2, 128]
        #Returns:
        #    A loss scalar.
        
        #contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape: [bsz * 2, dim]
        bsz = contrast_feature.shape[0] // 2

        sim = torch.einsum('i d, j d -> i j', contrast_feature, contrast_feature) # [bsz * 2, bsz * 2]

        pos_sim = torch.cat([torch.diagonal(sim, offset=bsz), torch.diagonal(sim, offset=-bsz)])[:, None] # [bsz * 2, 1]

        tau = self.tau[index].repeat(2)

        sim_d_temps = (sim / tau[:, None]).clone().detach_()
 
        exp_sim_d_temps = torch.exp(sim_d_temps) * self.mask_neg

        g = torch.sum(exp_sim_d_temps, dim=1, keepdim=True) / self.num_neg

        if epoch == 0:
            s1 = g.squeeze()[:bsz]
            s2 = g.squeeze()[bsz:]
        else:
            s1 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[:bsz]
            s2 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[bsz:]

        self.s[index] = (s1 + s2) / 2.0

        weights_1 = exp_sim_d_temps[:bsz, :] / s1[:, None]
        loss_1 = torch.sum(weights_1 * sim[:bsz, :], dim=1, keepdim=True) / self.num_neg - pos_sim[:bsz, :]

        weights_2 = exp_sim_d_temps[bsz:, :] / s2[:, None]
        loss_2 = torch.sum(weights_2 * sim[bsz:, :], dim=1, keepdim=True) / self.num_neg - pos_sim[bsz:, :]

        loss = loss_1 + loss_2

        # gradient of tau
        grad_tau_1 = torch.log(s1) + self.rho - torch.sum(weights_1 * sim_d_temps[:bsz, :], dim=1, keepdim=False) / self.num_neg
        grad_tau_2 = torch.log(s2) + self.rho - torch.sum(weights_2 * sim_d_temps[bsz:, :], dim=1, keepdim=False) / self.num_neg 

        grad_tau = ((grad_tau_1 + grad_tau_2) / 2.0).clamp_(min=-self.grad_clip, max=self.grad_clip)
        
        self.u[index] = (1.0-self.beta_u) * self.u[index] + self.beta_u * grad_tau

        self.tau[index] = (self.tau[index] - self.eta_init * self.u[index]).clamp_(min=self.tau_min, max=self.tau_max)
        
        avg_tau = tau.mean()

        self.epoch = 1
        
        return loss.mean()#, avg_tau, self.eta_init, grad_tau.mean().item(), 0.0 #old_b.mean().item()

