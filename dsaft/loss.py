import torch
from torch import Tensor

## DSAFT - rank loss function
def dsaft_rank_loss(theta, durations, events):
    '''
    theta: prediction output from DNN layers
    durations: log-scaled observed time (log(Y))
    events: right-censoring-indicator (delta)
    '''
    # compute residual e_i
    e_vector = theta.sub(durations.view(-1,1).add(1e-32).log()).neg()
    
    # evaluate loss function based on formula,
    # mul(1 / e_vector.shape[0] ** 2) : scaling constant
    return e_vector.sub(e_vector.view(-1)).mul(e_vector.sub(e_vector.view(-1))<=0).abs().mul(events.view(-1,1)).sum().mul(1 / e_vector.shape[0] ** 2)



def dsaft_mae_loss(theta, durations, events):
    '''
    theta: prediction output from DNN layers
    durations: log-scaled observed time (log(Y))
    events: right-censoring-indicator (delta)
    '''
    # compute residual e_i
    e = theta.sub(durations.view(-1, 1).add(1e-32).log()).neg()
    n = e.shape[0]
    
    # sort e_i w.r.t values and store sorted indices
    e_sorted = torch.sort(e.view(-1)).values
    e_indices = torch.sort(e.view(-1)).indices.type(torch.float32)
    
    # sort durations(log(Y)), events(delta), and theta(output from DNN) w.r.t. sorted indices of e_i
    tmp = torch.stack([e_indices, durations.view(-1), events.type(torch.float32).view(-1), theta.view(-1)], dim = 1)
    tmp = tmp[tmp[:, 0].sort()[1]]
    durations_sorted = tmp[:, 1]
    events_sorted = tmp[:, 2]
    theta_sorted = tmp[:, 3]
    
    # get risk set and removed (instances whose events had been occured)
    diff_ = e_sorted.view(-1, 1).sub(e_sorted).sub(1e-32)
    removed = diff_.abs().div(diff_).relu()
    at_risks = removed.neg().add(torch.ones(n, n))    

    # estimate survival function of e_i via KM estimator
    surv = events_sorted.div(at_risks.sum(1)).sub(1).neg().abs().mul(removed).add(at_risks).prod(dim = 1)    

    # estimate differential of F (cumulative density function) i.e. dF(u)
    h = torch.cat([surv.neg().add(1)[1:], torch.ones(1)]).sub(torch.cat([torch.zeros(1), surv.neg().add(1)[:-1]]))    

    # evaluate imputed y using conditional expected value of epsilon
    imputed = e_sorted.mul(h).mul(at_risks).sum(1).div(surv).add(theta_sorted)    
    
    # evaluate y_hat
    y_hat = events_sorted.mul(durations_sorted.add(1e-32).log().sub(imputed)).add(imputed)    
    
    # takes MAE form
    loss = y_hat.sub(theta_sorted).abs().div(n)    
    
    return loss


def dsaft_rmse_loss(theta, durations, events):
    '''
    theta: prediction output from DNN layers
    durations: log-scaled observed time (log(Y))
    events: right-censoring-indicator (delta)
    '''
    # compute residual e_i
    e = theta.sub(durations.view(-1, 1).add(1e-32).log()).neg()
    n = e.shape[0]
    
    # sort e_i w.r.t values and store sorted indices
    e_sorted = torch.sort(e.view(-1)).values
    e_indices = torch.sort(e.view(-1)).indices.type(torch.float32)
    
    # sort durations(log(Y)), events(delta), and theta(output from DNN) w.r.t. sorted indices of e_i
    tmp = torch.stack([e_indices, durations.view(-1), events.type(torch.float32).view(-1), theta.view(-1)], dim = 1)
    tmp = tmp[tmp[:, 0].sort()[1]]
    durations_sorted = tmp[:, 1]
    events_sorted = tmp[:, 2]
    theta_sorted = tmp[:, 3]
    
    # get risk set and removed (instances whose events had been occured)
    diff_ = e_sorted.view(-1, 1).sub(e_sorted).sub(1e-32)
    removed = diff_.abs().div(diff_).relu()
    at_risks = removed.neg().add(torch.ones(n, n))    

    # estimate survival function of e_i via KM estimator
    surv = events_sorted.div(at_risks.sum(1)).sub(1).neg().abs().mul(removed).add(at_risks).prod(dim = 1)    

    # estimate differential of F (cumulative density function) i.e. dF(u)
    h = torch.cat([surv.neg().add(1)[1:], torch.ones(1)]).sub(torch.cat([torch.zeros(1), surv.neg().add(1)[:-1]]))    

    # evaluate imputed y using conditional expected value of epsilon
    imputed = e_sorted.mul(h).mul(at_risks).sum(1).div(surv).add(theta_sorted)    
    
    # evaluate y_hat
    y_hat = events_sorted.mul(durations_sorted.add(1e-32).log().sub(imputed)).add(imputed)    
    
    # takes RMSE form
    loss = y_hat.sub(theta_sorted).pow(2).sum().div(n).pow(0.5)    
    
    return loss


## DSAFT - negative kernel-smoothed profile likelihood loss function
def dsaft_nkspl_loss(theta, durations, events,
                     an = 1.0,
                     sigma = 1.0):
    '''
    theta: prediction output from DNN layers
    durations: log-scaled observed time (log(Y))
    events: right-censoring-indicator (delta)
    
    an: bandwidth parameter a_n for kernel smoothing (default = 1.0)
    sigma: scale parameter of Gaussian kernel
    kernel: pre-defined kernel function for kernel smoothing of residual difference (K(.) in paper)
            (default: pdf of standard normal distribution)
    '''
    kernel = torch.distributions.normal.Normal(0, sigma)
    # compute residual e_i
    e = theta.sub(durations.view(-1, 1).add(1e-32).log()).neg()

    # sort e_i w.r.t values and store sorted indices
    e_sorted = torch.sort(e.view(-1)).values
    e_indices = torch.sort(e.view(-1)).indices.type(torch.float32)
    
    # sort durations(log(Y)), events(delta), and theta(output from DNN) w.r.t. sorted indices of e_i
    tmp = torch.stack([e_indices, durations.view(-1), events.type(torch.float32).view(-1), theta.view(-1)], dim = 1)
    tmp = tmp[tmp[:, 0].sort()[1]]    
    durations_sorted = tmp[:, 1]
    events_sorted = tmp[:, 2]
    theta_sorted = tmp[:, 3]

    # number of instances
    n = e_sorted.shape[0]
        
    # conditional expectation of exponential of residual
    cond_E = kernel.log_prob(e_sorted.view(-1, 1).sub(e_sorted).div(an)).exp().mul(events_sorted).div(n * an).add(1e-32).sum(dim = 1)
    
    # conditional survival probability of exponential of residual
    surv = kernel.cdf(e_sorted.view(-1, 1).sub(e_sorted).div(an)).div(n).sum(dim = 1)
    
    # loss = - delta * (cond_E - surv - e_sorted + theta_sorted)
    # loss = cond_E.log().sub(surv.log()).sub(e_sorted).add(theta_sorted).mul(events_sorted).div(n).sum().neg()
    loss = cond_E.log().sub(surv.log()).sub(theta_sorted).mul(events_sorted).div(n).sum().neg()
    
    return loss


## DSAFT - negative kernel-smoothed profile likelihood loss function
def dsaft_nkspl_loss_new(theta, durations, events,
                         an = 1.0,
                         sigma = 1.0):
    '''
    theta: prediction output from DNN layers
    durations: log-scaled observed time (log(Y))
    events: right-censoring-indicator (delta)
    
    an: bandwidth parameter a_n for kernel smoothing (default = 1.0)
    sigma: scale parameter of Gaussian kernel
    kernel: pre-defined kernel function for kernel smoothing of residual difference (K(.) in paper)
            (default: pdf of standard normal distribution)
    '''
    kernel = torch.distributions.normal.Normal(0, sigma)
    # compute residual e_i
    e = theta.sub(durations.view(-1, 1).add(1e-32).log()).neg()

    # sort e_i w.r.t values and store sorted indices
    e_sorted = torch.sort(e.view(-1)).values
    e_indices = torch.sort(e.view(-1)).indices.type(torch.float32)
    
    # sort durations(log(Y)), events(delta), and theta(output from DNN) w.r.t. sorted indices of e_i
    tmp = torch.stack([e_indices, durations.view(-1), events.type(torch.float32).view(-1), theta.view(-1)], dim = 1)
    tmp = tmp[tmp[:, 0].sort()[1]]    
    durations_sorted = tmp[:, 1]
    events_sorted = tmp[:, 2]
    theta_sorted = tmp[:, 3]

    # number of instances
    n = e_sorted.shape[0]
        
    # conditional expectation of exponential of residual
    cond_E = kernel.log_prob(e_sorted.view(-1, 1).sub(e_sorted).div(an)).exp().mul(events_sorted).div(n * an).add(1e-32).sum(dim = 1)
    
    # conditional survival probability of exponential of residual
    surv = kernel.cdf(e_sorted.view(-1, 1).sub(e_sorted).div(an)).div(n).sum(dim = 1)
    
    # loss = - delta * (cond_E - surv - e_sorted + theta_sorted)
    loss = cond_E.log().sub(surv.log()).sub(e_sorted).add(theta_sorted).mul(events_sorted).div(n).sum().neg()
    
    return loss



## Classes below
class DSAFTRankLoss(torch.nn.Module):
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return dsaft_rank_loss(log_h, durations, events)

class DSAFTMAELoss(torch.nn.Module):
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return dsaft_mae_loss(log_h, durations, events)

class DSAFTRMSELoss(torch.nn.Module):
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return dsaft_rmse_loss(log_h, durations, events)

class DSAFTNKSPLLoss(torch.nn.Module):
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return dsaft_nkspl_loss(log_h, durations, events)
 
class DSAFTNKSPLLossNew(torch.nn.Module):
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return dsaft_nkspl_loss_new(log_h, durations, events)
