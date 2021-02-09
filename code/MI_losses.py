"""
Information theory and statistical losses for deep networks.
"""

import math
import torch
import torch.nn.functional as F
import MI_losses

def fenchel_dual_loss(currSamples, currClasses,crossSamples, measure, do_cross):
    '''Computes the f-divergence distance between positive and negative
    joint distributions defined through classes.
    Divergences supported are Jensen-Shannon `JSD` and Donsker-Varadhan `DV`.
    Args:
        currSamples: samples of size (n_samples, dim).
        currClasses: class label for each sample in x (size n_labels).
        measure: f-divergence measure.
    Returns:
        torch.Tensor: Loss.
    '''
    def get_positive_expectation(p_samples, measure):
        log_2 = math.log(2.)
        if measure == 'JSD':
            Ep = log_2 - F.softplus(-p_samples)  # JSD is shifted
        elif measure == 'DV':
            Ep = p_samples

        return Ep

    def get_negative_expectation(n_samples, measure):
        log_2 = math.log(2.)
        if measure == 'JSD':
            En = F.softplus(-n_samples) + n_samples - log_2  # JSD is shifted
        elif measure == 'DV':
            sample_max = torch.max(n_samples)
            En = torch.log(torch.exp(n_samples - sample_max).mean()
                           + 1e-8) + sample_max

        return En

    supported_measures = ['DV', 'JSD']
    if measure not in supported_measures:
        raise NotImplementedError(
            'Measure `{}` not supported. Supported: {}'.format(
                measure, supported_measures))

    n_samples, dim = currSamples.size()

    # Similarity matrix mixing positive and negative samples
    sim = currSamples @ currSamples.t()


    # Compute the positive and negative score
    E_pos = get_positive_expectation(sim, measure)
    E_neg = get_negative_expectation(sim, measure)

    p_mask = (currClasses[:, None] == currClasses[None, :]).float()
    n_mask = 1 - p_mask

    # added by me: set diagonal to zero
    p_mask = p_mask -  torch.diag(torch.ones(n_samples)).cuda()

    # 2048 x 2048 matrix, is 1 if two samples do NOT belong to the same class, 0 on diagonal, symmetric
    E_pos = (E_pos * p_mask).sum() / p_mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()

    if do_cross:
        cross_sim = currSamples @ crossSamples.t()
        E_neg_cross = get_negative_expectation(cross_sim, measure)
        E_neg += (E_neg_cross).mean()

    return E_neg - E_pos


def infonce_loss(currSamples, currClasses, crossSamples, do_cross):
    '''Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.
    Args:
        currSamples: samples of size (n_samples, dim).
        currClasses: class label for each sample in x (size n_labels).
    Returns:
        torch.Tensor: Loss.
    '''
    n_samples, dim = currSamples.size()

    # Compute similarities and mask them for positive and negative ones
    sim = currSamples @ currSamples.t()

    # Positive similarities
    p_mask = currClasses[:, None] == currClasses[None, :]
    n_mask = ~p_mask
    # set diagonal to zero, since these are literally the same patches
    diag_mask = torch.diag(torch.ones(p_mask.size()[0])).cuda()
    p_mask = (p_mask.float() - diag_mask.float()).bool()

    sim_p = sim[p_mask].clone().unsqueeze(1)
    #print ("Size of sim_p is " + str(list(sim_p.size())))
    # Flat tensor of size numClasses * (samplesPerClass^2 - samplesPerClass)
    # e.g. (1024 * (4-2)) = 2048 or (128 * (256-16)) = 30720

    # Negative similarities
    sim[p_mask] -= 10.  # mask out the positive samples by making them neglegible in the softmax
    sim_n = sim[torch.arange(0,n_samples).unsqueeze(1).repeat(1, n_samples)[p_mask]]
    #print("Size of sim_n is " + str(list(sim_n.size())))
    pred_lgt = torch.cat([sim_p, sim_n], dim=1)

    if do_cross:
        cross_sim = currSamples @ crossSamples.t()
        cross_sim_n = cross_sim[torch.arange(0,n_samples).unsqueeze(1).repeat(1, n_samples)[p_mask]]
        #print("Size of cross_sim_n is " + str(list(cross_sim_n.size())))
        pred_lgt = torch.cat([pred_lgt, cross_sim_n], dim=1)

    #Take the log_softmax of each row
    # Calculates a value for each column in that row, but we are only interested in the first value
    pred_log = F.log_softmax(pred_lgt, dim=1)

    # take the mean of the first column
    loss = -pred_log[:, 0].mean()
    return loss
