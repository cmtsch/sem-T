"""
Information theory and statistical losses for deep networks.
"""

import math
import torch
import torch.nn.functional as F
import MI_losses


def mi_loss(x_joint, x_marginal, y, mi_net, unbiased=False):
    """ Compute the Mutual Information Neural Estimator between x and y.
    Args:
        (x_joint, y): sample of the joint distribution of x and y.
        x_marginal: sample of the marginal distribution of x.
        mi_net: the MINE network.
        unbiased: True to use an unbiased gradient estimator.
    Returns:
        A lower bound on the true mutual information, averaged over the batch.
    """
    t = mi_net(x_joint, y).mean()
    et = torch.exp(mi_net(x_marginal, y)).mean()
    if unbiased:
        if mi_net.ma_et is None:
            mi_net.ma_et = et.detach().item()
        mi_net.ma_et += mi_net.ma_rate * (et.detach().item() - mi_net.ma_et)
        mi = (t - torch.log(torch.clamp(et, 0) + 1e-8)
              * et.detach() / (mi_net.ma_et + 1e-8))
    else:
        mi = t - torch.log(torch.clamp(et, 0) + 1e-8)
    return mi


def fenchel_dual_loss(x, classes, measure=None):
    '''Computes the f-divergence distance between positive and negative
    joint distributions defined through classes.
    Divergences supported are Jensen-Shannon `JSD` and Donsker-Varadhan `DV`.
    Args:
        x: samples of size (n_samples, dim).
        classes: class label for each sample in x (size n_labels).
        measure: f-divergence measure.
    Returns:
        torch.Tensor: Loss.
    '''
    supported_measures = ['DV', 'JSD']

    #print ("Fenchel dual loss called")

    def get_positive_expectation(p_samples, measure):
        log_2 = math.log(2.)
        if measure == 'JSD':
            Ep = log_2 - F.softplus(-p_samples)  # JSD is shifted
        elif measure == 'DV':
            Ep = p_samples
        else:
            raise NotImplementedError(
                'Measure `{}` not supported. Supported: {}'.format(
                    measure, supported_measures))
        return Ep

    def get_negative_expectation(n_samples, measure):
        log_2 = math.log(2.)
        if measure == 'JSD':
            En = F.softplus(-n_samples) + n_samples - log_2  # JSD is shifted
        elif measure == 'DV':
            sample_max = torch.max(n_samples)
            En = torch.log(torch.exp(n_samples - sample_max).mean()
                           + 1e-8) + sample_max
        else:
            raise NotImplementedError(
                'Measure `{}` not supported. Supported: {}'.format(
                    measure, supported_measures))
        return En

    # 2048, 128
    n_samples, dim = x.size()

    # Similarity matrix mixing positive and negative samples
    # 2048x2048 matrix
    sim = x @ x.t()

    # Compute the positive and negative score
    E_pos = get_positive_expectation(sim, measure)
    E_neg = get_negative_expectation(sim, measure)
    # Each is a 2048x2048 matrix

    # Mask positive and negative terms for positive and negative parts of loss
    p_mask = (classes[:, None] == classes[None, :]).float()
    # 2048 x 2048 matrix, is 1 if two samples belong to the same class, 1 on diagonal, symmetric
    n_mask = 1 - p_mask

    # added by me: set diagonal to zero
    p_mask = p_mask -  torch.diag(torch.ones(n_samples)).cuda()

    # 2048 x 2048 matrix, is 1 if two samples do NOT belong to the same class, 0 on diagonal, symmetric
    E_pos = (E_pos * p_mask).sum() / p_mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()

    return E_neg - E_pos


def infonce_loss(x, classes):
    '''Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.
    Args:
        x: samples of size (n_samples, dim).
        classes: class label for each sample in x (size n_labels).
    Returns:
        torch.Tensor: Loss.
    '''
    n_samples, dim = x.size()
    print ("Size of x is " + str(list(x.size())))    
    #print ("InfoNCE loss called")

    # Compute similarities and mask them for positive and negative ones
    sim = x @ x.t()

    # Positive similarities
    p_mask = classes[:, None] == classes[None, :]
    # Probably a matrix of size 2048x 2048

    #print ("Size of p_mask is " + str(list(p_mask.size())))
    sim_p = sim[p_mask].clone().unsqueeze(1)
    print ("Size of sim_p is " + str(list(sim_p.size())))
    # Flat tensor of size dim_sem0² + dim_sem1² + ...

    # All positive entries tensor of dimension e.g. 163840 x 1
    # where 128 * 128 * 10 = 163840
    # maybe it is 256 * 256 * 10 = 655360

    # Negative similarities
    sim[p_mask] -= 10.  # mask out the positive samples
    helper= classes.unsqueeze(1).repeat(1, n_samples)
    #print ("Size of helper is " + str(list(helper.size())))

    #sim_n = sim[classes.unsqueeze(1).repeat(1, n_samples)[p_mask]]
    sim_n = sim[torch.arange(0,n_samples).unsqueeze(1).repeat(1, n_samples)[p_mask]]
    # tensor of dimension 163840 x 1920
    # where 1920 = 2048 - 128

    print ("Size of sim_n is " + str(list(sim_n.size())))

    # The positive score is the first element of the log softmax.
    pred_lgt = torch.cat([sim_p, sim_n], dim=1)
    # pred_lgt will be of size 163840 x 1921
    #print ("Size of pred_lgt is " + str(list(pred_lgt.size())))

    #Take the log_softmax of each row
    # Calculates a value for each column in that row, but we are only interested in the first value
    pred_log = F.log_softmax(pred_lgt, dim=1)
    #print ("Size of pred_log is " + str(list(pred_log.size())))

    # take the mean of the first column
    loss = -pred_log[:, 0].mean()
    return loss
