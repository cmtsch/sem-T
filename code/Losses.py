import torch
import torch.nn as nn
import sys
import numpy as np
import MI_losses
import time


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-5
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def distance_vectors_pairwise(anchor, positive, negative = None):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)

    eps = 1e-8
    d_a_p = torch.sqrt(a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    if negative is not None:
        n_sq = torch.sum(negative * negative, dim=1)
        d_a_n = torch.sqrt(a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
        d_p_n = torch.sqrt(p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p

def loss_Rho(currSamples, currClasses, crossSamples, crossClasses, doCross ):
#def loss_Rho(anchor, positive, classes):

    Rs = []
    means = []
    allClasses = torch.unique(currClasses)

    for label in allClasses:
        sampleSum = torch.sum(currSamples[ currClasses == label, :], dim=0)
        N = (currClasses == label).sum()
        Rs.append( (torch.norm(sampleSum) / N))
        means.append( (sampleSum)/N)

    if doCross:
        cross_allClasses = torch.unique(currClasses)
        for label in cross_allClasses:
            means.append(torch.mean(crossSamples[crossClasses == label, :], dim=0))

    R_intra = sum(Rs) / len(Rs)
    R_inter = torch.norm(sum(means)) / len(means)
    Rho = R_inter / R_intra
    return Rho

def loss_OnlyPositive( currSamples, currClasses, samplesPerClass):
    assert currSamples.dim() == 2, "Input must be a 2D matrix."
    eps = 1e-8
    batch_size = currClasses.size()[0]

    class_mask = (currClasses[:, None] == currClasses[None, :]).float()
    dist_matrix = distance_matrix_vector(currSamples, currSamples) + eps
    dist_matrix_mask_neg= dist_matrix + (class_mask.type_as(dist_matrix) -1.0) * 42
    all_positives = torch.max(dist_matrix_mask_neg, 0)[0]
    num_classes = int(batch_size / samplesPerClass)
    class_positives = torch.zeros(num_classes).cuda()
    for i in range (num_classes):
        class_positives[i] = torch.max(all_positives[i*samplesPerClass : (i+1)* samplesPerClass])

    loss = torch.mean(class_positives)
    return loss

def loss_TripletHiB( currSamples, currClasses, crossSamples, crossClasses, samplesPerClass, do_cross_batch):
    assert currSamples.dim() == 2, "Input must be a 2D matrix."
    eps = 1e-8
    batch_size = currClasses.size()[0]

    class_mask = (currClasses[:, None] == currClasses[None, :]).float()

    dist_matrix = distance_matrix_vector(currSamples, currSamples) + eps

    dist_matrix_mask_pos= dist_matrix + class_mask.type_as(dist_matrix) * 42
    dist_matrix_mask_neg= dist_matrix + (class_mask.type_as(dist_matrix) -1.0) * 42

    #hardest positives
    all_positives = torch.max(dist_matrix_mask_neg, 1)[0]
    #hardest negatives
    all_negatives = torch.min(dist_matrix_mask_pos, 1)[0]
    cross_chosen = 0

    if do_cross_batch:
        x_dist_matrix = distance_matrix_vector(currSamples, crossSamples) +eps
        cross_negatives = torch.min(x_dist_matrix,1)[0]
        helper = torch.cat ((all_negatives.unsqueeze(1), cross_negatives.unsqueeze(1)),1)
        all_negatives, min_ind = torch.min(helper, 1)
        cross_chosen = torch.sum(min_ind).item()

    do_reduction = False

    if do_reduction:
        num_classes = int(batch_size / samplesPerClass)
        class_positives = torch.zeros(num_classes).cuda()
        class_negatives = torch.zeros(num_classes).cuda()
        step = int(samplesPerClass /2 )
        off = int(batch_size/2 )
        # maybe a shortcut for samplesPerClass = 2?
        for i in range (num_classes):
            class_positives[i] = torch.max(torch.max(all_positives[i*step : (i+1)* step],all_positives[off+ i*step : off+(i+1)* step]))
            class_negatives[i] = torch.min(torch.min(all_negatives[i*step : (i+1)* step], all_negatives[off+i*step: off+(i+1)*step]))

        margin = 1.0
        loss = torch.mean(torch.clamp(margin + class_positives - class_negatives, min=0.0))
    else:
        margin = 1.0
        loss = torch.mean(torch.clamp(margin + all_positives - all_negatives, min=0.0))

    return loss, cross_chosen

def loss_ArcLength(currSamples, currClasses, crossSamples, crossClasses, doCross):

    #Calculate all class means
    means = torch.empty([currClasses.size(), 128])
    allClasses = torch.unique(currClasses)
    idx = 0

    for label in allClasses:
        means[idx]= torch.mean(currSamples[ currClasses == label, :], dim=0)

    if doCross:
        cross_allClasses = torch.unique(currClasses)
        cross_means = torch.empty([crossClasses.size(), 128])
        idx = 0
        for label in cross_allClasses:
            cross_means[idx] = torch.mean(crossSamples[crossClasses == label, :], dim=0)

        torch.cat( (means,cross_means), 0)

    inner_prod = means @ means.t()
    eps = 1e-8
    arc_lenghts = torch.pow(torch.arccos(inner_prod+eps), -1)

    # somehow normalize by multiplying with number of clusters
    loss = torch.mean(arc_lenghts) * arc_lenghts.size()[0]
    return loss

def loss_MaxMean(currSamples, currClasses, crossSamples, crossClasses, doCross):

    #Calculate all class means
    means = torch.empty([currClasses.size(), 128])
    allClasses = torch.unique(currClasses)
    idx = 0

    for label in allClasses:
        means[idx]= torch.mean(currSamples[ currClasses == label, :], dim=0)

    if doCross:
        cross_allClasses = torch.unique(currClasses)
        cross_means = torch.empty([crossClasses.size(), 128])
        idx = 0
        for label in cross_allClasses:
            cross_means[idx] = torch.mean(crossSamples[crossClasses == label, :], dim=0)

        torch.cat( (means,cross_means), 0)

    eps = 1e-8
    dist_matrix = distance_matrix_vector(means, means) + eps
    loss = -1 * torch.mean(dist_matrix)
    return loss



def chrisTest():
    anchors = torch.randn(5,3)
    positives = torch.randn(5,3)
    print(loss_SOS(anchors, positives))


def partition_assign(a, n):
    idx = np.argpartition(a,-n,axis=1)[:,:n]
    out = np.zeros(a.shape, dtype=int)
    #idx = idx.cpu().numpy()
    np.put_along_axis(out,idx,1,axis=1)
    return out


def loss_SOS_new (currSamples, currClasses, crossSamples, crossClasses, doCross, use_KnearestNeighbors = True, k = 5):
    # the anchors and the positives should have a similar distance
    eps = 1e-8

    # Idea in the spirit of SOSNet
    # For each combination of classes (i,j) we calculate the variance of the distances between descriptors of i and j


    dist_matrix = distance_matrix_vector(currSamples, currSamples) + eps
    allClasses = torch.unique(currClasses)
    numClasses = allClasses.size()

    means = torch.ones( [numClasses, numClasses]) * 42
    variances = torch.ones( [numClasses, numClasses])
    idx_i= 0
    idx_j= 0

    for label_i in allClasses:
        for label_j in allClasses:
            if (label_i != label_j):
                mask = torch.ger((currClasses == label_i).float(), (currClasses == label_j).float()).bool()
                variances [idx_i, idx_j], means [idx_i, idx_j]  = torch.var_mean(dist_matrix[mask])
            idx_j+=1

        idx_i+=1

    if doCross:
        cross_dist_matrix = distance_matrix_vector(currSamples, crossSamples) + eps
        cross_allClasses = torch.unique(crossClasses)
        cross_numClasses = cross_allClasses.size()
        cross_means = torch.ones([numClasses, cross_numClasses]) * 42
        cross_variances = torch.ones([numClasses, cross_numClasses])
        idx_i = 0
        idx_j = 0
        for label_i in allClasses:
            for label_j in cross_allClasses:
                if (label_i != label_j):
                    mask = torch.ger((currClasses == label_i).float(), (crossClasses == label_j).float()).bool()
                    cross_variances [idx_i, idx_j], cross_means [idx_i, idx_j] = torch.var_mean(cross_dist_matrix[mask])
                idx_j+=1
            idx_i+=1

        means = torch.cat((means, cross_means), 1)
        variances = torch.cat((variances, cross_variances), 1)

    vals, inds = torch.topk(means, k, dim=1)
    loss = torch.mean(torch.gather(variances,1,inds))

    return loss
    
def loss_MI (currSamples, currClasses, crossSamples, crossClasses, MI_type, do_cross):
    
    if (MI_type != 'infoNCE'):
        return MI_losses.fenchel_dual_loss (currSamples, currClasses, crossSamples, MI_type, do_cross)
    else: # infoNCE
        return MI_losses.infonce_loss(currSamples,currClasses, crossSamples, do_cross)

class CorrelationPenaltyLoss(nn.Module):
    def __init__(self):
        super(CorrelationPenaltyLoss, self).__init__()

    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        zeroed = input - mean1.expand_as(input)
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        d = torch.diag(torch.diag(cor_mat))
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag
        return torch.sqrt(d_sq.sum())/input.size(0)

def global_orthogonal_regularization(anchor, negative):

    neg_dis = torch.sum(torch.mul(anchor,negative),1)
    dim = anchor.size(1)
    gor = torch.pow(torch.mean(neg_dis),2) + torch.clamp(torch.mean(torch.pow(neg_dis,2))-1.0/dim, min=0.0)
    
    return gor


def loss_SOS(anchors, positives, use_KnearestNeighbors=True, k=2):
    # the anchors and the positives should have a similar distance
    eps = 1e-8

    dist_matrix_anchors = distance_matrix_vector(anchor, anchor)
    dist_matrix_positives = distance_matrix_vector(positive, positive)

    # Construct two masks: which correspond to the k-nearest neightbors: mask_anchor/positive
    # The total mask is then mask_total = mask_anchor v mask_positive

    mask_anchor = partition_assign(dist_matrix_anchors.detach().cpu().numpy(), k)
    mask_positive = partition_assign(dist_matrix_positives.detach().cpu().numpy(), k)

    # logical_or function not in torch 1.4.0
    # mask_total = torch.logical_or(mask_anchor, mask_positive)
    # dummy or function
    mask_total = mask_anchor + mask_positive
    mask_total = (mask_total >= 1).astype(int)
    mask_total = torch.from_numpy(mask_total)

    mask_total = mask_total.cuda()

    helper = dist_matrix_anchors * mask_total - dist_matrix_positives * mask_total

    loss = torch.norm(helper, dim=1)
    loss = torch.mean(loss)
    return loss

def loss_HardNet(anchor, positive, classes, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # positive distances
    pos1 = torch.diag(dist_matrix)

    ## New: we need to filter out the ones that belong to the same class
    p_mask = (classes[:, None] == classes[None, :]).float()
    p_mask = p_mask.type_as(dist_matrix) * 42
    dist_without_min_on_diag = dist_matrix+p_mask

    # we don't wanna consider entries <= 0.008, maybe because those might be accidental positives?
    mystery_mask = dist_without_min_on_diag.le(0.008).float()
    mystery_mask = mystery_mask.type_as(dist_without_min_on_diag)*42
    dist_without_min_on_diag = dist_without_min_on_diag+mystery_mask


    if batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg
        pos = pos1
    else:
        print ('Unknown batch reduce mode. Average or random no longer supported. Use min ')
        sys.exit(1)

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == "triplet_quadratic":
        lin_loss = torch.clamp(margin + pos - min_neg, min=0.0)
        loss = torch.square(lin_loss)
    else:
        print ('Unknown loss type. Try triplet_margin or triplet_quadratic. Softmax and contrastive no longer supported')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

