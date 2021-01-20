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

def loss_Rho(anchor, positive, classes):

    Rs = []
    means = []

    start1 = time.time()
    allClasses = torch.unique(classes)

    # Tech consulting IBM, Accenture,
    if ( classes.size()[0] == allClasses.size()[0] ):
        #short cut
        R_intra = torch.mean( torch.norm(anchor + positive , dim=1) / 2. )
        R_inter = torch.mean( torch.norm( (anchor + positive)/2. , dim=1) )
        Rho = R_inter / R_intra
        return  Rho

    for label in allClasses:
        start1 = time.time()
        help1 = torch.sum(anchor[ classes== label, :], dim=0)
        help2 = torch.sum(positive[ classes== label, :], dim=0)
        N = (classes == label).sum() * 2
        Rs.append( (torch.norm(help1 + help2) / N))
        means.append( (help1 + help2)/N)

    R_intra = sum(Rs) / len(Rs)
    R_inter = torch.norm(sum(means)) / len(means)
    Rho = R_inter / R_intra
    return Rho

def loss_HardNet(anchor, positive, classes, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()


    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)

    ## New: we need to filter out the ones that belong to the same class
    p_mask = (classes[:, None] == classes[None, :]).float()
    #print ("Numer of same classes is ")
    #print (torch.sum(p_mask))
    p_mask = p_mask.type_as(dist_matrix) * 42
    dist_without_min_on_diag = dist_matrix+p_mask

    mystery_mask = dist_without_min_on_diag.le(0.008).float()
    mystery_mask = mystery_mask.type_as(dist_without_min_on_diag)*42
    dist_without_min_on_diag = dist_without_min_on_diag+mystery_mask


    ##dist_without_min_on_diag = dist_matrix+eye*10
    ##mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    ## mask marks elements which are smaller than 0.008
    ## we assume those are the exact same patch and should not be considered
    ##mask = mask.type_as(dist_without_min_on_diag)*10
    ## now we use the mask to artificially increase the distance of those
    ##dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
            min_neg = torch.min(min_neg,min_neg2)
        if False:
            dist_matrix_a = distance_matrix_vector(anchor, anchor)+ eps
            dist_matrix_p = distance_matrix_vector(positive,positive)+eps
            dist_without_min_on_diag_a = dist_matrix_a+eye*10
            dist_without_min_on_diag_p = dist_matrix_p+eye*10
            min_neg_a = torch.min(dist_without_min_on_diag_a,1)[0]
            min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p,0)[0])
            min_neg_3 = torch.min(min_neg_p,min_neg_a)
            min_neg = torch.min(min_neg,min_neg_3)
            print (min_neg_a)
            print (min_neg_p)
            print (min_neg_3)
            print (min_neg)
        min_neg = min_neg
        pos = pos1
    #elif batch_reduce == 'average':
    #    pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
    #    min_neg = dist_without_min_on_diag.view(-1,1)
    #    if anchor_swap:
    #        min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
    #        min_neg = torch.min(min_neg,min_neg2)
    #    min_neg = min_neg.squeeze(0)
    #elif batch_reduce == 'random':
    #    idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
    #    min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
    #    if anchor_swap:
    #        min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
    #        min_neg = torch.min(min_neg,min_neg2)
    #    min_neg = torch.t(min_neg).squeeze(0)
    #    pos = pos1
    else: 
        print ('Unknown batch reduce mode. Average or random no longer supported. Use min ')
        sys.exit(1)

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == "triplet_quadratic":
        lin_loss = torch.clamp(margin + pos - min_neg, min=0.0)
        loss = torch.square(lin_loss)
    #elif loss_type == 'softmax':
    #    exp_pos = torch.exp(2.0 - pos);
    #    exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
    #    loss = - torch.log( exp_pos / exp_den )
    #elif loss_type == 'contrastive':
    #    loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin or triplet_quadratic. Softmax and contrastive no longer supported')
        sys.exit(1)
    loss = torch.mean(loss)
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


def loss_SOS (anchor, positive, use_KnearestNeighbors = True, k = 2):
    # the anchors and the positives should have a similar distance

    Nbatch, dimensions = anchor.size()
    #print ("Dimensions of desriptor is " + str(dimensions))
    #print ("Number in batch is " + str(Nbatch))

    # first we need to get a NxN (batch size) distance matrix for the anchors and the positives
    # dist_anchors and dist_positives
    # Calculate the L2 norm
    dist_matrix_anchors = distance_matrix_vector(anchor, anchor)
    dist_matrix_positives = distance_matrix_vector(positive, positive)

    # Construct two masks: which correspond to the k-nearest neightbors: mask_anchor/positive
    # The total mask is then mask_total = mask_anchor v mask_positive
  
    
    mask_anchor = partition_assign (dist_matrix_anchors.detach().cpu().numpy(), k)
    mask_positive = partition_assign (dist_matrix_positives.detach().cpu().numpy(), k)

    #logical_or function not in torch 1.4.0
    #mask_total = torch.logical_or(mask_anchor, mask_positive)
    #dummy or function
    mask_total = mask_anchor + mask_positive
    mask_total = (mask_total >= 1).astype(int) 
    mask_total = torch.from_numpy(mask_total)

    mask_total = mask_total.cuda()

    helper = dist_matrix_anchors * mask_total - dist_matrix_positives * mask_total

    loss = torch.norm(helper,  dim = 1)
    loss = torch.mean(loss)
    return loss
    
def loss_MI (anchor, positive, classes, MI_type):
    
    #print ("Called loss_MI with type " + MI_type)

    Nbatch, dimensions = anchor.size()
    
    x=torch.cat([anchor, positive], dim=0)
    
    labels = classes 
    labels = torch.cat([labels, labels], dim=0)
    labels = labels.cuda()


    if (MI_type != 'infoNCE'):
        return MI_losses.fenchel_dual_loss (x, labels, MI_type)
    else: # infoNCE
        return MI_losses.infonce_loss(x,labels)

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

