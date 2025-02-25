#!/usr/bin/python2 -utt
#-*- coding: utf-8 -*-
"""
This is HardNet local patch descriptor. The training code is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.

If you use this code, please cite 
@article{HardNet2017,
 author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
     year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin 
"""

from __future__ import division, print_function
import sys
from copy import deepcopy
import math
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
import PIL
from EvalMetrics import ErrorRateAt95Recall
from Losses import loss_HardNet, global_orthogonal_regularization, loss_SOS, loss_MI, CorrelationPenaltyLoss
from W1BS import w1bs_extract_descs_and_save
from Utils import L2Norm, cv2_scale, np_reshape
from Utils import str2bool
import torch.nn as nn
import torch.nn.functional as F
from Datastuff import TripletPhotoTour, create_loaders
from HardNetExtendedModel import HardNetExtended
from torch.utils.tensorboard import SummaryWriter


# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options

parser.add_argument('--w1bsroot', type=str,
                    default='data/sets/wxbs-descriptors-benchmark/code/',
                    help='path to dataset')
parser.add_argument('--dataroot', type=str,
                    default='data/sets/',
                    help='path to dataset')
parser.add_argument('--enable-logging',type=str2bool, default=False,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='data/logs/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='data/models/',
                    help='folder to output model checkpoints')
parser.add_argument('--experiment-name', default= 'liberty_train/',
                    help='experiment path')
parser.add_argument('--training-set', default= 'liberty',
                    help='Other options: notredame, yosemite')
parser.add_argument('--loss', default= 'triplet_margin',
                    help='Other options: softmax, contrastive, triplet_quadratic')
parser.add_argument('--batch-reduce', default= 'min',
                    help='Other options: average, random, random_global, L2Net')
parser.add_argument('--num-workers', default= 0, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--decor',type=str2bool, default = False,
                    help='L2Net decorrelation penalty')
parser.add_argument('--anchorave', type=str2bool, default=False,
                    help='anchorave')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=2, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--anchorswap', type=str2bool, default=True,
                    help='turns on anchor swap')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--gor',type=str2bool, default=False,
                    help='use gor')
parser.add_argument('--freq', type=float, default=10.0,
                    help='frequency for cyclic learning rate')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                    help='gor parameter')
parser.add_argument('--lr', type=float, default=10.0, metavar='LR',
                    help='learning rate (default: 10.0. Yes, ten is not typo)')
parser.add_argument('--fliprot', type=str2bool, default=True,
                    help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--augmentation', type=str2bool, default=False,
                    help='turns on shift and small scale rotation augmentation')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
                    
## I moved this distinction to the --LossType parameter
#parser.add_argument('--tripletLoss', default= 'linear',
#                    help='Other options: quadratic (SOSNet)')
parser.add_argument('--lossSOS', type=str2bool, default=False, metavar='SOS',
                    help='use the Second Order Similarity loss')
parser.add_argument('--MiLoss', default= 'JSD',
                    help='Other options: JSD, DV, infoNCE')
parser.add_argument('--uniquePairs', type=str2bool, default=True,
                    help='in one training batch there is at most one pair corresponding to the same real world oject')

parser.add_argument('--skipInit', type=str2bool, default=False,
                    help='Skip Initialization of weights, for quick debug runs')


args = parser.parse_args()

#suffix = '{}_{}_{}'.format(args.experiment_name, args.training_set, args.batch_reduce)
suffix = args.experiment_name

if args.uniquePairs:
    suffix = suffix + '_A_'
else:
    suffix = suffix + '_B_'

suffix = suffix + args.MiLoss



#if args.gor:
#    suffix = suffix+'_gor_alpha{:1.1f}'.format(args.alpha)
#if args.anchorswap:
#    suffix = suffix + '_as'
#if args.anchorave:
#    suffix = suffix + '_av'
#if args.fliprot:
#        suffix = suffix + '_fliprot'

triplet_flag = (args.batch_reduce == 'random_global') or args.gor

dataset_names = ['liberty', 'notredame', 'yosemite']

TEST_ON_W1BS = False
# check if path to w1bs dataset testing module exists
if os.path.isdir(args.w1bsroot):
    sys.path.insert(0, args.w1bsroot)
    import utils.w1bs as w1bs
    TEST_ON_W1BS = True

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()

print (("NOT " if not args.cuda else "") + "Using cuda")

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# set random seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# TensorBoard summary writer
writer = SummaryWriter(comment=suffix)

def train(train_loader, model, optimizer, epoch, logger, load_triplets  = False):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:

        data_a, data_p, classes = data
        if args.cuda:
            data_a, data_p, classes  = data_a.cuda(), data_p.cuda(), classes.cuda()
            data_a, data_p = Variable(data_a), Variable(data_p)
            out_a, mi_out_a = model(data_a)
            out_p , mi_out_p =  model(data_p)

        # Here hardest-in-batch aka HardNet
        loss = loss_HardNet(out_a, out_p, classes,
                        margin=args.margin,
                        anchor_swap=args.anchorswap,
                        anchor_ave=args.anchorave,
                        batch_reduce = args.batch_reduce,
                        loss_type = args.loss)
        
        if args.lossSOS:
            # do we need the classes arguemnt here
            loss += loss_SOS(out_a, out_p)

        if args.MiLoss != 'none':
            loss += loss_MI (mi_out_a, mi_out_p,classes, args.MiLoss)
        
        if args.decor:
            loss += CorrelationPenaltyLoss()(out_a)
            
        if args.gor:
            loss += args.alpha*global_orthogonal_regularization(out_a, out_n)

            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('Loss/train', loss.item(), batch_idx + epoch*len(train_loader))
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.item()))
        

    if (args.enable_logging):
        logger.log_value('loss', loss.item()).step()

    try:
        os.stat('{}{}'.format(args.model_dir,suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir,suffix))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,epoch))
    return loss.item()

def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a.requires_grad_(False)
        data_p.requires_grad_(False)

        out_a = model(data_a)[0]
        out_p = model(data_p)[0]
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy().reshape(-1,1))
        ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)

        if batch_idx % args.log_interval == 0:
            pbar.set_description(logger_test_name+' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    writer.add_scalar('Evaluation/'+logger_test_name, fpr95, epoch)

    if (args.enable_logging):
        logger.log_value(logger_test_name+' fpr95', fpr95)
    return

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
        1.0 - float(group['step']) * float(args.batch_size) / (args.n_triplets * float(args.epochs)))
    return

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, test_loaders, model, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    #if (args.enable_logging):
    #    file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()


    #optimizer1 = create_optimizer(model.features, args.lr)
    optimizer1 = create_optimizer(model, args.lr)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            
    
    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):

        # iterate over test loaders and test results
        currLoss = train(train_loader, model, optimizer1, epoch, logger, triplet_flag)

        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])
        
        if TEST_ON_W1BS :
            # print(weights_path)
            patch_images = w1bs.get_list_of_patch_images(
                DATASET_DIR=args.w1bsroot.replace('/code', '/data/W1BS'))
            desc_name = 'curr_desc'# + str(random.randint(0,100))
            
            DESCS_DIR = LOG_DIR + '/temp_descs/' #args.w1bsroot.replace('/code', "/data/out_descriptors")
            OUT_DIR = DESCS_DIR.replace('/temp_descs/', "/out_graphs/")

            for img_fname in patch_images:
                w1bs_extract_descs_and_save(img_fname, model, desc_name, cuda = args.cuda,
                                            mean_img=args.mean_image,
                                            std_img=args.std_image, out_dir = DESCS_DIR)


            force_rewrite_list = [desc_name]
            w1bs.match_descriptors_and_save_results(DESC_DIR=DESCS_DIR, do_rewrite=True,
                                                    dist_dict={},
                                                    force_rewrite_list=force_rewrite_list)
            if(args.enable_logging):
                w1bs.draw_and_save_plots_with_loggers(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name],
                                         logger=file_logger,
                                         tensor_logger = logger)
            else:
                w1bs.draw_and_save_plots(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name])
        #randomize train loader batches
        train_loader, test_loaders2 = create_loaders(args, dataset_names)


if __name__ == '__main__':
    print ("Start of main")
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = os.path.join(args.log_dir, suffix)
    DESCS_DIR = os.path.join(LOG_DIR, 'temp_descs')
    if TEST_ON_W1BS:
        if not os.path.isdir(DESCS_DIR):
            os.makedirs(DESCS_DIR)
    logger, file_logger = None, None
    model = HardNetExtended(args.skipInit)
    if(args.enable_logging):
        from Loggers import Logger, FileLogger
        logger = Logger(LOG_DIR)
        #file_logger = FileLogger(./log/+suffix)
    train_loader, test_loaders = create_loaders(args, dataset_names)
    main(train_loader, test_loaders, model, logger, file_logger)
