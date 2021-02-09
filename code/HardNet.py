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
import argparse
import torch
import torch.nn.init
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import time
from EvalMetrics import ErrorRateAt95Recall
from Losses import loss_TripletHiB, loss_OnlyPositive, loss_SOS_new, loss_MI, loss_Rho, loss_ArcLength, loss_MaxMean
from Utils import str2bool
from Datastuff import TripletPhotoTour, create_loaders
from HardNetModel import HardNet
from HardNetExtendedModel import HardNetExtended
from torch.utils.tensorboard import SummaryWriter


# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options
parser.add_argument('--dataroot', type=str,
                    default='data/sets/',
                    help='path to dataset')
parser.add_argument('--enable-logging',type=str2bool, default=False,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='data/logs/',
                    help='folder to output log')
#parser.add_argument('--model-dir', default='data/models/',
#                    help='folder to output model checkpoints')
parser.add_argument('--experiment-name', default= 'liberty_train/',
                    help='experiment path')
parser.add_argument('--training-set', default= 'liberty',
                    help='Other options: notredame, yosemite')
parser.add_argument('--batch-reduce', default= 'min',
                    help='Other options: average, random, random_global, L2Net')
parser.add_argument('--num-workers', default= 0, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
#parser.add_argument('--decor',type=str2bool, default = False,
#                    help='L2Net decorrelation penalty')
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
parser.add_argument('--batch-size', type=int, default=2048, metavar='BS',
                    help='input batch size for training, not in pairs but total samples (default: 2048)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
#parser.add_argument('--gor',type=str2bool, default=False,
#                    help='use gor')
parser.add_argument('--freq', type=float, default=10.0,
                    help='frequency for cyclic learning rate')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                    help='gor parameter')
parser.add_argument('--lr', type=float, default=10.0, metavar='LR',
                    help='learning rate (default: 10.0. Yes, ten is not typo)')
parser.add_argument('--fliprot', type=str2bool, default=False,
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

parser.add_argument('--PrimLoss', default= 'Triplet',
                    help='Other options: Triplet, Positive, None')
parser.add_argument('--loss', default= 'triplet_margin',
                    help='Other options: triplet_margin, triplet_quadratic')
parser.add_argument('--MiLoss', default= 'None',
                    help='Other options: JSD, DV, infoNCE')
parser.add_argument('--HyperLoss', default= 'None',
                    help='Other options: Rho, ArcLength, MaxMean')
parser.add_argument('--lossSOS', type=str2bool, default=False, metavar='SOS',
                    help='use the Second Order Similarity loss')

parser.add_argument('--adjLR', type=str2bool, default=True,
                    help='Adjust the learning rate')
parser.add_argument('--samplesPerClass', type=int, default=2,
                    help='Other options: 16, 64')
parser.add_argument('--cross_batch', type=int, default=0,
                    help='Other options: 3')
parser.add_argument('--highDim', type=str2bool, default=False,
                    help='indicates wether we want to use a high-dimensional embedding')
parser.add_argument('--myAugmentation', default= 'N',
                    help='Other options: O,R,N')

parser.add_argument('--skipInit', type=str2bool, default=False,
                    help='Skip Initialization of weights, for quick debug runs')
args = parser.parse_args()

#suffix = '{}_{}_{}'.format(args.experiment_name, args.training_set, args.batch_reduce)
suffix = args.experiment_name
suffix = suffix + '_'

# Sampline A is the original sampling, B is the new one which allows more than on pair per class
if args.samplesPerClass not in [2, 16, 64]:
    raise NotImplementedError('Unknown argument of samplesPerClass argument')
if args.samplesPerClass == 2:
    suffix = suffix + 'A2'
else:
    suffix = suffix + 'B' + str(args.samplesPerClass)

#this variable takes on values O (old), R (Rémi), N (none)
if (args.myAugmentation == 'O'):
    args.augmentation = True
else:
    args.augmentation = False

suffix = suffix + args.myAugmentation

# indicates wether we want to use a high-dimensional embedding
if args.highDim:
    suffix = suffix + 'H'
else:
    suffix = suffix + 'L'
# X-batch (cross-batch memory information)
suffix = suffix + 'X' + str(args.cross_batch)

suffix = suffix + '_'
suffix = suffix + args.MiLoss

myModelDir = 'data/models/'+args.experiment_name+'/'

triplet_flag = (args.batch_reduce == 'random_global')

dataset_names = ['liberty', 'notredame', 'yosemite']

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

    num_cross_batches = args.cross_batch

    if epoch <3:
        num_cross_batches = 0

    cross_flag = True
    if num_cross_batches == 0:
        cross_flag = False
        crossSamples = torch.tensor( 0.0)
        crossClasses = torch.tensor( 0.0)
    else:
        crossSamples = torch.empty([args.batch_size * num_cross_batches, 128]).cuda()
        crossClasses = torch.empty([args.batch_size * num_cross_batches],dtype=torch.int).cuda()

    cross_initialized = False
    cross_ptr = 0

    for batch_idx, data in pbar:

        data_a, data_p, classes, _ = data
        data_a = data_a[0]
        data_p = data_p[0]
        classes = classes[0]
        batchSize = data_a.size()[0] *2

        #Push data through the network
        if args.cuda:
            data_a, data_p, classes  = data_a.cuda(), data_p.cuda(), classes.cuda()
            data_a, data_p = Variable(data_a), Variable(data_p)
            if (args.highDim):
                out_a, mi_out_a = model(data_a)
                out_p , mi_out_p =  model(data_p)
            else:
                out_a = model(data_a)
                out_p = model(data_p)
                mi_out_a = out_a
                mi_out_p = out_p


        #Now the losses are calculated
        loss_prim = torch.tensor(0.0).cuda()
        loss_mi =  torch.tensor(0.0).cuda()
        loss_hyper = torch.tensor(0.0).cuda()

        currSamples= torch.cat((out_a, out_p),0)
        currClasses = torch.cat((classes, classes),0)


        #Hanlde primary loss
        start_loss = time.time()
        if args.PrimLoss == "Triplet":
            help, cross_chosen = loss_TripletHiB(currSamples,
                                    currClasses,
                                    crossSamples,
                                    currClasses,
                                    args.samplesPerClass,
                                         cross_initialized)
            loss_prim += help
        elif args.PrimLoss == "Positive":
             loss_prim += loss_OnlyPositive (currSamples, currClasses, args.samplesPerClass)
        end_loss = time.time()
        #else:
            # if None do nothing

        #Handle MI Loss
        if args.MiLoss in ['JSD', 'DV', 'infoNCE']:
            loss_mi += loss_MI (currSamples,currClasses,crossSamples, crossClasses, args.MiLoss,cross_initialized)
        #else:
            # if None do nothing

        if args.HyperLoss == 'Rho':
            loss_hyper += loss_Rho(currSamples, currClasses, crossSamples, crossClasses, cross_initialized)
        elif args.PrimLoss == "ArcLength":
            loss_hyper += loss_ArcLength(currSamples, currClasses, crossSamples, crossClasses, cross_initialized)
        elif args.PrimLoss == "MaxMean":
            loss_hyper += loss_MaxMean(currSamples, currClasses, crossSamples, crossClasses, cross_initialized)

        loss_total =loss_prim + loss_mi + loss_hyper

        # If we compare with SOSNet
        if args.lossSOS:
            loss_total += loss_SOS_new(currSamples,currClasses, crossSamples, crossClasses, cross_initialized)


        optimizer.zero_grad()
        start_gradient = time.time()
        loss_total.backward()
        end_gradient = time.time()
        optimizer.step()
        if args.adjLR:
            adjust_learning_rate(optimizer)

        # update cross memory samples
        if cross_flag:
            crossSamples[cross_ptr*batchSize: (cross_ptr+1)* batchSize, : ] = currSamples.detach()
            crossClasses[cross_ptr*batchSize: (cross_ptr+1)* batchSize ] = currClasses.detach()
            cross_ptr += 1
            if cross_ptr == num_cross_batches and not cross_initialized:
                print ("Cross-batch activated")
                cross_initialized = True

            cross_ptr = cross_ptr%num_cross_batches


        # Write Data for tensorboard visualization
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('Loss/train_total', loss_total.item(), batch_idx + epoch*len(train_loader))
            writer.add_scalar('Loss/train_primary', loss_prim.item(), batch_idx + epoch*len(train_loader))
            writer.add_scalar('Loss/train_MI', loss_mi.item(), batch_idx + epoch*len(train_loader))
            writer.add_scalar('Loss/train_hyper', loss_hyper.item(), batch_idx + epoch*len(train_loader))
            writer.add_scalar('Time/loss', end_loss - start_loss, batch_idx + epoch*len(train_loader))
            writer.add_scalar('Time/gradient', end_gradient - start_gradient, batch_idx + epoch*len(train_loader))
            writer.add_scalar('Time/cross_chosen', cross_chosen, batch_idx + epoch*len(train_loader))
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset)*len(data_a),
                           100. * batch_idx / len(train_loader),
                    loss_total.item()))
        

    if (args.enable_logging):
        logger.log_value('loss', loss_total.item()).step()
    try:
        os.stat('{}{}'.format(myModelDir,suffix))
    except:
        os.makedirs('{}{}'.format(myModelDir,suffix))

    #Save current weights of the network
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(myModelDir,suffix,epoch))
    return loss_total.item()

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

        if (args.highDim):
            out_a = model(data_a)[0]
            out_p = model(data_p)[0]
        else:
            out_a = model(data_a)
            out_p = model(data_p)
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
        1.0 - float(group['step']) * float(args.batch_size/2) / (args.n_triplets * float(args.epochs)))
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

    if args.cuda:
        model.cuda()
    optimizer1 = create_optimizer(model.features, args.lr)

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

    # Train and test for the number
    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, epoch, logger, triplet_flag)

        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])

       #randomize train loader batches
        #train_loader, test_loaders2 = create_loaders(args, dataset_names)


if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = os.path.join(args.log_dir, suffix)
    DESCS_DIR = os.path.join(LOG_DIR, 'temp_descs')
    logger, file_logger = None, None
    if (args.highDim):
        model = HardNetExtended(args.skipInit)
    else:
        model = HardNet(args.skipInit)
    if(args.enable_logging):
        from Loggers import Logger, FileLogger
        logger = Logger(LOG_DIR)

    train_loader, test_loaders = create_loaders(args, dataset_names)
    main(train_loader, test_loaders, model, logger, file_logger)
