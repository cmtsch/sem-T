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



class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, n_triplets = 50000, fliprot = False, batch_size = None, uniquePairs=True,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.train = train
        self.n_triplets = n_triplets
        self.fliprot = fliprot
        self.batch_size = batch_size
        self.unique_pairs = uniquePairs

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets, self.batch_size, self.unique_pairs)

    @staticmethod
    def generate_triplets(labels, num_triplets, batch_size, unique_pairs):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]


        #lengths = {k: len(v) for k, v in indices.items()}
        #maxSamples = max(lengths, key=lengths.get)
        #print ( "Most patches for a given class: " + str(len(indices[maxSamples])))
        #print ( "Total number of patches: " + str(sum(lengths.values())))
        
        tripletCtr = 0
        already_idxs = set()

        if (unique_pairs):
            ##### SAMPLING B ######
            ## Alternative way fo generating pairs
            ## Iterate over all classes
            ## Consider all pair-wise possibilities
            while (tripletCtr < num_triplets):
                # randomly select a class label
                c = np.random.randint(0, n_classes)
                while c in already_idxs:
                    c = np.random.randint(0, n_classes)
                already_idxs.add(c)

                if (len (already_idxs) == n_classes):
                    already_idxs = set()

                # take all combinations
                for n1 in range(len(indices[c])):
                    for n2 in range (n1+1, len(indices[c])):
                        triplets.append([indices[c][n1], indices[c][n2], c])
                        tripletCtr += 1
        else:
            ##### SAMPLING A ######
            # add only unique indices in batch
            #duplicateCounter = 0

            for x in tqdm(range(num_triplets)):
                if len(already_idxs) >= batch_size:
                    already_idxs = set()

                #if tripletCtr >= batch_size:
                    #already_idxs = set()
                    #print ("There are " + str(duplicateCounter) + " duplicates in that batch of size " + str(tripletCtr))
                    #duplicateCounter = 0
                    #tripletCtr = 0
                c = np.random.randint(0, n_classes)
                if c in already_idxs:
                    #if (unique_pairs):
                    #    duplicateCounter += 1
                    #else: 
                    while c in already_idxs:
                        c = np.random.randint(0, n_classes)
                already_idxs.add(c)

                # find subIDs of patche in class c
                if len(indices[c]) == 2:  # hack to speed up process
                    n1, n2 = 0, 1
                else:
                    n1 = np.random.randint(0, len(indices[c]))
                    n2 = np.random.randint(0, len(indices[c]))
                    while n1 == n2:
                        n2 = np.random.randint(0, len(indices[c]))
                triplets.append([indices[c][n1], indices[c][n2], c])
                #tripletCtr +=1

        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, label = self.data[t[0]], self.data[t[1]], t[2]

        img_a = transform_img(a)
        img_p = transform_img(p)

        # transform images if required
        if self.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
        #return image patches and class label
        return (img_a, img_p, label)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)


def create_loaders(args , dataset_names):
    print ("Creation of data sets starts")

    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(args.training_set)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))
    transform_test = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor()])
    transform_train = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.RandomRotation(5,PIL.Image.BILINEAR),
            transforms.RandomResizedCrop(32, scale = (0.9,1.0),ratio = (0.9,1.1)),
            transforms.Resize(32),
            transforms.ToTensor()])
    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,), (args.std_image,))])
    if not args.augmentation:
        transform_train = transform
        transform_test = transform
    train_loader = torch.utils.data.DataLoader(
            TripletPhotoTour(train=True,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=args.training_set,
                             fliprot = args.fliprot,
                             n_triplets = args.n_triplets,
                             download=True,
                             transform=transform_train,
                             uniquePairs= args.uniquePairs),
                             batch_size=args.batch_size,
                             shuffle=False, **kwargs)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             TripletPhotoTour(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     fliprot = args.fliprot,
                     n_triplets = args.n_triplets,
                     download=True,
                     transform=transform_test,
                     uniquePairs = args.uniquePairs),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    print ("Creation of data sets has ended")
    return train_loader, test_loaders
