from __future__ import division, print_function
import torch
import torch.nn.init
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import random
import copy
import PIL
import HardNetModel
from Utils import L2Norm, cv2_scale, np_reshape

class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, n_triplets = 50000, fliprot = False, batch_size = None, samplesPerClass=2, transformFlag = False,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.train = train
        self.n_triplets = n_triplets
        self.fliprot = fliprot
        #self.batch_size = batch_size
        self.batch_size = 1

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets, batch_size, samplesPerClass,
                                                   self.data, self.transform, transformFlag)

    @staticmethod
    def generate_triplets(labels, num_triplets, batch_size, samplesPerClass, data,transform, transformFlag):


        def transform_img(img):
            img = img.numpy()
            if transformFlag:
                if (random.random() < 0.3):
                    #stddev = np.random.uniform(5,95)
                    stddev = np.random.uniform(5,30)
                    noise = np.random.normal(0., stddev, size=img.shape)
                    tmpImg = img.copy()
                    tmpImg = (tmpImg + noise)
                    img = tmpImg.clip(0.,255.).astype(np.uint8)
                if (random.random() < 0.3):
                    strength = np.random.uniform(0.8,1.2)
                    contrasted_img = img.copy()
                    mean = np.mean(contrasted_img)
                    contrasted_img = (contrasted_img-mean) * strength + mean
                    img = contrasted_img.clip(0.,255.).astype(np.uint8)
                if (random.random() < 0.3):
                    img = transform[0](img)
                else:
                    img = transform[1](img)
            else:
                #Just apply the normal transformation
                img = transform(img)
            return img

        def create_indices(_labels):
            #idxMap = dict()
            idxMap = [ [] for _ in range(n_classes)]

            for patchIdx, classIdx in enumerate(_labels):
                #if classIdx not in idxMap:
                #    idxMap[classIdx] = []
                if (classIdx >= n_classes):
                    print("Not good")

                idxMap[classIdx].append(patchIdx)
            return idxMap

        def generatePartnerClasses(model):
            partnerClasses = torch.empty(n_classes, dtype=torch.int32)
            hBatchSize = 10000
            idx = 0

            #print ("n_classes is " + str(n_classes))
            while (idx < n_classes ):
                upperLim = min(n_classes, idx + hBatchSize)
                with torch.no_grad():
                    currIndices = [item[0] for item in indices[idx:upperLim]]
                    #prediction = torch.empty([upperLim -idx , 128])
                    ctr=0
                    tmp = torch.empty([upperLim-idx,1, 32, 32])
                    for ind in currIndices:
                        tmp[ctr,:, :, :] =transform_img(data[ind])
                        #prediction[ctr,:] = model(torch.unsqueeze(transform_img(data[ind]),0))
                        ctr+=1
                    prediction = model(tmp)

                # prediction is 10k x 128 matrix
                dist_mat = torch.mm(prediction, prediction.t())
                minInds = torch.argmax(dist_mat, dim = 0)
                partnerClasses[ idx : upperLim] = minInds

                idx = upperLim

            return partnerClasses

        def hardPairsBatch(partnerClass):

            batchCtr = 0
            batch1 = torch.empty(batch_size, 1, 32, 32)
            batch2 = torch.empty(batch_size, 1, 32, 32)
            classes = torch.empty(batch_size)
            already_idxs = set()

            while (batchCtr < batch_size):

                # randomly select a class label
                c1 = np.random.randint(0, n_classes)
                while c1 in already_idxs:
                    c1 = np.random.randint(0, n_classes)
                already_idxs.add(c1)

                # partnerClass is a map (or rather a vector) which contains a difficult partner class for each class
                c2 = partnerClass[c1]

                # take all combinations
                for c in [c1, c2]:
                    classCtr = 0
                    for n1 in range(len(indices[c])):
                        for n2 in range(n1 + 1, len(indices[c])):
                            # Here we do not necessarily need to transform, could just use original patches
                            if classCtr < samplesPerClass:
                                tmpPathc = transform_img(data[indices[c][n1]])
                                # print (tmpPathc.size())
                                batch1[batchCtr, :, :, :] = transform_img(data[indices[c][n1]])
                                batch2[batchCtr, :, :, :] = transform_img(data[indices[c][n2]])
                                classes[batchCtr] = c
                                batchCtr += 1
                                classCtr += 1

                    # New stuff
                    while (classCtr < samplesPerClass):
                        aug1 = np.random.randint(0, len(indices[c]))
                        aug2 = np.random.randint(0, len(indices[c]))
                        batch1[batchCtr, :, :, :] = transform_img(data[indices[c][aug1]])
                        batch2[batchCtr, :, :, :] = transform_img(data[indices[c][aug2]])
                        classes[batchCtr] = c
                        classCtr += 1
                        batchCtr += 1
            return batch1, batch2, classes

        def nonUniqueBatch(samplesPerClass):
            batchCtr=0
            batch1 = torch.empty(batch_size, 1, 32, 32)
            batch2 = torch.empty(batch_size, 1, 32, 32)
            classes = torch.empty(batch_size)
            already_idxs = set()

            while (batchCtr < batch_size):

                # randomly select a class label
                c = np.random.randint(0, n_classes)
                while c in already_idxs:
                    c = np.random.randint(0, n_classes)
                already_idxs.add(c)

                # take all combinations
                classCtr= 0
                for n1 in range(len(indices[c])):
                    for n2 in range (n1+1, len(indices[c])):
                        # Here we do not necessarily need to transform, could just use original patches
                        if classCtr < samplesPerClass:
                            tmpPathc = transform_img(data[indices[c][n1]])
                            #print (tmpPathc.size())
                            batch1[batchCtr,:,:,:] = transform_img(data[indices[c][n1]])
                            batch2[batchCtr,:, :, :] = transform_img(data[indices[c][n2]])
                            classes[batchCtr] = c
                            batchCtr +=1
                            classCtr += 2

                # New stuff
                while ( classCtr < samplesPerClass):
                    aug1 = np.random.randint(0, len(indices[c]) )
                    aug2 = np.random.randint(0, len(indices[c]) )
                    batch1[batchCtr, :,:, :] = transform_img(data[indices[c][aug1]])
                    batch2[batchCtr, :,:, :] = transform_img(data[indices[c][aug2]])
                    classes[batchCtr] = c
                    classCtr += 2
                    batchCtr += 1
            return batch1, batch2, classes

        def uniqueBatch():
            already_idxs = set()
            batch1 = torch.empty(batch_size, 1, 32, 32)
            batch2 = torch.empty(batch_size, 1, 32, 32)
            classes = torch.empty(batch_size)
            for batchCtr in range(batch_size):
                c = np.random.randint(0, n_classes)
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

                batch1[batchCtr, :, :, :] = transform_img(data[indices[c][n1]])
                batch2[batchCtr, :, :, :] = transform_img(data[indices[c][n2]])
                classes[batchCtr] = c

            return batch1, batch2, classes

        triplets = []
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        indices = create_indices(labels.numpy())

        tripletCtr = 0
        chooser = 0

        if (samplesPerClass == 2):
            uniqueFlag = True
        else:
            uniqueFlag = False

        if False:
            model = HardNetModel.HardNet()
            # load a specific set of pretrained parameters
            print("Trying to load pre-trained model")
            checkpoint = torch.load('pretrained/train_liberty/checkpoint_liberty_no_aug.pth')
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded pre-trained model")

            partnerClasses = generatePartnerClasses(model)
            print("Determined partner classes")

        while (tripletCtr < num_triplets - batch_size):

            #print("Now generating a hardPairsBatch")
            #batch1, batch2, classes = hardPairsBatch(partnerClasses)
            #print("Done generating a hardPairsBatch")

            if (uniqueFlag):
                batch1, batch2, classes = uniqueBatch()
            else:
                batch1, batch2, classes = nonUniqueBatch(samplesPerClass)

            triplets.append([batch1, batch2, classes, uniqueFlag])
            tripletCtr += batch_size
            chooser+=1

        return triplets

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img
        # If this is a test set, just return patches and bool value (match / no match)
        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, label, uniqueFlag = t[0], t[1], t[2], t[3]
        img_a = a
        img_p = p
        return (img_a, img_p, label, uniqueFlag)

    def __len__(self):
        if self.train:
            return len(self.triplets)
            #return self.triplets.size(0)
        else:
            return self.matches.size(0)


def create_loaders(args , dataset_names):
    print ("Creation of data sets starts")

    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(args.training_set)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))

    # This transformation does not do anything, but reshape the patch to 32x32
    transform_test = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor()])
    #This transformation does a cropping and rotating
    transform_train1 = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.RandomRotation(5,PIL.Image.BILINEAR, fill=(0,)),
            transforms.RandomResizedCrop(32, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.Resize(32),
            transforms.ToTensor()])
    #This transformation is the same as transfrom_test, only resizes
    transform_train2 = transforms.Compose([
        transforms.Lambda(np_reshape64),
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor()])
    #This transformation normalizes, why????
    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,), (args.std_image,))])
    transform_train = [transform_train1, transform_train2]
    transformFlag =True
    if not args.augmentation:
        transform_train = transform
        transform_test = transform
        transformFlag = False
    train_loader = torch.utils.data.DataLoader(
            TripletPhotoTour(train=True,
                             batch_size=int(args.batch_size / 2),
                             root=args.dataroot,
                             name=args.training_set,
                             fliprot = args.fliprot,
                             n_triplets = args.n_triplets,
                             download=True,
                             transform=transform_train,
                             transformFlag = transformFlag,
                             samplesPerClass= args.samplesPerClass),
                             batch_size=1,
                             shuffle=True, **kwargs)

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
                     transformFlag=transformFlag),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    print ("Creation of data sets has ended")
    return train_loader, test_loaders
