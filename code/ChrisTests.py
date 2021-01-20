import Losses
import torch


anchors = torch.randn(5,3)
positives = torch.randn(5,3)
#classes = torch.randint(0,10,[3])
classes = torch.tensor([1,1,1,1,1])
print (classes)
#print(anchors)
#print(positives)
Losses.loss_Rho(anchors, positives, classes)