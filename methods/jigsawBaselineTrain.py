import backbone_rot
from backbone_rot import SimpleBlock
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
import itertools
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

print('jigsaw')




class ShufflePatches(object):
    def __init__(self, patch_size):
        self.ps = patch_size

    def create_jigsaw_permutations(self):
        permutations = []
        for perm in itertools.permutations([0, 1, 2, 3]):
            permutations.append(perm)
        return permutations

    def __call__(self, x):
        # divide the batch of images into non-overlapping patches
        u = F.unfold(x, kernel_size=self.ps, stride=self.ps, padding=0)
        # permute the patches of each image in the batch
        jigsaw_permutations = self.create_jigsaw_permutations()
        permuted_patches = []
        for b_ in u:
            perm = torch.randperm(b_.shape[-1])
            permuted_patch = b_[:, perm]
            permuted_patches.append(permuted_patch)
        pu = torch.cat(permuted_patches, dim=0)
        # fold the permuted patches back together
        f = F.fold(pu, x.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
        perm = tuple(perm.tolist())
        permutation_index = jigsaw_permutations.index(perm)
        return f, permutation_index



class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type='softmax', jigsaw_permutations=None):
        super(BaselineTrain, self).__init__()
        #self.feature = model_func().cuda()
        self.feature    = backbone_rot.ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten = True)
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)


        self.jigsaw_classifier = nn.Linear(self.feature.final_feat_dim, 24)  # 24 possible permutations
        self.loss_type = loss_type
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()
        self.create_jigsaw_permutations()

    def create_jigsaw_permutations(self):
        permutations = []
        for perm in itertools.permutations([0, 1, 2, 3]):
            permutations.append(perm)
        return permutations


    def forward(self, x):

        x = Variable(x.cuda())

        batch_size = x.size(0)

        shuffled_images = []
        permutation_labels = []

        for i in range(batch_size):
            input_image = x[i]
            input_image = input_image.unsqueeze(0)
            shuffled_image, permutation_index = ShufflePatches(patch_size=112)(input_image)
            shuffled_images.append(shuffled_image)
            permutation_index = torch.tensor(permutation_index)
            permutation_labels.append(permutation_index)

        shuffled_images = torch.stack(shuffled_images)
        permutation_labels = torch.tensor(permutation_labels)

        #shuffled_image_tensor = shuffled_images[0].cpu()
        #shuffled_image = Image.fromarray((shuffled_image_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        #shuffled_image.save("/workspace/data/ldp-net/cdfslbenchmark/methods/shuffled_image.jpg")


        out = self.feature.forward(shuffled_images)
        out2 = self.feature.forward(x)
        classification_scores = self.classifier.forward(out)
        classification_scores_2 = self.classifier.forward(out2)
        jigsaw_scores = self.jigsaw_classifier.forward(out)


        return classification_scores, classification_scores_2, jigsaw_scores, permutation_labels





    def forward_loss(self, x, y, classification_weight=0.5, jigsaw_weight=0.5):
        y = Variable(y.cuda())
        classification_scores, classification_scores_2, jigsaw_scores, permutation_labels = self.forward(x)

        _, predicted = torch.max(classification_scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))

        # Compute the classification loss
        classification_loss = self.loss_fn(classification_scores, y)
        classification_loss_2 = self.loss_fn(classification_scores_2, y)
        # Compute the jigsaw classification loss
        jigsaw_labels = Variable(torch.LongTensor(permutation_labels).cuda())
        jigsaw_loss = self.loss_fn(jigsaw_scores, jigsaw_labels)

        # Compute the total loss
        total_loss = classification_loss + jigsaw_loss + classification_loss_2

        return total_loss





    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration
