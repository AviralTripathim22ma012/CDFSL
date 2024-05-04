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

print('jigsaw with rotation')

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


        self.feature.rotm = nn.Sequential(nn.Conv2d(512,512,3,1,1),
                                            nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1),
                                            nn.Conv2d(512,512,3,1,1),
                                            nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1),
                                            nn.Conv2d(512,512,3,1,1),
                                            nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1),
                                            nn.Conv2d(512,512,3,1,1),
                                            nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1))
        self.feature.rotm_avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature.rotm_class = nn.Linear(512,4)



    def apply_2d_rotation(self, input_tensor1, rotation):
        """Apply a 2d rotation of 0, 90, 180, or 270 degrees to a tensor.

        The code assumes that the spatial dimensions are the last two dimensions,
        e.g., for a 4D tensors, the height dimension is the 3rd one, and the width
        dimension is the 4th one.
        """
        assert input_tensor1.dim() >= 2
        input_tensor = input_tensor1.clone()

        height_dim = input_tensor.dim() - 2
        width_dim = height_dim + 1

        flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
        flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
        spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)

        if rotation == 0:  # 0 degrees rotation
            return input_tensor
        elif rotation == 90:  # 90 degrees rotation
            return flip_upside_down(spatial_transpose(input_tensor))
        elif rotation == 180:  # 90 degrees rotation
            return flip_left_right(flip_upside_down(input_tensor))
        elif rotation == 270:  # 270 degrees rotation / or -90
            return spatial_transpose(flip_upside_down(input_tensor))
        else:
            raise ValueError(
                "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
            )


    def create_4rotations_images(self, images, stack_dim=None):
        """Rotates each image in the batch by 0, 90, 180, and 270 degrees."""
        images_4rot = []
        for r in range(4):
            images_4rot.append(self.apply_2d_rotation(images, rotation=r * 90))

        if stack_dim is None:
            images_4rot = torch.cat(images_4rot, dim=0)
        else:
            images_4rot = torch.stack(images_4rot, dim=stack_dim)

        return images_4rot


    # In[14]:


    def create_rotations_labels(self, batch_size, device):
        """Creates the rotation labels."""
        labels_rot = torch.linspace(0, 3, steps=4, dtype=torch.float32).view(4, 1).to(device)
        labels_rot = labels_rot.repeat(1, batch_size).view(-1)
        return labels_rot




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

        out = self.feature.forward(shuffled_images)
        classification_scores = self.classifier.forward(out)
        jigsaw_scores = self.jigsaw_classifier.forward(out)


        return classification_scores, jigsaw_scores, permutation_labels


    def forward_loss(self, x, y, classification_weight=0.5, jigsaw_weight=0.5):
        y = Variable(y.cuda())
        classification_scores, jigsaw_scores, permutation_labels = self.forward(x)

        _, predicted = torch.max(classification_scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))

        # Compute the classification loss
        classification_loss = self.loss_fn(classification_scores, y)

        # Compute the jigsaw classification loss
        jigsaw_labels = Variable(torch.LongTensor(permutation_labels).cuda())
        jigsaw_loss = self.loss_fn(jigsaw_scores, jigsaw_labels)

        # Compute the total loss
        total_loss = classification_loss + jigsaw_loss

        return total_loss







    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()


            rot_img = self.create_4rotations_images(x)
            labels_rotation = self.create_rotations_labels(len(x), rot_img)

            rot_img = rot_img.cuda()
            y = y.cuda()
            labels_rotation = labels_rotation.cuda()

            target_rot = y.repeat(4).view(-1)
            target_rot = target_rot.cuda()

            lt = self.feature(rot_img, is_feat = True)[0]
            # print("lt shape:", lt.shape)

            logits_rot=self.feature.rotation(lt)
            logits_rot = logits_rot.view(-1, 4)  # Assuming there are 4 rotation classes
            logits_rot = logits_rot.float()

            labels_rotation = labels_rotation.view(-1)
            labels_rotation = labels_rotation.long()

            loss_rot = nn.CrossEntropyLoss()(logits_rot, labels_rotation)

            loss = self.forward_loss(rot_img, target_rot) + loss_rot

            #loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))

    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration
