import backbone_rot
from backbone_rot import SimpleBlock
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

print("SimSiam with rotation")

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        #self.feature    = model_func()
        self.feature    = backbone_rot.ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten = True)


        hidden_dim=2048
        output_dim=512



        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone_rot.distLinear(self.feature.final_feat_dim, num_class)

        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.cosine_similarly = nn.CosineSimilarity(dim=-1)
        self.top1 = utils.AverageMeter()


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


        #self.projection_layer = nn.Linear(self.feature.final_feat_dim, hidden_dim, bias=False) # 512->2048

        self.predictor = nn.Sequential(
                         nn.Linear(self.feature.final_feat_dim, hidden_dim, bias=False),
                         nn.BatchNorm1d(hidden_dim),
                         nn.ReLU(inplace=True),
                         nn.Linear(hidden_dim, output_dim)
                         )


    def apply_2d_rotation(self, input_tensor1, rotation):

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




    def augmentation(self, x):
        B, C, H, W = x.size()
        augmentation = T.Compose([
            T.RandomResizedCrop(size=(H, W)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0)),
        ])
        return augmentation(x)


    def forward(self,x):
        x    = Variable(x.cuda())

        x1 = self.augmentation(x)
        x2 = self.augmentation(x)

        z1 = self.feature.forward(x1)
        z2 = self.feature.forward(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores, z1, z2, p1, p2



    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores, z1, z2, p1, p2 = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))

        loss_simsiam = -0.5 * ( self.cosine_similarly(p1, z2.detach()).mean() +
                                self.cosine_similarly(p2, z1.detach()).mean() )

        classification_loss = self.loss_fn(scores, y )

        return 0.1*loss_simsiam + classification_loss




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
