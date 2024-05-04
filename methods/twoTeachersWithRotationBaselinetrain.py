import backbone_rot
from backbone_rot import SimpleBlock
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
print("2 Teachers with rotation")
class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        self.feature    = backbone_rot.ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten = True)
        
        teacher_checkpoint_1 = torch.load('/workspace/data/ldp-net/benchmarkwithRotationSubnet/checkpoints/miniImageNet/ResNet10_baseline_aug_rotation/399.tar')
        
        state1 = teacher_checkpoint_1['state']
        
        new_state_dict_1 = {}
        for key, value in state1.items():
            if 'rotm' not in key and 'classifier' not in key:
                new_key = key.replace('feature.', '')

                new_state_dict_1[new_key] = value

        teacher_model_1 = backbone_rot.ResNet(SimpleBlock, 
                                                   [1,1,1,1],
                                                   [64,128,256,512], 
                                                   flatten = True)
        
        teacher_model_1.load_state_dict(new_state_dict_1)

        self.teacher_model_1 = teacher_model_1


        teacher_checkpoint_2 = torch.load('/workspace/data/ldp-net/grayBenchmark128/checkpoints/miniImageNet/ResNet10_baseline_aug_bw/399.tar')

        state2 = teacher_checkpoint_2['state']

        new_state_dict_2 = {}
        for key, value in state2.items():
            if 'colr' not in key and 'classifier' not in key:
                new_key = key.replace('feature.', '')

                new_state_dict_2[new_key] = value


        teacher_model_2 = backbone_rot.ResNet(SimpleBlock, 
                                                   [1,1,1,1],
                                                   [64,128,256,512], 
                                                   flatten = True)

        teacher_model_2.load_state_dict(new_state_dict_2)
        
        self.teacher_model_2 = teacher_model_2





        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

        
        
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
    
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone_rot.distLinear(self.feature.final_feat_dim, num_class)

        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()

        # extra sub-network for rotation prediction
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


    # In[13]:


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


    def forward(self,x):
        x    = x.cuda()
        out  = self.feature.forward(x)
        scores = self.classifier(out)
        return scores

    def forward_loss(self, x, y):
        y = Variable(y.cuda())
        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))

        return self.loss_fn(scores, y )



    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.cuda()
            rot_img = self.create_4rotations_images(x)

            labels_rotation = self.create_rotations_labels(len(x), rot_img)
            rot_img_size =  rot_img.size(0)

            rot_img = rot_img.cuda()
            y = y.cuda()
            labels_rotation = labels_rotation.cuda()

            target_rot = y.repeat(4).view(-1)

            lt = self.feature(rot_img, is_feat = True)[0]


            logits_rot=self.feature.rotation(lt)
            logits_rot = logits_rot.view(-1, 4)  # there are 4 rotation classes
            logits_rot = logits_rot.float()

            labels_rotation = labels_rotation.view(-1)

            labels_rotation = labels_rotation.long()

            loss_rot = nn.CrossEntropyLoss()(logits_rot, labels_rotation)

            self.teacher_model_1 = self.teacher_model_1.cuda()
            self.teacher_model_1 = self.teacher_model_1.eval()

            self.teacher_model_2 = self.teacher_model_2.cuda()
            self.teacher_model_2 = self.teacher_model_2.eval()

            with torch.no_grad():
                logits_teacher_1 = self.teacher_model_1(x)
                logits_teacher_1 = logits_teacher_1.view(-1, 4).float()

                logits_teacher_2 = self.teacher_model_2(x)
                logits_teacher_2 = logits_teacher_2.view(-1, 4).float()

            logits_student = self.feature(x)
            logits_student = logits_student.view(-1, 4).float()


            kl_loss_1 = self.kl_loss_fn(F.log_softmax(logits_student, dim=1),
                                      F.softmax(logits_teacher_1, dim=1))


            kl_loss_2 = self.kl_loss_fn(F.log_softmax(logits_student, dim=1),
                                      F.softmax(logits_teacher_2, dim=1))


            loss = (self.forward_loss(rot_img, target_rot) + 
                    loss_rot + 
                    kl_loss_1 + 
                    kl_loss_2)
            
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))

    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration
