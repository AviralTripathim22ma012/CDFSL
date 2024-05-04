import backbone_rot
from backbone_rot import SimpleBlock
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
print("A")
class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        self.feature    = backbone_rot.ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten = True)
        
        teacher_checkpoint = torch.load('/workspace/data/ldp-net/benchmarkwithRotationSubnet/checkpoints/miniImageNet/ResNet10_baseline_aug_rotation/399.tar')
        
        state = teacher_checkpoint['state']
        
        new_state_dict = {}
        for key, value in state.items():
            if 'rotm' not in key and 'classifier' not in key:
                new_key = key.replace('feature.', '')

                new_state_dict[new_key] = value

        teacher_model_1 = backbone_rot.ResNet(SimpleBlock, 
                                                   [1,1,1,1],
                                                   [64,128,256,512], 
                                                   flatten = True)
        
        teacher_model_1.load_state_dict(new_state_dict)

        self.teacher_model_1 = teacher_model_1
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

        
        
        if loss_type == 'softmax':
            # print("Final Feature Dimension:", self.feature.final_feat_dim)
            # in_features = 64*7*7 #(64, 7, 7)
            # out_features = num_class

            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            #self.classifier = nn.Linear((64, 7, 7), num_class[0]) # Final Feature Dimension: [64, 7, 7]
            self.classifier.bias.data.fill_(0)
            #self.rot_classifier = nn.Linear(self.feature.final_feat_dim, 4)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone_rot.distLinear(self.feature.final_feat_dim, num_class)

        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()
        # self.model_rot = backbone_rot.ResNet(block = SimpleBlock, n_blocks = [1,1,1,1], num_classes=num_class, avg_pool=True)

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
        #print("forward function is called")
        x    = x.cuda()
        out  = self.feature.forward(x)
        #print("out.shap before reshaping: ", out.shape)
        #out = out.view(out.size(0), -1)
        #print("out.shape after reshaping: ", out.shape)
        #print("self.classifier.forward(out).shape: ", self.classifier.forward(out).shape)
        scores = self.classifier(out)
        #print("scores.shape: ", scores.shape)
        return scores

    def forward_loss(self, x, y):
        #print("forward_loss function is called")
        y = Variable(y.cuda())
        #print(y.shape)
        #print(x.shape)
        scores = self.forward(x)
        #print("scores:", scores.shape)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))

        return self.loss_fn(scores, y )


    #os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    #opt.n_gpu = torch.cuda.device_count()




    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.cuda()
            rot_img = self.create_4rotations_images(x)
            #input_tensor = torch.tensor(input)
            labels_rotation = self.create_rotations_labels(len(x), rot_img)
            rot_img_size =  rot_img.size(0)

            rot_img = rot_img.cuda()
            y = y.cuda()
            labels_rotation = labels_rotation.cuda()

            target_rot = y.repeat(4).view(-1)

            lt = self.feature(rot_img, is_feat = True)[0]
            # print("lt shape:", lt.shape)

            logits_rot=self.feature.rotation(lt)
            logits_rot = logits_rot.view(-1, 4)  # Assuming there are 4 rotation classes
            logits_rot = logits_rot.float()

            labels_rotation = labels_rotation.view(-1)
            #labels_rotation = torch.argmax(labels_rotation, dim=1)
            labels_rotation = labels_rotation.long()

            loss_rot = nn.CrossEntropyLoss()(logits_rot, labels_rotation)

            #loss_rot=  nn.CrossEntropyLoss(logits_rot, labels_rotation)
            # self.teacher_model_1 = self.teacher_model_1.load_state_dict(new_state_dict)
            self.teacher_model_1 = self.teacher_model_1.cuda()
            self.teacher_model_1 = self.teacher_model_1.eval()
            with torch.no_grad():
                logits_rot_teacher = self.teacher_model_1(x) #(rot_img)
                logits_rot_teacher = logits_rot_teacher.view(-1, 4).float()
            logits_rot_student = self.feature(x) #(rot_img)
            logits_rot_student = logits_rot_student.view(-1, 4).float()
       

            kl_loss = self.kl_loss_fn(F.log_softmax(logits_rot_student, dim=1),
                                      F.softmax(logits_rot_teacher, dim=1))


            loss = self.forward_loss(rot_img, target_rot) + loss_rot + kl_loss
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))

    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration
