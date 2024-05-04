import backbone_rot
from backbone_rot import SimpleBlock
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

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

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
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
            y = y.cuda()


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



            loss = self.forward_loss(x, y) + kl_loss_1 + kl_loss_2
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration
