import backbone
import utils
#from backbone import BottleneckBlock
from backbone import SimpleBlock

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


print("Normal Pre Train with batch_size=64")

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        #self.feature    = model_func()
        #self.feature    = backbone.ResNet(BottleneckBlock, [1,1,1,1],[64,128,256,512], flatten = True)
        self.feature    = backbone.ResNet(SimpleBlock, [1,1,1,1], [64,128,256,512], flatten = True)
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)

        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores



    def forward_loss_1(self, x, y):
        y = Variable(y.cuda())

        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))

        return self.loss_fn(scores, y)



    def forward_loss_2(self, x, y):
        y = Variable(y.cuda())

        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))




        weights = []
        for i in range(len(scores)):
            logit_current = scores[i]
            logit_except_current = torch.cat((scores[:i], scores[i+1:]), dim=0)
            logit_except_current = torch.mean(logit_except_current, dim = 0)

            label_current = y[i].item()
            logit_same_class = [logit for j, logit in enumerate(scores) if y[j].item() == label_current]
            if len(logit_same_class) == 0:
              weight = 1
            else:
              kl_div = F.kl_div(F.log_softmax(logit_current),
                                F.softmax(logit_except_current))

              weight = 1 - 1000*kl_div
              weight = weight.clip(0.5, 1) + 0.5

            weights.append(weight)

        weights = torch.tensor(weights, device='cuda')

        ce_loss = nn.CrossEntropyLoss(reduction='none')(scores, y )

        weighted_ce_loss = ce_loss * weights
        weighted_ce_loss = weighted_ce_loss.sum() / weights.sum()

        return weighted_ce_loss

        #return self.loss_fn(scores, y )

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0

        #if epoch <= 200:
        #    loss_function = self.forward_loss_1
        #else:
        #    loss_function = self.forward_loss_2


        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            loss = self.forward_loss_1(x, y)
            #loss = loss_function(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))

    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration
