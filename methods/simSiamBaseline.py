import backbone_rot
from backbone_rot import SimpleBlock
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tv
import torchvision.transforms as T

print("SimSiam")

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        #self.feature    = model_func().cuda()

        self.feature    = backbone_rot.ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten = True)



        hidden_dim=2048
        output_dim=512

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)

        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.cosine_similarly = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.top1 = utils.AverageMeter()

        #self.projection_layer = nn.Linear(self.feature.final_feat_dim, hidden_dim, bias=False) # 512->2048

        self.predictor = nn.Sequential(
                         nn.Linear(self.feature.final_feat_dim, hidden_dim, bias=False),
                         nn.BatchNorm1d(hidden_dim),
                         nn.ReLU(inplace=True),
                         nn.Linear(hidden_dim, output_dim)
                         )

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


        #z1 = self.feature.forward(x1)
        #scores  = self.classifier.forward(z1)
        #z1 = self.projection_layer(z1)
        
        #z2 = self.feature.forward(x2)
        # scores2  = self.classifier.forward(z2)
        #z2 = self.projection_layer(z2)

        #p1 = self.predictor(z1)
        #p2 = self.predictor(z2)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))

        loss_simsiam = -0.5 * (self.cosine_similarly(p1, z2.detach()).mean() +
                               self.cosine_similarly(p2, z1.detach()).mean())
        classification_loss = self.loss_fn(scores, y )

        print(loss_simsiam)
        print(classification_loss)

        return 0.1*loss_simsiam + classification_loss

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()

            #loss_simsiam, classification_loss = self.forward_loss(x, y)

            #classification_loss.backward(retain_graph = True)
            #loss_simsiam.backward()

            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))

    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration
