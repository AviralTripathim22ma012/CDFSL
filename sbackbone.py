import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)





# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)

    
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)


        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:

            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)


        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out



class ResNet(nn.Module):
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                           bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)


        self.av1 = nn.AdaptiveAvgPool2d((14, 14))
        self.av2 = nn.AdaptiveAvgPool2d((14, 14))
        self.av3 = nn.AdaptiveAvgPool2d((14, 14))

        self.ac1 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.ac2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.ac3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)


        self.ac1i = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.ac2i = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.ac3i = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)


        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)

        self.sigmoid = nn.Sigmoid()

        self.av_at = nn.AdaptiveAvgPool2d(1)


    def forward(self,x):
        # out = self.trunk(x)
        # print(self.trunk)
        print ("Attention")
        out = self.trunk[0](x)
        out = self.trunk[1](out)
        out = self.trunk[2](out)
        out = self.trunk[3](out)

        out4 = self.trunk[4](out)
        out4a = self.ac1(out4)
        out4a = self.av_at(out4a)
        out4a = self.flatten(out4a)
        out4a = self.fc1(out4a)
        out4a = self.fc2(out4a)
        out4a = torch.unsqueeze(out4a, dim=2)  # Add singleton dimension along height
        out4a = torch.unsqueeze(out4a, dim=3)  # Add singleton dimension along width
        out4a = self.ac1i(out4a)
        out4a = self.sigmoid(out4a)
        out = self.trunk[4](out)
        out = out + out4*out4a


        out5 = self.trunk[5](out)
        out5a = self.ac2(out5)
        out5a = self.av_at(out5a)
        out5a = self.flatten(out5a)
        out5a = self.fc1(out5a)
        out5a = self.fc2(out5a)
        out5a = torch.unsqueeze(out5a, dim=2)  # Add singleton dimension along height
        out5a = torch.unsqueeze(out5a, dim=3)  # Add singleton dimension along width
        out5a = self.ac2i(out5a)
        out5a = self.sigmoid(out5a)
        out = self.trunk[5](out)
        out = out + out5*out5a


        out6 = self.trunk[6](out)
        out6a = self.ac3(out6)
        out6a = self.av_at(out6a)
        out6a = self.flatten(out6a)
        out6a = self.fc1(out6a)
        out6a = self.fc2(out6a)
        out6a = torch.unsqueeze(out6a, dim=2)  # Add singleton dimension along height
        out6a = torch.unsqueeze(out6a, dim=3)  # Add singleton dimension along width
        out6a = self.ac3i(out6a)
        out6a = self.sigmoid(out6a)
        out = self.trunk[6](out)
        out = out + out6*out6a




        # attention
        out4 = self.av1(out4)
        out4 = self.ac1(out4)

        out5 = self.av2(out5)
        out5 = self.ac2(out5)

        out6 = self.av3(out6)
        out6 = self.ac3(out6)

        avg = (out4 + out5 + out6) # [512 14 14]
        avg = self.av_at(avg) # [512 1 1]
        avg = self.flatten(avg) # [512]
        avg = self.fc1(avg) # [256]
        avg = self.fc2(avg) # [512]
        avg = torch.unsqueeze(avg, dim=2)  # Add singleton dimension along height
        avg = torch.unsqueeze(avg, dim=3)  # Add singleton dimension along width

        attention = self.sigmoid(avg)



        out7 = self.trunk[7](out)
        out7a = self.av_at(out7)
        out7a = self.flatten(out7a)
        out7a = self.fc1(out7a)
        out7a = self.fc2(out7a)
        out7a = torch.unsqueeze(out7a, dim=2)  # Add singleton dimension along height
        out7a = torch.unsqueeze(out7a, dim=3)  # Add singleton dimension along width

        out = out7 + out7 * attention + out7*out7a


        out = self.trunk[8](out)
        out = self.trunk[9](out)

        return out




    def rotation(self,z):
        h = z
        h=self.rotm(z)
        h=self.rotm_avgpool(h)
        h = h.view(h.size(0), -1)
        h=self.rotm_class(h)
        return h




    #def colorization(self,z):
     #   h=self.colr(z)
     #   return h




def ResNet10( flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)




