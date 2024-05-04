import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations
import sys

import configs
import sbackbone
from sbackbone import SimpleBlock
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import imgs_miniImageNet_few_shot

print("RSN init : attention")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return torch.mean(alpha_weight * focal_loss)

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def augmentation(x):
    B, C, H, W = x.size()
    augmentation = T.Compose([
            T.RandomResizedCrop(size=(H, W)),
            T.RandomHorizontalFlip(p=1),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.9),
            T.RandomGrayscale(p=0.7),
            T.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0)),
    ])

    return augmentation(x)


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

def finetune(novel_loader, n_query = 15, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5): 
    correct = 0
    count = 0

    iter_num = len(novel_loader) 

    acc_all = []

    for idx, (x_cr, x, y) in enumerate(novel_loader):
        print (f'Processing Itteration {idx + 1}/{iter_num}')
        ###############################################################################################
        # load pretrained model on miniImageNet

        #print ("Loading pretrained model...")
        pretrained_model = sbackbone.ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten = True)
        tmp = torch.load("/workspace/data/ldp-net/benchmarkwithRotationSubnet/checkpoints/miniImageNet/ResNet10_baseline_aug_rotation/399.tar")

        new_state_dict = {}
        for key, value in tmp['state'].items():
            if 'rotm' not in key and 'classifier' not in key:
               new_key = key.replace('feature.', '')

               new_state_dict[new_key] = value

        print('\n')

        pretrained_model.load_state_dict(new_state_dict,  strict=False)
        pretrained_model.cuda()

        ###############################################################################################

                                                ## T E A C H E R ##

        ###############################################################################################

        #teacher = backbone.ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten = True)
        #tmp_t = torch.load("/workspace/data/ldp-net/checkpoint/399JigsawRotation.tar")

        #new_state_dict_t = {}
        #for key, value in tmp_t['state'].items():
        #    if 'rotm' not in key and 'classifier' not in key:
        #       new_key = key.replace('feature.', '')

        #       new_state_dict_t[new_key] = value

        print('\n')

        #teacher.load_state_dict(new_state_dict_t)
        #teacher.cuda()

        ###############################################################################################

        classifier = Classifier(pretrained_model.final_feat_dim, n_way)

        ###############################################################################################
        n_query = x.size(1) - n_support
        x = x.cuda()
        x_cr = x_cr.cuda()
        x_var = Variable(x)

    
        batch_size = 4
        support_size = n_way * n_support 
       
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) 
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        #x_cr_a_i = x_cr[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)

         ###############################################################################################
        #loss_fn = nn.CrossEntropyLoss().cuda()
        loss_fn = FocalLoss().cuda()
        criterion_div = DistillKL(4).cuda()
        classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        

        if freeze_backbone is False:
            delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = 0.01)


        pretrained_model.cuda()
        classifier.cuda()
        ###############################################################################################
        total_epoch = 100

        if freeze_backbone is False:
            #print ("Training backbone network...")
            pretrained_model.train()
        else:
            #print ("Training backbone network...")
            pretrained_model.eval()
        
        classifier.train()

        for epoch in range(total_epoch):
            #print(f"Epoch [{epoch + 1}/{total_epoch}]")
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                #print (f"Processing batch {j // batch_size + 1}/{(support_size + batch_size - 1) // batch_size}")
                classifier_opt.zero_grad()
                if freeze_backbone is False:
                    delta_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
               
                z_batch = x_a_i[selected_id] # [4 3 224 224]
                #z_aug_batch = augmentation(z_batch) # [4 3 224 224]
                #z_cr_batch = x_cr_a_i[selected_id]
                y_batch = y_a_i[selected_id] 
                #####################################

                output = pretrained_model(z_batch)
                #output_aug = pretrained_model(z_aug_batch)
                #logit_s = output_aug

                output = classifier(output)
                #output_aug = classifier(output_aug)

                ce_1 = loss_fn(output, y_batch)
                #ce_2 = loss_fn(output_aug, y_batch)

                #####################################
                #logit_s = logit_s
                #with torch.no_grad():
                #    logit_t = teacher(z_batch)
                #loss_div = criterion_div(logit_s, logit_t)

                loss = ce_1 #+ ce_2 + loss_div

                #####################################
                loss.backward()

                classifier_opt.step()

                if freeze_backbone is False:
                    delta_opt.step()
                
                #print(f"\tBatch [{j + 1}/{support_size}] Loss: {loss.item():.4f}")

        pretrained_model.eval()
        classifier.eval()

        output = pretrained_model(x_b_i.cuda())
        scores = classifier(output)
       
        y_query = np.repeat(range( n_way ), n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        print (f'Accuracy for batch {idx + 1}: {correct_this / count_this * 100:.2f}%')
        print (correct_this/ count_this *100)
        acc_all.append((correct_this/ count_this *100))
        
        ###############################################################################################

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print ('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

    torch.save({'state' : pretrained_model.state_dict()},
               '/workspace/data/ldp-net/checkpoint/attention.tar')

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    ##################################################################
    image_size = 224
    iter_num = 600

    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   
    freeze_backbone = params.freeze_backbone
    ##################################################################
    pretrained_dataset = "miniImageNet"

    #dataset_names = ["miniImageNet"]
    dataset_names = ["miniImageNet"]
    novel_loaders = []

    print ("miniImageNet")
    datamgr             =  imgs_miniImageNet_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug1=True, aug2=False)
    novel_loaders.append(novel_loader)
    
    
    #########################################################################
    for idx, novel_loader in enumerate(novel_loaders):
        print (dataset_names[idx])

        print (freeze_backbone)

        # replace finetine() with your own method
        with open(f'/workspace/data/ldp-net/logs/training_attention.txt', 'w') as f:
            sys.stdout = f
            finetune(novel_loader, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)


