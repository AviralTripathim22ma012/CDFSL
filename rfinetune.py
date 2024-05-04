import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations

import configs
import backbone_rot
from backbone_rot import SimpleBlock
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file

from utils import *

from datasets import miniImageNet_few_shot

print("finetuning with Rotation JRSN Initialisation")


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


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x



        # extra sub-network for rotation prediction
rotm = nn.Sequential(nn.Conv2d(512,512,3,1,1),
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
                     nn.LeakyReLU(0.1),

                     nn.AdaptiveAvgPool2d(1)
                     #nn.Linear(512,4)
                     ).cuda()

        #h=self.rotm_avgpool(h)
        #h = h.view(h.size(0), -1)
        #h=self.rotm_class(h)


#rotm_avgpool = nn.AdaptiveAvgPool2d(1).cuda()
rotm_class = nn.Linear(512,4).cuda()

def apply_2d_rotation(input_tensor1, rotation):
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


def create_4rotations_images(images, stack_dim=None):
        images_4rot = []
        for r in range(4):
            images_4rot.append(apply_2d_rotation(images, rotation=r * 90))

        if stack_dim is None:
            images_4rot = torch.cat(images_4rot, dim=0)
        else:
            images_4rot = torch.stack(images_4rot, dim=stack_dim)

        return images_4rot



def create_rotations_labels(batch_size, device):
        labels_rot = torch.linspace(0, 3, steps=4, dtype=torch.float32).view(4, 1).cuda()
        labels_rot = labels_rot.repeat(1, batch_size).view(-1)
        return labels_rot


def finetune(novel_loader, n_query = 15, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5):
    correct = 0
    count = 0

    iter_num = len(novel_loader)

    acc_all = []

    for idx, (x, y) in enumerate(novel_loader):
        print (f'Processing Itteration {idx + 1}/{iter_num}')
        ###############################################################################################
        #                                       PRE-TRAINED MODEL                                     #
        # load pretrained model on miniImageNet

        #print ("Loading pretrained model...")
        pretrained_model = backbone_rot.ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten = True)
        tmp = torch.load("/workspace/data/ldp-net/checkpoint/399JigsawRotation.tar")

        new_state_dict = {}
        for key, value in tmp['state'].items():
            if 'rotm' not in key and 'classifier' not in key:
               new_key = key.replace('feature.', '')
               new_state_dict[new_key] = value

        print('\n')

        pretrained_model.load_state_dict(new_state_dict)
        pretrained_model.cuda()

        ###############################################################################################

        #                                    	T E A C H E R                                         #
        teacher = backbone_rot.ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten = True)
        tmp_t = torch.load("/workspace/data/ldp-net/checkpoint/399JigsawRotation.tar")
        new_state_dict_t = {}
        for key, value in tmp_t['state'].items():
            if 'rotm' not in key and 'classifier' not in key:
               new_key = key.replace('feature.', '')
               new_state_dict_t[new_key] = value

        print('\n')

        teacher.load_state_dict(new_state_dict_t)
        teacher.cuda()

        ###############################################################################################

        classifier = Classifier(pretrained_model.final_feat_dim, n_way)

        ###############################################################################################
        n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)


        batch_size = 4
        support_size = n_way * n_support

        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)
        #target_rot = y_a_i.repeat(4).view(-1) # (100,)

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:])
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        #rot_img = create_4rotations_images(x_a_i).cuda() # (100, 3, 224, 224)


        #labels_rotation = create_rotations_labels(len(x_a_i), rot_img).cuda()
        #labels_rotation = labels_rotation.view(-1)
        #labels_rotation = labels_rotation.long()
 
         ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().cuda()
        #loss_fn = FocalLoss().cuda()
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

                #zr_batch = rot_img[selected_id]
                #yr_batch = target_rot[selected_id]
                #rr_batch = labels_rotation[selected_id] # rotation labels


                z_batch = x_a_i[selected_id] # 4 images
                zr_batch = create_4rotations_images(z_batch) # 4*4 rotated images

                rr_batch = create_rotations_labels(len(z_batch), zr_batch).cuda() # 4*4 rotation labels
                rr_batch = rr_batch.view(-1)
                rr_batch = rr_batch.long()

                y_batch = y_a_i[selected_id] # (4,)
                yr_batch = y_batch.repeat(4).view(-1) # (4*4,)


                #####################################
                lt = pretrained_model(zr_batch, is_feat = True)[0]

                logits_rot = rotm(lt).cuda()
                logits_rot = logits_rot.view(logits_rot.size(0), -1)
                logits_rot = rotm_class(logits_rot)
                logits_rot = logits_rot.view(-1, 4)  # Assuming there are 4 rotation classes
                logits_rot = logits_rot.float()


                loss_rot = nn.CrossEntropyLoss()(logits_rot, rr_batch)

                output = pretrained_model(zr_batch)
                logit_s = output
                output = classifier(output)

                loss_ce = loss_fn(output, yr_batch)

                #loss = ce_1 + loss_rot

                #####################################

                logit_s = logit_s
                with torch.no_grad():
                     logit_t = teacher(zr_batch)

                loss_div = criterion_div(logit_s, logit_t)

                #####################################

                loss = loss_ce + loss_rot + loss_div

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
               '/workspace/data/ldp-net/checkpoint/JRSNinitRotCrossDistlJRSN.tar')

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
    datamgr             =  miniImageNet_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug=False)
    novel_loaders.append(novel_loader)


    #########################################################################
    for idx, novel_loader in enumerate(novel_loaders):
        print (dataset_names[idx])

        print (freeze_backbone)

        # replace finetine() with your own method
        finetune(novel_loader, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)


