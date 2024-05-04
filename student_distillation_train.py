import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.dist_baselinetrain import BaselineTrain
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file  
from datasets import miniImageNet_few_shot, DTD_few_shot

def train(base_loader, model, optimization, start_epoch, stop_epoch, params, teacher_model=None, distillation_weight=0.5):  # Add arguments for knowledge distillation
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')     

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, optimizer, teacher_model, distillation_weight)  # Pass teacher_model and distillation_weight

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        
    return model

if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')

    image_size = 224
    optimization = 'Adam'

    if params.method in ['baseline']:

        if params.dataset == "miniImageNet":
        
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size=16)
            base_loader = datamgr.get_data_loader(aug=params.train_aug)

        # ... (other dataset cases)

        model = BaselineTrain(model_dict[params.model], params.num_classes)



    model = model.cuda()
    save_dir = configs.save_dir

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug_distillation'

    if not params.method in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    # Load the pre-trained teacher model (399.tar)
    teacher_checkpoint = torch.load('/workspace/data/LDP-Net-PM/CDFSLbenchmark/checkpoints/miniImageNet/ResNet10_baseline_aug/399.tar')
    teacher_model = backbone.ResNet10()
    #teacher_model.load_state_dict(teacher_checkpoint['state'])


    # Example of modifying keys
    state_dict = teacher_checkpoint['state']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('feature.', '') 
        if 'classifier' not in new_key:  # Skip 'classifier' keys
            new_state_dict[new_key] = value
    teacher_model.load_state_dict(new_state_dict)
    #teacher_model = teacher_model.cuda()
    teacher_model.eval()  # Set the teacher model to evaluation mode

    # Train the student model with knowledge distillation
    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params, teacher_model=teacher_model, distillation_weight=0.5)
