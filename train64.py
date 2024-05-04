import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import conf64
import backbone
#import coatnet
from data.datamgr import SimpleDataManager, SetDataManager

from methods.baselinetrainsixtyfour import BaselineTrain

#from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file
from datasets import miniImageNet_few_shot, DTD_few_shot


def train(base_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer )

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    image_size = 224
    optimization = 'Adam'

    if params.method in ['baseline'] :

        if params.dataset == "miniImageNet":

            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 64)
            base_loader = datamgr.get_data_loader(aug = params.train_aug )


        model           = BaselineTrain( model_dict[params.model], params.num_classes)



    model = model.cuda()
    save_dir =  conf64.save_dir

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug_64'

    if not params.method  in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params)
