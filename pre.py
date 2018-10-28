import argparse
import os
import torch
import random
import sys
from pretrain.dataloader import itemDataset,ToTensor,collate_fn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

from model.model_builder import build_model_pre #still need some little modify
from model.util.optimizers import build_optim
from model.util.Logger import logger,init_logger  #finish
from model.util.saver import build_model_saver  #finish
from model.util import Report_manager

from model import Trainer
import opts

def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if(not os.path.exists(model_dirname)):
        os.makedirs(model_dirname)

def training_opt_postprocessing(opt):
    if(torch.cuda.is_available() and not opt.gpuid):
        logger.info("you should use gpu to train the model.")
    
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if opt.gpuid:
        torch.cuda.set_device(opt.device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)
    return opt

def main(opt):
    opt = training_opt_postprocessing(opt)
    init_logger(opt)

    if(opt.train_from is not None):
        logger.info('loading checkpoint from {0}'.format(opt.train_from))
        device = torch.device('cpu')
        checkpoint = torch.load(opt.train_from,map_location=device)
        
        model_opt = checkpoint['opt']
    else:
        raise ValueError("you should choose a model to load from in this mode`")
        checkpoint = None
        model_opt = opt
    
    #deal with data input here
    #count for the vocabular size and get the dataset
     
    #build model
    #send the token file into the model set up function
    data_ori = dict()
    for ttype in ['source','target','tag']:
        data_ori[ ttype ] = dict()
        with open('./ch_en/subword.{0}'.format(ttype)) as f_in:
            for j,word in enumerate(f_in):
                data_ori[ttype][word.strip()[1:-1]] = j
    logger.info("source size:{0}".format(len(data_ori['source'])))
    logger.info("target size:{0}".format(len(data_ori['target'])))
    logger.info("tag size:{0}".format(len(data_ori['tag'])))

    data_new = dict()
    for ttype in ['source','target','tag']:
        data_new[ ttype ] = dict()
        with open('./pretrain/subword.{0}'.format(ttype)) as f_in:
            for j,word in enumerate(f_in):
                if(ttype == 'target'):
                    data_new[ttype][word.strip()+'_'] = j
                else:
                    data_new[ttype][word.strip()[1:-1]] = j
                    
    logger.info("source size:{0}".format(len(data_new['source'])))
    logger.info("target size:{0}".format(len(data_new['target'])))
    logger.info("tag size:{0}".format(len(data_new['tag'])))

    logger.info("start build model")
    model = build_model_pre(model_opt,opt,data_ori,data_new,gpu=torch.cuda.is_available(),checkpoint=checkpoint)
    logger.info(model)
        
    logger.info("start build training,validing data")
    train_dataset = itemDataset(file_name='./pretrain/playlist_20181024_train.csv',
                        transform=transforms.Compose([ToTensor()]))
    valid_dataset = itemDataset(file_name='./pretrain/playlist_20181024_valid.csv',
                        transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size ,
                        shuffle=False, num_workers=32,collate_fn=collate_fn)
    validloader = DataLoader(valid_dataset, batch_size=opt.valid_batch_size,
                        shuffle=False, num_workers=32,collate_fn=collate_fn)
  
    logger.info("finish build training,validing data")
  


    logger.info("start build optimizer")
    optim = build_optim(model,opt,None,ttype = 'pretrain')

    #logger.info("model:{0}".format(model))

    #initial model saving place
    _check_save_model_path(opt)

    logger.info("start build reporter")
    report_manager = Report_manager.build_report_manager(opt)

    logger.info("start build save")
    model_saver = build_model_saver(model_opt,opt,model,optim,report_manager)

    logger.info("start build trainer")
    trainer = Trainer.build_trainer(opt,model,optim,report_manager,checkpoint = checkpoint,model_saver=model_saver)

    trainer.train(trainloader,validloader,opt.train_steps,opt.valid_steps,opt.replace)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.add_md_help_argument(parser)
    opts.pre_model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
