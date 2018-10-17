import os
import torch
import torch.nn as nn

from collections import deque

from .Logger import logger

def build_model_saver(model_opt, opt, model, optim,reporter):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             optim,
                             reporter,
                             opt.save_checkpoint_steps,
                             opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    """
        Base class for model saving operations
        Inherited classes must implement private methods:
            * `_save`
            * `_rm_checkpoint
    """
    def __init__(self,save_checkpoint_steps,keep_checkpoint=-1):
        self.save_checkpoint_steps = save_checkpoint_steps
        self.keep_checkpoint = keep_checkpoint

        if(keep_checkpoint>0):
            self.checkpoint_queue = deque([],maxlen=keep_checkpoint)
    
    def maybe_save(self,step):
        """
        Main entry point for model saver
        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """
        if(self.keep_checkpoint == 0):
            return 
        if(step % self.save_checkpoint_steps != 0):
            return 

        chkpt,chkpt_name = self._save(step)

        if(self.keep_checkpoint > 0):
            if(len(self.checkpoint_queue) == self.checkpoint_queue.maxlen):
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)
    
    def _save(self,step):
        """
        Save the checkpoint.

        args:
            step(int) : step num
        returns:
            checkpoint:the saved object
            checkpoint name: name of the chkpt file 
        """

        raise NotImplementedError()
    
    def _rm_checkpoint(self,name):
        """
        remove the checkpoint

        args:
            name: name og the identity checlpoint.
        """
        raise NotImplementedError()

class ModelSaver(ModelSaverBase):
    """
        simple model saver to filesystem
    """

    def __init__(self,base_path,model,model_opt,optim,reporter,
                save_checkpoint_steps,keep_checkpoint=0):
        super(ModelSaver,self).__init__(save_checkpoint_steps,keep_checkpoint)
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.optim = optim
        self.reporter = reporter
    
    def _save(self,step):
        real_model = (self.model.module if(isinstance(self.model,nn.DataParallel))
                                        else self.model)

        model_state_dict = real_model.state_dict()

        #check for dataset_setting
        checkpoint = {
            "model" : model_state_dict,
            "opt":self.model_opt,
            'optim':self.optim,
            "reporter":self.reporter.data()
        }
        
        checkpoint_path = "{0}/step_{1}.pt".format(self.base_path,step)
        torch.save(checkpoint,checkpoint_path)
        logger.info("saving checkpoint to {0}".format(checkpoint_path))
        
        return checkpoint,checkpoint_path
    
    def _rm_checkpoint(self,name):
        os.remove(name)





if(__name__ == '__main__'):
    print('test')












