import torch
import torch.nn as nn
import torch.optim as Optim
from torch.nn.utils import clip_grad_norm_
from .Logger import logger 
import math

def build_optim(model,opt,checkpoint=None,ttype=None):
    """"
    Build optimizer
    """
    saved_optimizer_state_dict = None
    #i am not sure why they dont save the state dict at checkpoint but the
    #hold model

    if(checkpoint):
        #load the optim from previous checkpoint
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            opt.optim_method,
            opt.learning_rate,
            opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_steps=opt.start_decay_step,
            decay_steps=opt.decay_steps,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            decay_method = opt.decay_method,
            warmup_steps = opt.warmup_steps,
            train_steps = opt.train_steps,
            model_size = opt.model_dim
        )

    # Stage 1:
    # Essentially optim.set_parameters (re-)creates and optimizer using
    # model.paramters() as parameters that will be stored in the
    # optim.optimizer.param_groups field of the torch optimizer class.
    # Importantly, this method does not yet load the optimizer state, as
    # essentially it builds a new optimizer with empty optimizer state and
    # parameters from the model.
    if(ttype == 'pretrain'):
        params = []    
        logger.info('*'*10)
        logger.info('things will be optim')
        
        if(opt.replace):
            for name,p in model.mid.named_parameters():
                if p.requires_grad:
                    logger.info('{0}'.format(name))
                    params.append(p)

        for name,p in model.decoder.named_parameters():
            if p.requires_grad:
                logger.info('{0}'.format(name))
                params.append(p)
        logger.info('part of the embedding')
        params.append(model.encoder.embedding.word_emb.weight[2:11])
        logger.info('*'*10)
    else:
        params = []    
        logger.info('*'*10)
        logger.info('things will be optim')
        for name,p in model.named_parameters():
            if p.requires_grad:
                logger.info('{0}'.format(name))
                params.append(p)
        logger.info('*'*10)

    optim.set_optim_type(params,ttype)

    if(checkpoint):
        # Stage 2: In this stage, which is only performed when loading an
        # optimizer from a checkpoint, we load the saved_optimizer_state_dict
        # into the re-created optimizer, to set the optim.optimizer.state
        # field, which was previously empty. For this, we use the optimizer
        # state saved in the "saved_optimizer_state_dict" variable for
        # this purpose.
        # See also: https://github.com/pytorch/pytorch/issues/2830
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
         
        if(torch.cuda.is_available()):
            for state in optim.optimizer.state.values():
                for k,v in state.items():
                    if(torch.is_tensor(v)):
                        state[k] = v.cuda()
        
        # We want to make sure that indeed we have a non-empty optimizer state
        # when we loaded an existing model. This should be at least the case
        # for Adam, which saves "exp_avg" and "exp_avg_sq" state
        # (Exponential moving average of gradient and squared gradient values)
        if (optim.optim_method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim



class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.

    Args:
      method (:obj:`str`): one of [sgd, adagrad, adadelta, adam]
      lr (float): learning rate
      lr_decay (float, optional): learning rate decay multiplier
      start_decay_steps (int, optional): step to start learning rate decay
      beta1, beta2 (float, optional): parameters for adam
      adagrad_accum (float, optional): initialization parameter for adagrad
      decay_method (str, option): custom decay options
      warmup_steps (int, option): parameter for `noam` decay
      model_size (int, option): parameter for `noam` decay

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well
    """

    def __init__(self,optim_method,learning_rate,max_grad_norm,
                lr_decay=1,start_decay_steps=None,decay_steps=None,
                beta1=0.9,beta2=0.999,
                adagrad_accum=0.0,
                decay_method = None,
                warmup_steps = 4000,train_steps = 30000,
                model_size=None):
        
        self.learning_rate = learning_rate
        self.original_lr = learning_rate*10
        self.max_grad_norm = max_grad_norm
        
        self.optim_method = optim_method
        self.decay_method = decay_method.strip().split('*')

        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.train_steps = train_steps
        self.warmup_steps = warmup_steps
        self.start_decay = False

        self._step = 0
        
        self.betas = [beta1,beta2]
        self.adagrad_accum = adagrad_accum

        #this should be model_dim
        self.model_size = model_size
        
    def set_optim_type(self,params,ttype=None):
        """
        set up the optimizer type and the correspond coefficient
        """

        self.params = params

        if(self.optim_method.lower() == 'sgd'):
            self.optimizer = Optim.SGD(self.params,lr=self.learning_rate)
        elif(self.optim_method.lower() == 'adagrad'):
            self.optimizer = Optim.Adagrad(self.params,lr=self.learning_rate)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.optimizer.state[p]['sum'] = self.optimizer.state[p]['sum'].\
                    fill_(self.adagrad_accum)
                    #this is th initial setting of the accumulate value

        elif(self.optim_method.lower() == 'adadelta'):
            self.optimizer = Optim.Adadelta(self.params,lr=self.learning_rate)
        elif(self.optim_method.lower() == 'adam'):
            self.optimizer = Optim.Adam(self.params,lr=self.learning_rate,
                                        betas=self.betas,eps = 1e-9)
        else:
            raise RuntimeError("invalid optim method: {0}".format(self.optim_method))
        
    def _set_rate(self,learning_rate):
        """
        update the model learning rate
        """
        self.learning_rate = learning_rate

        self.optimizer.param_groups[0]['lr'] = self.learning_rate

    def step(self):
        """
        update the learning according to the method it used
        tipically copy from tensor2tensor ToT
        """

        self._step += 1
        def method_rate(name):
            """
            here the method can be check out here:https://zhuanlan.zhihu.com/p/32923584
            """
            if(name == 'constant'):
                #logger.info("base learning rate:{0}".format(self.original_lr ))
                return self.original_lr 
            
            elif(name == 'linear_warmup'):
                return min(1.0 , self._step / self.warmup_steps)
            
            elif(name == 'linear_decay'):
                ret = (self.train_steps - self._step) / self.decay_steps
                return min(1.0,max(0.0,ret))
            
            elif(name == 'rsqrt_decay'):
                return 1 / math.sqrt(max(self._step,self.warmup_steps))
            
            elif(name == 'rsqrt_normalized_decay'):
                scale = math.sqrt(1.0*self.warmup_steps)
                return scale / math.sqrt(max(self._step,self.warmup_steps))
            
            elif(name == 'exp_decay'):
                p = max(0.0,(self._step - self.warmup_steps) / self.decay_steps)
                return pow(self.lr_decay,p)
                #here has the setting that make the learning to be the int
            elif(name == 'rsqrt_hidden_size'):
                return self.model_size ** (-0.5)
            elif(name == 'noam'):
                #here perform the noam method
                return self.original_lr*(self.model_size**(-0.5) * \
                        min(self._step**(-0.5),self._step*self.warmup_steps**(-1.5)))
            else:
                #this should be the problem that the method not implement   
                raise ValueError("unknown learning rate factor %s" % name)
        
        self.learning_rate = 1

        for name in self.decay_method:
            self.learning_rate *= method_rate(name)

        self._set_rate(self.learning_rate)

        #here perform the gradient clipping
        if(self.max_grad_norm):
            clip_grad_norm_(self.params,self.max_grad_norm)

        self.optimizer.step()

      












