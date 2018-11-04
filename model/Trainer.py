from .util.Logger import logger
from .util import Loss
from .util import statistics
from .util import distribute
from . import Constant 
import torch
#in this file, we should deal with the dataloader problem

def to_cuda(batch):
    if(torch.cuda.is_available()):
        for name in batch:
            if(isinstance(batch[name],torch.Tensor)):
                batch[name] = batch[name].cuda()
    return batch

def build_trainer(opt,model,optim,report_manager,checkpoint=None,model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #first build the training and valid loss function.
    #read the length from the label file
    with open('./data/subword.target') as f:
        target_dict_len = len(f.readlines())
    
    train_loss = Loss.build_loss_computer(model,target_dict_len,opt)
    valid_loss = Loss.build_loss_computer(model,target_dict_len,opt,train=False)

    trunc_size = opt.truncated_decoder
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = len(opt.gpuid)
    gpu_rank = opt.gpu_rank
    gpu_verbose_level = opt.gpu_verbose_level


    trainer = Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver)
    return trainer

class Trainer(object):
    """
    class that controls the training process

    args:
            model(:py:class:`model.NMTModel`): translation model
                to train
            train_loss(:obj:`loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`loss.LossComputeBase`):
               training loss computation
            optim(:obj:`optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """
    def __init__(self,model,train_loss,valid_loss,optim,
                trunc_size=0,shard_size=32,norm_method='tokens',grad_accum_count=1,
                n_gpu=1,gpu_rank=1,gpu_verbose_level=0,report_manager=None,model_saver=None):
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver

        assert(grad_accum_count>0)
        if(grad_accum_count>1):
            assert(self.trunc_size == 0),\
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""
    
        self.model.train()
    
    def train(self, train_loader, valid_loader, train_steps, valid_steps,relace=False):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('start training...')

        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0

        total_stats = statistics.Statistics()
        report_stats = statistics.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while(step <= train_steps):
            
            self.model.train()

            reduce_counter = 0
            for i,batch in enumerate(train_loader):
                if(torch.cuda.is_available()):
                    batch = to_cuda(batch)
                if(self.n_gpu == 0 or (i %self.n_gpu == self.gpu_rank)):
                    if(self.gpu_verbose_level > 1):
                        logger.info("GpuRank {0}: index :{1} accum: {2}".format(
                            self.gpu_rank,i,accum
                        ))

                    true_batchs.append(batch)

                    if(self.norm_method == 'tokens'):
                        try:
                            num_tokens = batch['target'].ne(Constant.PAD).sum()
                        except:
                            print(batch['target'])
                            num_tokens = batch['target'].ne(Constant.PAD).sum()
                        normalization += num_tokens.item()
                    else:
                        normalization += batch['target'].shape[0]
                    
                    accum += 1
                    if(accum == self.grad_accum_count):
                        reduce_counter += 1
                        if(self.gpu_verbose_level > 0):
                            logger.info("gpurank {0}: reduce_counter: {1} \
                            n_minibatch {2}".format(self.gpu_rank,reduce_counter,
                            len(true_batchs)))
                        
                        if(self.n_gpu > 1):
                            normalization = sum(distribute.all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs,normalization,total_stats,report_stats,relace
                        )

                        report_stats = self._maybe_report_training(
                            step,train_steps,
                            self.optim.learning_rate,report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0

                        #here do the validation step after training a hold epoch
                        if( step % valid_steps == 0):
                            if(self.gpu_verbose_level > 0):
                                logger.info('GpuRank {0}: validate step {1}'.format(
                                    self.gpu_rank,step))
                            #dataloader
                            valid_stats = self.validate(valid_loader,relace)
                            
                            if(self.gpu_verbose_level > 0):
                                logger.info('GpuRank {0}: gather valid stat\
                                step {1}'.format(self.gpu_rank,step))
                            valid_stats = self._maybe_gather_stats(valid_stats)
                            
                            if(self.gpu_verbose_level > 0):
                                logger.info('GpuRank {0}: report stat step {1}'.format(
                                    self.gpu_rank,step))
                            
                            self._report_step(self.optim.learning_rate,step,
                                            valid_stats=valid_stats)
                    
                        if(self.gpu_rank == 0):
                            self._maybe_save(step)
                        step += 1
                        if(step>train_steps):
                            break

            if( self.gpu_verbose_level > 0):
                logger.info('GpuRank %d: we completed an epoch \
                            at step %d' % (self.gpu_rank, step))
        valid_stats = self.validate(valid_loader,replace)
        return total_stats

    def validate(self,valid_loader,replace):
        """
            use the valid set to check if the model works correct.
        """
        self.model.eval()

        stats = statistics.Statistics()
        logger.info("start validation")
        for i,batch in enumerate(valid_loader):
            if(torch.cuda.is_available()):
                batch = to_cuda(batch)

            output, attn, _ = self.model(
                batch['source'],batch['target'],batch['source_len'],replace=replace
                )
            
            batch_stat = self.valid_loss.monolithic_compute_loss(
                batch,output,attn)
            
            stats.update(batch_stat)
            
            #for first five batch print first data for each
            if(i<5):
                logger.info("batch['target'] {0}".format(batch['target'].transpose(0,1)[0]))
                logger.info("output {0}".format(torch.argmax(output.transpose(0,1)[0],dim=-1)))

        self.model.train()

        return stats
    
    def _gradient_accumulation(self,true_batchs,normalization,total_stats,
                                report_stats,replace):
        """
            accumulate the input and perform bp for each input
        """
        if(self.grad_accum_count > 1):
            self.model.zero_grad()
        
        for batch in true_batchs:
            target_size = batch['target'].shape[0]

            if(self.trunc_size):
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size
            
            dec_state = None
            report_stats.n_src_words += batch['source_len'].sum().item()
            
            for j in range(0,target_size-1,trunc_size):
                target = batch['target'][j:j+trunc_size+1]
                
                if(self.grad_accum_count == 1):
                    self.model.zero_grad()
                output, attn, dec_state = self.model(
                    batch['source'],target,batch['source_len'],dec_state,replace
                )

                batch_stat = self.train_loss.sharded_compute_loss(
                    batch, output, attn, j,
                    trunc_size+1, self.shard_size , normalization)

                total_stats.update(batch_stat)
                report_stats.update(batch_stat)

                if(self.grad_accum_count == 1):
                    if(self.n_gpu > 1):
                        grads = [p.grad.data for p in self.model.parameters()
                                    if p.requires_grad and p.grad is not None]
                        
                        distribute.all_reduce_and_rescale_tensors(grads,float(1))   
                    self.optim.step()
                
                if(dec_state is not None):
                    dec_state.detach()
        #not sure why he choose to bp during the percedure of the output

        if(self.grad_accum_count > 1):
            if(self.n_gpu > 1):
                grads = [p.grad.data for p in self.model.parameters()
                            if p.requires_grad and p.grad is not None]
                
                distribute.all_reduce_and_rescale_tensors(grads,float(1))   
            self.optim.step()
    
    def _start_report_manager(self, start_time = None):
        """
        simple function to start report manager(if any)
        """
        if(self.report_manager is not None):
            if(start_time is None):
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self,stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if(stat is not None and self.n_gpu>1):
            return statistics.all_gather_list(stat)
        
        return stat
    
    def _maybe_report_training(self,step,num_steps,learning_rate,
                                report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `.utils.ReportManagerBase.report_training` for doc
        """
        if(self.report_manager is not None):
            return self.report_manager.report_training(
                step,num_steps,learning_rate,report_stats,
                multigpu=self.n_gpu>1
            )
    
    def _report_step(self,learning_rate,step,train_stats=None,
                        valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if(self.report_manager is not None):
            return self.report_manager.report_step(
                learning_rate,step,train_stats=train_stats,
                valid_stats=valid_stats
            )
    def _maybe_save(self,step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)













