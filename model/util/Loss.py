import torch
import torch.nn as nn 
import torch.nn.functional as F
import model.Constant as Constant
from . import statistics
from .Logger import logger
import numpy as np

def build_loss_computer(model,tgt_dict_size,opt,train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compute = NMTLossCompute(
        model,tgt_dict_size,label_smoothing = opt.label_smoothing)
    
    compute.to(device)

    return compute

class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self,model,tgt_dict):
        super(LossComputeBase,self).__init__()
        self.model = model
        self.tgt_dict = tgt_dict
        self.padding_idx = Constant.PAD
    
    #follow up are just copy from the opennmt repos
    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def _compute_corr(self, output, origin):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self,batch,output,attns):
        """
        Compute the forward loss for the batch.

        Args:
            tgt (batch): batch of labeled examples
            output (:obj:`FloatTensor`):
                output of decoder model `[tgt_len x batch x hidden]`
            attns (dict of :obj:`FloatTensor`) :
                dictionary of attention distributions
                `[tgt_len x batch x src_len]`
        Returns:
            obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0,batch['target'].shape[0])
        shard_state = self._make_shard_state(batch['target'],output,range_,attns)
        
        loss = self._compute_loss(**shard_state,shard_size=batch['target'].shape[0],batch=batch)
        num_non_padding,num_correct = self._compute_corr(output,batch['origin'])

        batch_stats = statistics.Statistics(loss.item(),num_non_padding,num_correct)
        return batch_stats

    def sharded_compute_loss(self,batch,output,attns,
                            cur_trunc,trunc_size,shard_size,normalization):
        """
        Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        
        range_ = (cur_trunc,cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch['target'],output,range_,attns)
        
        loss_total = 0
        for i,shard in enumerate(shards(shard_state,shard_size)):
            loss = self._compute_loss(**shard,shard_size=shard_size,batch=batch,now=i)
            
            #compute the gradient
            loss.div(float(normalization)).backward()

            loss_total += loss.item()
            #batch_stats.update(stats)
        
        num_non_padding,corr = self._compute_corr(output,batch['origin'])

        batch_stats = statistics.Statistics(loss_total,num_non_padding,corr)

        return batch_stats

    def _compute_corr(self, output, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        #this return two value. one for value and the other for position
        pred = output.max(-1)[1]
        non_padding = target.ne(self.padding_idx)
        
        num_correct = 0
		
        #here we use set rathe than compare correct at same place
        #iterate over the betch size
        for i in range(output.shape[1]):
            origin = set( np.array(target[i]))
            ans = set( np.array( pred[:,i]))
			
            num_correct += len(origin & ans)
        num_non_padding = non_padding.sum().item()

        return num_non_padding,num_correct
        #return statistics.Statistics(loss.item(),num_non_padding,num_correct)

    def _bottle(self,_v):
        return _v.view(-1,_v.shape[-1])

    def _unbottle(self,_v,batch_size):
        return _v.view(-1,batch_size,_v.shape[1])

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self,label_smoothing,tgt_dict_size,padding_idx=-100):
        assert(0.0 <= label_smoothing <= 1.0)
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss,self).__init__()

        #in the paper the setting should be (num_class - 1)\
        #however here we meet that the padding problem. therefore 2
        smoothing_value = label_smoothing / (tgt_dict_size - 2)
        one_hot = torch.full((tgt_dict_size,),smoothing_value)
        one_hot = torch.full((tgt_dict_size,),0)
        
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot',one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self,output,target,shard_size,batch,part,now):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.shape[0],1)
		#print("origin",batch['origin'].shape)
		#print("target",batch['target'].shape)

        for i in range(model_prob.shape[0]):
            temp = now*shard_size+i//part
            
            #for the last put eos
            if( temp >= batch['target_len'][i%part]-2 ):
                continue
            #else put available ans
            else:  
				#print( temp , batch['target_len'][i%part]-2 )
                model_prob[i][ batch['origin'][i%part][temp:] ] = self.confidence
                model_prob[i][0] = 0
                
				#print("len",i,part ,batch['target_len'][i%part], -temp-2 )
                
                model_prob[i] /= ( batch['target_len'][i%part].float() -temp-2 )
        
        model_prob.scatter_(1,target.unsqueeze(1),self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1),0)

        output = F.log_softmax(output,dim=-1)
        loss = -torch.sum(output*model_prob)
        return loss

class NMTLossCompute(LossComputeBase):
    """
    Standard NMT loss computation
    """
    def __init__(self,model,tgt_dict_size,normalization='sents',
                    label_smoothing=0.0):
        super(NMTLossCompute,self).__init__(model,tgt_dict_size)
        
        if(label_smoothing >=-1):
            self.criterion = LabelSmoothingLoss(
                label_smoothing,tgt_dict_size,padding_idx=self.padding_idx)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx,
                                        size_average=False)

    def _make_shard_state(self,target,output,range_,attns=None):
        #logger.debug("range:{0} {1}".format(range_[0],range_[1]))
        #logger.debug("target:{0}".format(target))
        return {
            "output" : output,
            "target"  : target[ range_[0]+1 : range_[1] ]
        }
    
    def _compute_loss(self,target,output,shard_size=32,batch=None,now=0):
        #here i should deal with the dimension problem

        part = output.shape[1]

        bottled_output = self._bottle(output)
        truth = target.view(-1)
        
		#if(origin is None):
		#loss = self.criterion(bottled_output,truth)
		#else:
        loss = self.criterion(bottled_output,truth,shard_size,batch,part,now)
		#stats = self._stats(loss.clone(),bottled_output,truth)

        return loss
    



def filter_shard_state(state,shard_size=None):
    for k,v in state.items():
        if(shard_size is None):
            yield k,v
    
        if v is not None:
            v_split = []
            if(isinstance(v,torch.Tensor)):
                for v_chunk in torch.split(v,shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k,(v,v_split)

def shards(state,shard_size,eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if(eval_only):
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state,shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)















