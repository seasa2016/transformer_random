import torch
import torch.nn as nn
import numpy as np
from . import Constant
from .module.Embedding import Embedding
from .module.Layer import TransformerEncoderLayer,TransformerDecoderLayer,LayerNorm
from .util import misc
from .util.Logger import logger

class Encoder(nn.Module):
    """
        encoder for the transformer
    """
    def __init__(self,  num_layer ,num_head,
                        model_dim,nin_dim,dropout,embedding):
        
        super(Encoder,self).__init__()

        self.num_layer = num_layer
        
        self.embedding = embedding

        self.self_attention =  nn.ModuleList(
            [TransformerEncoderLayer(model_dim, num_head, nin_dim, dropout)
             for _ in range(self.num_layer)])
    
        self.layernorm = LayerNorm(model_dim)

    def _check_args(self,src,length=None,hidden=None):
        _, n_batch, _ = src.shape

        if(length is not None):
            n_batch_ = length.shape
            misc.aeq(n_batch,n_batch_)

    def forward(self,x,lengths):
        logger.debug('x shpae: {0}'.format(x.shape))
        
        #self._check_args(x,lengths)
        emb = self.embedding(x)

        out = emb.transpose(0,1).contiguous()
        words = x[:,:].transpose(0,1)

        w_batch,w_len = words.shape
        #check for the padding index and set up the mask for the attention weight
        padding_idx = self.embedding.padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)\
                    .expand(w_batch,w_len,w_len)

        total_attn =[]

        for i in range(self.num_layer):
            out,attn = self.self_attention[i](out,mask)
            total_attn.append(attn)
        out = self.layernorm(out)

        return emb,out.transpose(0,1).contiguous(),total_attn

class Decoder(nn.Module):
    """
        decoder for the transformer
    """
    def __init__(self,num_layer,num_head,model_dim,nin_dim,num_word,    
                max_len,copy_attn,self_attn_type,dropout,embedding):
        super(Decoder,self).__init__()
    
        self.decoder_type = 'transformer'
        self.num_layer = num_layer
        self.embedding = embedding
        self.self_attn_type = self_attn_type

        self.transformer_layer = nn.ModuleList([
            TransformerDecoderLayer(model_dim,num_head,nin_dim,dropout,max_len,
            self_attn_type=self_attn_type) for _ in range(num_layer)
        ])

        self._copy = False
        if(copy_attn):
            pass
            self._copy = True
        self.layer_norm = LayerNorm(model_dim)
        self.linear = nn.Linear(model_dim,num_word)

    def forward(self,tar,memory_bank,state,memory_length=None,
                step=None,cache=None):
        
        src = state.src
        src_words = src[:,:].transpose(0,1)
        tar_words = tar[:,:].transpose(0,1)
        src_batch,src_len = src_words.shape
        tar_batch,tar_len = tar_words.shape

        outputs = []
        attns = {"std":[]}
        if(self._copy):
            attns["copy"] = []

        emb = self.embedding(tar,pos=step)
        assert(emb.dim()==3)

        output = emb.transpose(0,1).contiguous()
        src_memory_bank = memory_bank.transpose(0,1).contiguous()

        padding_idx = self.embedding.padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1)\
                        .expand(src_batch,tar_len,src_len)
        tar_pad_mask = tar_words.data.eq(padding_idx).unsqueeze(1)\
                        .expand(tar_batch,tar_len,tar_len)
        
        if(state.cache is None):
            saved_inputs = []

        for i in range(self.num_layer):
            #print("layer",i)
            prev_layer_input = None

            if(state.cache is None):
                if(state.previous_input is not None):
                    prev_layer_input = state.previous_layer_inputs[i]
            # forward(self,inputs,memory_bank,src_pad_mask,tar_pad_mask,
            #        previous_input=None,layer_cache=None,step=None)
            
            output,attn,all_input \
                    = self.transformer_layer[i](
                        output,src_memory_bank,
                        src_pad_mask,tar_pad_mask,
                        previous_input=prev_layer_input,
                        layer_cache = state.cache["layer_{}".format(i)]
                        if(state.cache is not None) else None,
                        step=step
                    )
            if(state.cache is None):
                saved_inputs.append(all_input)

        if(state.cache is None):
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        output = output.transpose(0,1).contiguous()
        attn = attn.transpose(0,1).contiguous()

        attns["std"] = attn
        if(self._copy):
            attns["copy"] = attn
        
        if(state.cache is None):
            state = state.update_state(tar,saved_inputs)
        
        #I move the linear part to here.
        output = self.linear(output)
        return output,state,attns
    
    def init_decoder_state(self,src,memory_bank,enc_hidden,
                            with_cache=False):
        
        state = TransformerDecoderState(src)
        if(with_cache):
            state._init_cache(memory_bank,self.num_layer,self.self_attn_type)
        return state

class TransformerDecoderState(object):
    def __init__(self,src):
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None
    
    #property
    def _all(self):
        #with attribute to update self.beam_update()
        if(self.previous_input is not None and self.previous_layer_inputs is not None):
            return (self.previous_input,self.previous_layer_inputs.self.src)
        else:
            return (self.src)

    def detach(self):
        if(self.previous_input is not None):
            self.previous_input = self.previous_input.detach()
        if(self.previous_layer_inputs is not None):
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        
        self.src = self.src.detach()
    
    def update_state(self,new_input,previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        
        return state

    def beam_update(self,idx,positions,beam_size):
        for e in self._all:
            sizes = e.shape
            br = sizes[1]
            if(len(sizes) == 3):
                sent_states = e.view(sizes[0],beam_size,br // beam_size,
                                        sizes[2])[:,:,idx]

            else:
                sent_states = e.view(sizes[0],beam_size,br // beam_size,
                                        sizes[2],sizes[3])[:,:,idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1,positions))
    
    def _init_cache(self,memory_bank,num_layer,self_attn_type):
        self.cache = {}
        batch_size = memory_bank.shape[1]
        #this is for the vetor length
        depth = memory_bank.shape[-1]

        for l in range(num_layer):
            layer_cache = {
                "memory_key": None,
                "memory_value": None
            }
            if(self_attn_type == "scaled-dot"):
                layer_cache["self_key"] = None
                layer_cache["self_value"] = None

            elif(self_attn_type == "average"):
                layer_cache["prev_g"] = torch.zeros((batch_size,1,depth))
                
            else:
                layer_cache["self_key"] = None
                layer_cache["self_value"] = None
            
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self,beam_size):
        #repeat beam size times along batch dimension
        self.src = self.src.data.repeat(1,beam_size,1)
    
    def map_batch_fn(self,fn):
        def _recursive_map(struct,batch_dim=0):
            for k,v in struct.items():
                if(v is not None):
                    if(isinstance(v,dict)):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v,batch_dim)

        self.src = fn(self.src,1)
        if(self.cache is not None):
            _recursive_map(self.cache)

class Transformer(nn.Module):
    def __init__(self,encoder,decoder,multi_gpu=False):
        super(Transformer,self).__init__()
        self.multi_gpu = multi_gpu
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self,src,tar,length,dec_state=None):
        #leave last one for predict usually it is eos or padding
        tar = tar[:-1]

        enc_final,memory_bank,enc_attn = self.encoder(src,length)
        enc_state = self.decoder.init_decoder_state(src,memory_bank,enc_final)

        decoder_outputs,dec_state,attns = self.decoder(
            tar,memory_bank,
            enc_state if dec_state is None else dec_state,
            memory_length = length
        )

        return decoder_outputs, attns, dec_state










