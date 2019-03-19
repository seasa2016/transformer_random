import torch
import torch.nn as nn
import numpy as np

from .Norm import LayerNorm
from .Attention  import MultiHeadedAttention 


class Network_In_Network(nn.Module):
    def __init__(self,model_dim,nin_dim,dropout):
        super(Network_In_Network,self).__init__()

        self.w_1 = nn.Linear(model_dim,nin_dim)
        self.w_2 = nn.Linear(nin_dim,model_dim)

        self.layer_norm = LayerNorm(model_dim)
        
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        #x.shape
        #[batch_size,data_len,data_dim]
        output = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(output))
        
        return output+x


class TransformerEncoderLayer(nn.Module):
    def __init__(self,model_dim,num_head,nin_dim,dropout):
        super(TransformerEncoderLayer,self).__init__()
        
        self.layer_norm = LayerNorm(model_dim)
        
        self.self_attn = MultiHeadedAttention(model_dim,num_head,dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.feed_forward = Network_In_Network(model_dim,nin_dim,dropout)
                
    def forward(self,x,mask):
        x_norm = self.layer_norm(x)
        context,attn = self.self_attn(x_norm,x_norm,x_norm,mask=mask)

        output = self.dropout(context)+x

        return self.feed_forward(output),attn.detach()

class TransformerDecoderLayer(nn.Module):
    def __init__(self,model_dim,num_head,nin_dim,
                    dropout,max_len,self_attn_type="scaled_dot"):
        super(TransformerDecoderLayer,self).__init__()
        
        self.self_attn_type = self_attn_type
        #self.self_attn = MultiHeadedAttention(num_head,model_dim,dropout)

        if(self_attn_type == "scaled_dot"):
            self.self_attn = MultiHeadedAttention(model_dim,num_head,dropout)
        elif(self_attn_type == "average"):
            #self.self_attn = Attention.AverageAttention(model_dim,dropout)
            pass
        
        self.context_attn = MultiHeadedAttention(model_dim,num_head,dropout=dropout)
        self.feed_forward = Network_In_Network(model_dim,nin_dim,dropout)

        self.layer_norm_1 = LayerNorm(model_dim)
        self.layer_norm_2 = LayerNorm(model_dim)

        self.dropout_p = dropout
        self.dropout = nn.Dropout(self.dropout_p)

        mask = self._get_attn_subsequent_mask(max_len)

        self.register_buffer('mask',mask)

    def forward(self,inputs,memory_bank,src_pad_mask,tar_pad_mask,
                    previous_input=None,layer_cache=None,step=None):

        dec_mask = torch.gt(tar_pad_mask + 
                    self.mask[:,:tar_pad_mask.shape[1],:tar_pad_mask.shape[1]],0)
        
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm

        if(previous_input is not None):
            all_input = torch.cat([previous_input,input_norm],dim=1)
            dec_mask = None

        if(self.self_attn_type == "scaled_dot"):
            query, attn = self.self_attn(all_input,all_input,input_norm,
                                        mask = dec_mask,
                                        layer_cache=layer_cache,
                                        attn_type="self")
        
        elif(self.self_attn_type == "average"):
            query, attn = self.self_attn(input_norm,mask=dec_mask,
                                        layer_cache=layer_cache,step=step)
        else:
            raise ValueError('should use one of the attn type')

        query = self.dropout(query) + input_norm
        query_norm = self.layer_norm_2(query)

        mid, attn = self.context_attn(memory_bank,memory_bank,query_norm,
                                        mask=src_pad_mask,
                                        layer_cache=layer_cache,
                                        attn_type="context")

        output = self.feed_forward(self.dropout(mid)+query_norm)

        return output,attn,all_input
    
    def _get_attn_subsequent_mask(self,size):
        """
        Get an attention mask to avoid using the subsequent info.
        Args:
            size: int
        Returns:
            (`LongTensor`):
            * subsequent_mask `[1 x size x size]`
        """

        attn_shape = (1,size,size)

        subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        
        return subsequent_mask
