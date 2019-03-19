import torch
import torch.nn as nn
import math


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    """
    def __init__(self,model_dim,num_head,dropout):
        super(MultiHeadedAttention,self).__init__()
        assert( model_dim % num_head == 0)

        self.head_dim = model_dim // num_head
        self.model_dim = model_dim

        self.num_head = num_head

        self.linear_key = nn.Linear(model_dim,model_dim)
        self.linear_value = nn.Linear(model_dim,model_dim)
        self.linear_query = nn.Linear(model_dim,model_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.final_linear = nn.Linear(model_dim,model_dim)


    def forward(self,key,value,query,mask=None,layer_cache=None,attn_type=None):
        #print("0 query {0}".format(query.shape))
        #print("0 key {0}".format(key.shape))
        #print("0 layer_cache",layer_cache)
        batch_size = key.shape[0]
        key_len = key.shape[1]
        query_len = query.shape[1]

        head_dim = self.head_dim
        num_head = self.num_head

        def shape(x):
            return x.view(batch_size,-1,num_head,head_dim).transpose(1,2)
        
        def unshape(x):
            return x.transpose(1,2).contiguous()\
                    .view(batch_size,-1,num_head*head_dim)
        
        if(layer_cache is not None):
            if(attn_type == 'self' ):
                query,key,value =   self.linear_query(query),\
                                    self.linear_key(query),\
                                    self.linear_value(query)
                key = shape(key)
                value = shape(value)

                if(layer_cache is not None):
                    device = key.device

                    if(layer_cache['self_key'] is not None):
                        key = torch.cat((layer_cache['self_key'].to(device),key),dim=2)
                    
                    if(layer_cache['self_value'] is not None):
                        value = torch.cat((layer_cache['self_value'].to(device),value),dim=2)

                    layer_cache['self_key'] = key
                    layer_cache['self_value'] = value
           		
				
            elif(attn_type == 'context'):                
                query = self.linear_query(query)

                if(layer_cache is not None):
                    if(layer_cache['memory_key'] is None):
                        key,value = self.linear_key(key),self.linear_value(value)

                        key = shape(key)
                        value = shape(value)
                    else:
                        key,value = layer_cache['memory_key'],\
                                    layer_cache['memory_value']
                    
                    layer_cache['memory_key'] = key
                    layer_cache['memory_value'] = value
                
                else:
                    key,value = self.linear_key(key),self.linear_value(value)
            else:
                raise ValueError("no this {0} arrn type".format(attn_type))
        else:
            key = self.linear_key(key)
            value = self.linear_value(value)
            query = self.linear_value(query)

            key = shape(key)
            value = shape(value)
        
        query = shape(query)

        key_len = key.shape[2]
        query_len = query.shape[2]

        #calculate the score
        query = query / math.sqrt(self.head_dim)
        
        #print("query {0}".format(query.shape))
        #print("key {0}".format(key.shape))
        score = torch.matmul(query,key.transpose(2,3).contiguous())

        if(mask is not None):
            mask = mask.unsqueeze(1).expand_as(score)
            score = score.masked_fill(mask,-1e18)

        attn = self.softmax(score)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn,value) )

        output = self.final_linear(context)

        top_attn = attn.view(batch_size,num_head,query_len,key_len).contiguous()
            
        return output,top_attn

