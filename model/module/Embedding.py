import torch
import torch.nn as nn
import math
from ..util.Logger import logger

class Embedding(nn.Module):
    def __init__(self,num_word,max_len,emb_dim,feature_dim,padding_idx,dropout=0,dtype='sum'):
        super(Embedding,self).__init__()

        self.dtype = dtype
        self.padding_idx = padding_idx

        self.dim = emb_dim

        self.word_emb = nn.Embedding(num_word,emb_dim,padding_idx=padding_idx)
        self.word_emb.weight = nn.Parameter(self.emb_init(self.word_emb.weight.shape))
        

        if(self.dtype=='sum'):
            self.pos_emb = nn.Embedding(max_len,emb_dim,)
            self.pos_emb.weight = nn.Parameter(self.pos_init(self.pos_emb.weight.shape))
            self.pos_emb.weight.requires_grad = False
        elif(self.dtype=='cat'):
            self.pos_emb = nn.Embedding(max_len,feature_dim)
            self.pos_emb.weight = nn.Parameter(self.pos_init(self.pos_emb.weight.shape))
            self.pos_emb.weight.requires_grad = False
        else:
            self.pos_emb = None
        self.drop = nn.Dropout(p=dropout)

    def emb_init(self,shape):
        temp = torch.randn(shape)
        temp.data[0] = 0
        return temp 
    def pos_init(self,shape):
        pe = torch.zeros(shape)
        position = torch.arange(0, shape[0]).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, shape[1], 2,dtype=torch.float) *(-(math.log(10000.0,math.e) / shape[1]))).float())

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term) 

        return pe
    
    def forward(self,x_val,pos=None):
        #logger.debug('x: {0}'.format(x_val))
        output = self.word_emb(x_val)
        output = output * math.sqrt(self.dim)
        if(self.pos_emb is not None):
            if(pos is None):
                pos_out = self.pos_emb.weight[:output.shape[0]].unsqueeze(1)
            else:#here the method is for sequential output
                pos_out = self.pos_emb.weight[pos].unsqueeze(0).unsqueeze(0)
            if(self.dtype=='sum'):
                output += pos_out
            elif(self.dtype=='cat'):
                output = torch.cat([output,pos_out.expand_as(output)],dim=-1)
        output = self.drop(output)
        
        
        return output

#follow up are the testing file
def main():
    model = Embedding(num_word=6,max_len=5,emb_dim=6,feature_dim=6,dropout=0,dtype='sum',padding_idx=0)


if(__name__ == '__main__'):
    main()
