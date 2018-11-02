import torch
import torch.nn as nn
import math
from ..util.Logger import logger

class Embedding(nn.Module):
    def __init__(self,num_word,max_len,emb_dim,feature_dim,padding_idx,dropout=0,dtype='sum',tag=None):
        """
        mixual embedding for the language model encoder
        """
        super(Embedding,self).__init__()

        self.dtype = dtype
        self.tag = tag
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

        if(self.tag):
            self.tag_emb = nn.Embedding(self.tag,emb_dim)
            print("embedding with shape",self.tag_emb.weight.shape)
            self.tag_emb.weight = nn.Parameter(self.tag_init(self.tag_emb.weight.shape))


        self.drop_sym = nn.Dropout(p=dropout)
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
    def tag_init(self,shape):
        temp = torch.randn(shape)
        return temp 

    def forward(self,x,pos=None,tag=False):
        #take out the first word and extract the embedding
        
        if(tag):
            x_tag = x[0]
            x_val = x[1:]


            eword = self.word_emb(x_val)
            etag = self.tag_emb(x_tag).view(1,-1,eword.shape[-1])
            output = torch.cat((etag,eword),dim=0)
        else:
            output = self.word_emb(x)


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
