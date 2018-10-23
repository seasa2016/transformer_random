import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from .module.Embedding import Embedding
from .util.Logger import logger
from . import Constant
from . import transformer



def build_embedding(opt,word_dict,max_len,for_encoder=True,dtype='sum'):
    if(for_encoder):
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tar_word_vec_size
    
    #print(Constant.PAD_token)
    
    word_padding_idx = word_dict[Constant.PAD_token]
    num_word_embedding = len(word_dict)

    # num_word,max_len,emb_dim,feature_dim,dropout=0,dtype='sum'
    return Embedding(num_word= num_word_embedding,
                    max_len = max_len,
                    emb_dim = embedding_dim,
                    feature_dim = embedding_dim,
                    padding_idx = word_padding_idx,
                    dropout = opt.dropout,
                    dtype = dtype)

def build_encoder(opt,src_dict):
    """
    num_layer ,num_head,
    model_dim,nin_dim,dropout,embedding):
    """

    max_len = 128
    src_embedding = build_embedding(opt,src_dict,max_len)
    return transformer.Encoder( opt.num_layer,opt.num_head,
                                opt.model_dim,opt.nin_dim,
                                opt.dropout,src_embedding)

def build_decoder(opt,tar_dict):
    """
    num_layer,model_dim,num_head,nin_dim,
    copy_attn,self_attn_type,dropout,embedding
    """

    max_len = 128
    tar_embedding = build_embedding(opt,tar_dict,max_len,for_encoder=False,dtype='none')
    return transformer.Decoder(
        opt.num_layer,opt.num_head,
        opt.model_dim,opt.nin_dim,len(tar_dict),max_len,
        opt.copy_attn,opt.self_attn_type,opt.dropout,tar_embedding
    )

def load_test_model(opt,model_path=None):
    """
    use the method the acquire the data_dict and the model
    """
    if model_path is None:
        if(opt.test_from is None):
            raise ValueError('test_from shouble not be None')
        model_path = opt.test_from
    
    checkpoint = torch.load(model_path)

    data_token = dict()
    for t in ['source','target']:
        data_token[t] = dict()

        with open('./data/subword.{0}'.format(t)) as f_in:
            for i,word in enumerate(f_in):
                data_token[t][word.strip()[1:-1]] = i

    model = build_base_model(opt, data_token, torch.cuda.is_available(),checkpoint)
    model.eval()
    
    return model, opt

def build_base_model(model_opt,data_token,gpu,checkpoint=None):

    #in our work,we only use text
    
    #build encoder
    encoder = build_encoder(model_opt,data_token['source'])
    logger.info("finish build encoder")
    decoder = build_decoder(model_opt,data_token['target'])
    logger.info("finish build decoder")

    device = torch.device("cuda" if gpu else "cpu")
    model = transformer.Transformer(encoder,decoder)
    #print(model)
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print("the size will be {0} {1} {2}".format(n_params,enc,dec))
    if(checkpoint is not None):
        logger.info('loading model weight from checkpoint')
        model.load_state_dict(checkpoint['model'])
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
    
    model.to(device)
    logger.info('the model is now in the {0} mode'.format(device))
    return model

def build_model(opt,data_token,checkpoint):
    logger.info('Building model...')
    model = build_base_model(opt,data_token,torch.cuda.is_available(),checkpoint)

    return model
