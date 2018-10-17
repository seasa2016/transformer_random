from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import pandas as pd
import sys

#add sos bos
class itemDataset(Dataset):
    def __init__(self, file_source,file_target=None,transform=None):
        
        self.data=[]
        if(file_target is not None):
            self.test_mode = False
            with open(file_source) as f_1:
                with open(file_target) as f_2:
                    for i,(l1,l2) in enumerate(zip(f_1,f_2)):
                        l1 = [ int(_) for _ in l1.strip().split()] + [1]
                        l2 = [2] + [ int(_) for _ in l2.strip().split()] + [1]
                        
                        self.data.append({'source':l1,'target':l2,'source_len':[ len(l1) ],'target_len':[ len(l2) ]})
                        #if(i==20):
                        #    break
        else:
            self.test_mode = True
            with open(file_source) as f:
                for i,l in enumerate(f):
                    l = [ int(_) for _ in l.strip().split()]
                        
                    self.data.append({'source':l,'source_len':[ len(l) ]})
                    #if(i==20):
                    #    break
            
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self,sample):
        #prlong(sample)
        if('target' in sample):
            return  {
                    'source':torch.tensor(sample['source'],dtype=torch.long),
                    'target':torch.tensor(sample['target'],dtype=torch.long),
                    'source_len':sample['source_len'],
                    'target_len':sample['target_len']
                    }
        else:
            return  {
                    'source':torch.tensor(sample['source'],dtype=torch.long),
                    'source_len':sample['source_len']
                    }


def collate_fn(data):
    #print(data[0])
    output = dict()
    #deal with source and target
    l = 0
    for i in range(len(data)):
        l = max(l,data[i]['source'].shape[0])
    for i in range(len(data)):
        if(l-data[i]['source'].shape[0]):
            data[i]['source'] =  torch.cat([data[i]['source'],torch.zeros(l-data[i]['source'].shape[0],dtype=torch.long)],dim=-1)
    
    if('target' in data[0]):
        l = 0
        for i in range(len(data)):
            l = max(l,data[i]['target'].shape[0])
        for i in range(len(data)):
            if(l-data[i]['target'].shape[0]):
                data[i]['target'] =  torch.cat([data[i]['target'],torch.zeros(l-data[i]['target'].shape[0],dtype=torch.long)],dim=-1)
    
    for name in ['source','target']:
        if(name not in data[0]):
            continue

        arr = [ data[i][name] for i in range(len(data))]
        output[name] = torch.stack(arr,dim=0)
    
    #print("output",output)
    output['source'] = output['source'].transpose(0,1)
    output['source_len'] = np.concatenate([ data[i]['source_len'] for i in range(len(data))],axis=0)

    if('target' in output):
        output['target'] = output['target'].transpose(0,1)
        output['target_len'] = np.concatenate([ data[i]['target_len'] for i in range(len(data))],axis=0)

    return output

if(__name__ == '__main__'):
    dataset = itemDataset(file_source='ch_en.train.source',
                                transform=transforms.Compose([ToTensor()]))
    
    
    dataloader = DataLoader(dataset, batch_size=256,
                        shuffle=False, num_workers=16,collate_fn=collate_fn)

    for i,data in enumerate(dataloader):
        print(i,data) 
        break

            
