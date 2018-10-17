from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import pandas as pd
import sys

#add sos bos
class itemDataset(Dataset):
    def __init__(self, file_name,transform=None):
        
        temp = pd.read_csv(file_name)
        self.data=[]
        for i in range(temp.shape[0]):
            raw = temp.iloc[i]
            playlist = dict()
            
            for name in ["source","target"]:
                playlist[name] = []
                if(isinstance(raw[name],float)):
                    continue

                for _ in raw[name].strip().split(","):
                    playlist[name].append(int(_))
            
            playlist['source_len'] = [ len(playlist['source']) ]
            playlist['target_len'] = [ len(playlist['target']) ]

            self.data.append(playlist)


        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class tolist(object):
    def __init__(self,output_size):
        assert(isinstance(output_size,dict))
        
        self.output_size = output_size
                
    def __call__(self,sample):
        output = dict()
        output['source'] = sample['source']
        output['source_len'] = sample['source_len']
        
        output['target'] = sample['target']
        output['target_len'] = sample['target_len']


        return output

class ToTensor(object):
    def __call__(self,sample):
        #prlong(sample)
        return{
            'source':torch.tensor(sample['source'],dtype=torch.long),
            'target':torch.tensor(sample['target'],dtype=torch.long),
            'source_len':sample['source_len'],
            'target_len':sample['target_len']
            }

def collate_fn(data):
    
    output = dict()
    #deal with source and target
    l = 0
    for i in range(len(data)):
        l = max(l,data[i]['source'].shape[0])
    for i in range(len(data)):
        if(l-data[i]['source'].shape[0]):
            data[i]['source'] =  torch.cat([data[i]['source'],torch.zeros(l-data[i]['source'].shape[0],dtype=torch.long)],dim=-1)
    
    l = 0
    for i in range(len(data)):
        l = max(l,data[i]['target'].shape[0])
    if(l == 0):
        l = None
    for i in range(len(data)):
        if(l == None):
            break
        elif(l-data[i]['target'].shape[0]):
            data[i]['target'] =  torch.cat([data[i]['target'],torch.zeros(l-data[i]['target'].shape[0],dtype=torch.long)],dim=-1)
    
    for name in [ 'source','target']:
        if(name == 'target' and l is None):
            continue

        arr = [ data[i][name] for i in range(len(data))]
        output[name] = torch.stack(arr,dim=0)
    
    output['source'] = output['source'].transpose(0,1)
    output['source_len'] = np.concatenate([ data[i]['source_len'] for i in range(len(data))],axis=0)

    if('target' in output):
        output['target'] = output['target'].transpose(0,1)
        output['target_len'] = np.concatenate([ data[i]['target_len'] for i in range(len(data))],axis=0)

    
    return output

if(__name__ == '__main__'):
    output_size = dict()
    for name in ['source','language','year','acoustic','genre','context','o']:
        with open('token_{0}.label'.format(name)) as f:
            output_size[name] = len(f.readlines())
    print('QQQ')
    dataset = itemDataset(file_name='playlist_20180826_train.csv',
                                transform=transforms.Compose([tolist(output_size),ToTensor()]))
    
    
    dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=False, num_workers=10,collate_fn=collate_fn)

    for i,data in enumerate(dataloader):
        print(i)
        if(i==0):
            print(data) 

            
