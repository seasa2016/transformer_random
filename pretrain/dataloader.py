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
        a = 0
        b = 0
        c = 100
        d = 100
        for i in range(temp.shape[0]):
            raw = temp.iloc[i]
            playlist = dict()
            for name in ["source","target","language"]:
                playlist[name] = []
                if(isinstance(raw[name],float)):
                    continue
                if(type(raw[name]) != np.int64):
                    for _ in raw[name].strip().split(","):
                        playlist[name].append(int(_))
                else:
                    playlist[name] = [raw[name]]

            if(len(playlist['source'])<5 or len(playlist['target'])<5 ):
                continue
            if(len(playlist['source'])>a):
                a = len(playlist['source'])
            if(len(playlist['target'])>b):
                b = len(playlist['target'])
            if(len(playlist['source'])<c):
                c = len(playlist['source'])
            if(len(playlist['target'])<d):
                d = len(playlist['target'])

            playlist['source_len'] = [ len(playlist['source']) +2]
            playlist['target_len'] = [ len(playlist['target']) +2]
            playlist["source"] = np.array(playlist["language"] + playlist["source"]+[1],dtype=np.long)
            playlist["target"] = np.array(playlist["target"],dtype=np.long)
            self.data.append(playlist)
        print("max",a,b)
        print("min",c,d)

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
        np.random.shuffle(sample['target'])
        qq = sample['target'].copy()
        sample['target'] = np.concatenate([[2],sample['target'],[1]])
        return{
            'source':torch.tensor(sample['source'],dtype=torch.long),
            'target':torch.tensor(sample['target'],dtype=torch.long),
            'source_len':torch.tensor(sample['source_len'],dtype=torch.long),
            'target_len':torch.tensor(sample['target_len'],dtype=torch.long),
            'origin':torch.tensor(qq,dtype=torch.long)
            }

def collate_fn(data):
    
    output = dict()
    #deal with source and target
    for t in ['source','target','origin']:
        l = 0
        for i in range(len(data)):
            l = max(l,data[i][t].shape[0])
        if(l == 0):
            continue
        for i in range(len(data)):
            if(l-data[i][t].shape[0]):
                data[i][t] =  torch.cat([data[i][t],torch.zeros(l-data[i][t].shape[0],dtype=torch.long)],dim=-1)
    
    
    for name in [ 'source','target','origin']:
        if(name not in data[0]):
            continue

        arr = [ data[i][name] for i in range(len(data))]
        output[name] = torch.stack(arr,dim=0)
    
    output['source'] = output['source'].transpose(0,1)
    output['source_len'] = torch.cat([ data[i]['source_len'] for i in range(len(data))],dim=0)
    if('target' in output):
        output['target'] = output['target'].transpose(0,1)
        output['target_len'] = torch.cat([ data[i]['target_len'] for i in range(len(data))],dim=0)
    
    return output

if(__name__ == '__main__'):
    print('QQQ')
    dataset = itemDataset(file_name='playlist_20181024_train.csv',
                                transform=transforms.Compose([ToTensor()]))
    
    
    dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=False, num_workers=10,collate_fn=collate_fn)

    for i,data in enumerate(dataloader):
        if(i==0):
            print(data) 
        break
        print(i)
            
