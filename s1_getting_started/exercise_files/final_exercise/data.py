from typing_extensions import ParamSpecArgs
import torch
from os import walk
import numpy as np

# PATH="../../../data/corruptmnist/"

def mnist(PATH,train=True):
    # exchange with the corrupted mnist dataset
    filenames = next(walk(PATH), (None, None, []))[2]  # [] if no file
    train_paths=[PATH+i for i in filenames if 'train' in i]
    test_paths=[PATH+i for i in filenames if 'test' in i]

    if train:
        train_images=np.concatenate([np.load(i)['images'] for i in train_paths],axis=0)
        train_labels=np.concatenate([np.load(i)['labels'] for i in train_paths],axis=0)
        train = [train_images,train_labels]
        return train
    
    else: 
        test_images=np.concatenate([np.load(i)['images'] for i in test_paths],axis=0)
        test_labels=np.concatenate([np.load(i)['labels'] for i in test_paths],axis=0)
        test = [test_images,test_labels]
        return test

class Dataset_mnist():
    def __init__(self,data):
        self.data=data
        self.images=torch.from_numpy(self.data[0].copy()).float()
        self.labels=torch.from_numpy(np.array(self.data[1]).copy())

    def __getitem__(self, idx):
        
        return self.images[idx],self.labels[idx]

    def __len__(self):
        return (len(self.images))

