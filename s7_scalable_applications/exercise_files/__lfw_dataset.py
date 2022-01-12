"""
LFW dataloading
"""
import argparse
import enum
import time

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import os
from PIL import Image
from tqdm import tqdm

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform=None) -> None:
        # TODO: fill out with what you need
        path=path_to_folder
        classes=[ name for name in os.listdir(path_to_folder) if os.path.isdir(os.path.join(path_to_folder, name)) ]
        classes=np.sort(classes)
        class_map={name:c for c,name in enumerate(classes)}
        img_paths=[]
        labels=[]
        for class_name in class_map:
            path_=path+'/'+class_name
            files_names=[ name for name in os.listdir(path_) if not os.path.isdir(os.path.join(path_, name)) ]
            for file_name in files_names:
                img_paths.append(path_+'/'+file_name)
                labels.append(class_map[class_name])

        

        self.labels=np.array(labels).copy()
        self.img_paths=np.array(img_paths).copy()
        self.transform0=transforms.ToTensor()

        if transform is None:
            self.transform=self.transform0
        else:
            self.transform = transform
        
        sample_img=self.transform0(Image.open( img_paths[0]))
        self.all_imgs=torch.randn(*(len(img_paths), 3, *sample_img.shape[-2:]))
        for idx in tqdm(range(len(img_paths))):
            self.all_imgs[idx]=self.transform0(Image.open( img_paths[idx]))
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(self.all_imgs[index])

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='/home/slangen/Documents/lfw', type=str)
    parser.add_argument('-num_workers', default=2, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0))
        
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    batch_size=512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers)
    
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        dataiter = iter(dataloader)
        images = dataiter.next()    
        grid_img = torchvision.utils.make_grid(images, nrow=int(np.sqrt(batch_size))+1)
        plt.figure()
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.savefig('sample_batch.png')
        pass
        
    if args.get_timing:
        # lets do so repetitions
        res = [ ]
        for _ in tqdm(range(5)):
            start = time.time()
            for batch_idx, batch in tqdm(enumerate(dataloader)):
                if batch_idx > 100:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')
