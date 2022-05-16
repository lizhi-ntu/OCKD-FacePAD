import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GeneralSet(Dataset):
    def __init__(self, name=None, sub=None, mode=None, sps=None):
        imgs = []
        labels = []
        
        fh = open('datasets/{}/{}_{}.txt'.format(name, name, sub), 'r')
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append(words[0])
            labels.append(int(words[1]))
        fh.close()
        
        imgs = np.array(imgs)
        labels = np.array(labels)
        
        if mode == 'grand':
            idx = np.where(labels!=-1)
        if mode == 'adaptation':
            idx = np.where(labels==0)

        self.root = 'datasets/{}'.format(name)
        self.imgs = imgs[idx]
        self.labels = labels[idx]
        self.transform = transforms.Compose([
                transforms.Resize(sps),
                transforms.ToTensor()
                ])


    def __getitem__(self, index):
        label = self.labels[index]
        
        img_name = self.imgs[index] 
        img_path = '{}/rgb/{}.jpg'.format(self.root, img_name)
        img = Image.open(img_path)
        img = self.transform(img)
        img = (img - 0.5) / 0.5
        
        return img, label

    def __len__(self):
        return len(self.imgs)

class ClientSet(Dataset):
    def __init__(self, sub, client=None, sps=None):
        imgs = []
        clients = []
        labels = []
                
        fh = open('datasets/client/{}_list.txt'.format(sub), 'r')
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append(words[0])
            clients.append(int(words[1]))
            labels.append(int(words[2]))
        
        fh.close()
        
        imgs = np.array(imgs)
        clients = np.array(clients)
        labels = np.array(labels)
        self.root = 'datasets/client'
        self.sub = sub
        self.client = client
        self.transform = transforms.Compose([
                transforms.Resize(sps),
                transforms.ToTensor()
                ])

        idx = np.where(clients==client)

        self.imgs = imgs[idx]
        self.labels = labels[idx]

    def __getitem__(self, index):
        label = self.labels[index]
        
        img_name = self.imgs[index] 
        img_path = '{}/rgb/{}/{}.jpg'.format(self.root, self.sub, img_name)
        img = Image.open(img_path)
        img = self.transform(img)
        img = (img - 0.5) / 0.5
        return img, label

    def __len__(self):
        return len(self.imgs)
