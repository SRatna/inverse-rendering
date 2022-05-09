import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import glob
import numpy as np
import random
import json

IMAGE_SIZE = 128

class CelebADataset(Dataset):
    def __init__(self, path=None, read_first=None):
        df = pd.read_csv(path)
        df = df[:read_first]
        self.face = list(df['face'])
        self.dataset_len = len(self.face)
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        face = self.transform(Image.open(self.face[index]))
        return face

    def __len__(self):
        return self.dataset_len


class SfSNetDataset(Dataset):
    def __init__(self, path=''):
        albedo = []
        sh = []
        mask = []
        normal = []
        face = []

        for img in sorted(glob.glob(path + '*/*_albedo_*')):
            albedo.append(img)
        for img in sorted(glob.glob(path + '*/*_face_*')):
            face.append(img)
        for img in sorted(glob.glob(path + '*/*_normal_*')):
            normal.append(img)
        for img in sorted(glob.glob(path + '*/*_mask_*')):
            mask.append(img)
        for img in sorted(glob.glob(path + '*/*_light_*.txt')):
            sh.append(img)
        self.albedo = albedo
        self.face = face
        self.normal = normal
        self.mask = mask
        self.sh = sh
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        self.dataset_len = len(self.albedo)
        self.normal_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE)
        ])

    def __getitem__(self, index):
        albedo = self.transform(Image.open(self.albedo[index]))
        face = self.transform(Image.open(self.face[index]))
        # normal = io.imread(self.face[index]))
        normal = self.normal_transform(Image.open(self.normal[index]))
        normal = torch.tensor(np.array(normal)).permute([2, 0, 1])
        normal = normal.type(torch.float)
        normal = (normal - 128) / 128
        mask = self.transform(Image.open(self.mask[index]))
        pd_sh = pd.read_csv(self.sh[index], sep='\t', header=None)
        sh = torch.tensor(pd_sh.values).type(torch.float).reshape(-1)
        return albedo, normal, mask, sh, face

    def __len__(self):
        return self.dataset_len
    
class PseudoDataset(Dataset):
    def __init__(self, path=''):
        face = []
        for img in sorted(glob.glob(path + '*_face*')):
            face.append(img)
        self.face = face
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        self.dataset_len = len(self.face)

    def get_normal(self, image):
        n1 = np.asarray(image).astype(np.float32) / 255.
        n1 = 2*n1 - 1
        n1 = (n1 / np.maximum(np.linalg.norm(n1, axis=2, keepdims=True), 1e-4))
        n1 = torch.tensor(n1).permute([2, 0, 1]).type(torch.float)
        return n1

    def __getitem__(self, index):
        albedo = self.transform(Image.open(self.face[index].replace('_face', '_albedo')))
        face = self.transform(Image.open(self.face[index]))
        normal = self.get_normal(Image.open(self.face[index].replace('_face', '_normal')))
        mask = self.transform(Image.open(self.face[index].replace('_face', '_mask')))
        pd_sh = pd.read_csv(self.face[index].replace('_face.png', '_light.txt'), sep='\t', header=None)
        sh = torch.tensor(pd_sh.values).type(torch.float).reshape(-1)
        return albedo, normal, mask, sh, face

    def __len__(self):
        return self.dataset_len

class MaskedCelebADataset(Dataset):
    def __init__(self, path=''):
        face = []
        for img in sorted(glob.glob(path + '*_face*')):
            face.append(img)
        self.face = face
        self.transform = transforms.Compose([
#             transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face = self.transform(Image.open(self.face[index]))
        mask = self.transform(Image.open(self.face[index].replace('_face', '_mask')))
        return face, mask

    def __len__(self):
        return self.dataset_len
    
class PhotoFaceDataset(Dataset):
    def __init__(self, path='', count=100):
        face = []

        for img in sorted(glob.glob(path + '*_face*')):
            face.append(img)
        self.face = face if count == 0 else random.sample(face, count)
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        self.dataset_len = len(self.face)
        self.normal_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE)
        ])

    def __getitem__(self, index):
        face = self.transform(Image.open(self.face[index]))
        albedo = self.transform(Image.open(self.face[index].replace('_face', '_albedo')))
        normal = self.normal_transform(Image.open(self.face[index].replace('_face', '_normal')))
        normal = torch.tensor(np.array(normal)).permute([2, 0, 1])
        normal = normal.type(torch.float)
        normal = (normal - 128) / 128
        mask = self.transform(Image.open(self.face[index].replace('_face', '_mask')))
        return albedo, normal, mask, face

    def __len__(self):
        return self.dataset_len

class BFMDataset(Dataset):
    def __init__(self, path=''):
        face = []

        for img in sorted(glob.glob(path + 'img/*/*')):
            if ('albedo' not in img) and ('normals' not in img):
                face.append(img)
        self.face = face
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        self.dataset_len = len(self.face)
        self.normal_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE)
        ])


    def get_normal(self, image):
        n1 = np.asarray(image).astype(np.float32) / 255.
        n1 = 1 - 2*n1
        n1 = (n1 / np.maximum(np.linalg.norm(n1, axis=2, keepdims=True), 1e-4))
#         n1[:,:,2] = n1[:,:,2] * -1
        n1 = torch.tensor(n1).permute([2, 0, 1]).type(torch.float)
        return n1
    
    def __getitem__(self, index):
        albedo = self.transform(Image.open(self.face[index].replace('.png', '_albedo.png')))
        face = self.transform(Image.open(self.face[index]).convert('RGB'))
        normal = Image.open(self.face[index].replace('.png', '_normals.png'))
        normal = self.get_normal(normal)
        mask = torch.ones(3, IMAGE_SIZE, IMAGE_SIZE)
        rps_path = self.face[index].replace('/img/', '/rps/').replace('.png', '.rps')
        with open(rps_path) as json_file:
            data = json.load(json_file)
        pd_sh = np.array(data['environmentMap']['coefficients']).T.reshape(-1)
        sh = torch.tensor(pd_sh).type(torch.float)
        # works for both dataset: with [0-1] and without [0.0, 0.0, 0.0] specular defined
        specular = data['directionalLight']['specular']
        specular = specular if type(specular) == float else specular[0]
        specular = torch.tensor([specular]).type(torch.float)
        sh = torch.cat((sh, specular))
        return albedo, normal, mask, sh, face

    def __len__(self):
        return self.dataset_len

