#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import CelebADataset, SfSNetDataset, MaskedCelebADataset
from model_two import SfsNetPipeline, SkipNet
from utils import weights_init, save_image, get_normal_in_range, denorm
from log import telegram_logger as tl

path = '/work/ws-tmp/g051151-sfsnet/masked_celeba/'
batch_size = 1

test_set = MaskedCelebADataset(path)
dl = DataLoader(test_set, batch_size=batch_size, shuffle=False)

tl('generating maksed pseudo data using skipnet with 2nd-constraint ')
tl(len(dl))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sfs_net = SkipNet()

sfs_net.load_state_dict(torch.load('./skipnet_syn_2nd_constraint.pth')) #, map_location=device))
sfs_net = sfs_net.to(device)


out_folder = '/work/ws-tmp/g051151-sfsnet/pseudo-masked-skipnet-2nd-constraint/'

# see data_loading.py of sfsnet project
sfs_net.eval()
with torch.no_grad():
    for bix, data in enumerate(dl):
        face, mask = data
        mask = mask.to(device)
        face = face.to(device)
        masked_face = face * mask
        file_name = out_folder + str(bix)
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net(masked_face)
        predicted_normal = get_normal_in_range(predicted_normal)
        save_image(predicted_normal, path=file_name+'_normal.png', tl=False)
        save_image(predicted_albedo, path=file_name+'_albedo.png', tl=False)
        save_image(predicted_shading, path=file_name+'_shading.png', tl=False)
        save_image(predicted_face, path=file_name+'_recon.png', tl=False)
        save_image(face, path=file_name+'_face.png', tl=False)
        save_image(mask, path=file_name+'_mask.png', tl=False)
        np.savetxt(file_name+'_light.txt',
                   predicted_sh.cpu().detach().numpy(), delimiter='\t')
        if bix % 10000 == 0: tl(f'done till: {bix}')
        
tl('finished gen pseudo')
