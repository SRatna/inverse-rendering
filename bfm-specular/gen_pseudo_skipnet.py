#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import MaskedCelebADataset
from model_two import SkipNet
from utils import save_image, denorm
from log import telegram_logger as tl
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--celeba-data-dir', type=str)
parser.add_argument('--pseudo-data-dir', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--batch-size', type=int, default=40)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

path = args.celeba_data_dir
model_path = args.model
out_folder = args.pseudo_data_dir

test_set = MaskedCelebADataset(path)
dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
os.makedirs(out_folder, exist_ok=True)
tl(out_folder)
tl(len(dl))
tl(f'batch size: {args.batch_size}')

device = args.device
skip_net = SkipNet(device)

skip_net.load_state_dict(torch.load(model_path, map_location=device))
skip_net = skip_net.to(device)



# see data_loading.py of skipnet project
skip_net.eval()
with torch.no_grad():
    for bix, data in enumerate(dl):
        face, mask = data
        mask = mask.to(device)
        face = face.to(device)
        masked_face = face * mask
        file_name = out_folder + str(bix)
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = skip_net(masked_face)
        predicted_normal = denorm(predicted_normal)
        for i in range(len(face)):
            file_name = f'{out_folder}{bix}_{i}'
            save_image(predicted_normal[i], path=file_name+'_normal.png', tl=False)
            save_image(predicted_albedo[i], path=file_name+'_albedo.png', tl=False)
            save_image(predicted_shading[i], path=file_name+'_shading.png', tl=False)
            save_image(predicted_face[i], path=file_name+'_recon.png', tl=False)
            save_image(face[i], path=file_name+'_face.png', tl=False)
            save_image(mask[i], path=file_name+'_mask.png', tl=False)
            np.savetxt(file_name+'_light.txt',
                    predicted_sh[i].cpu().detach().numpy(), delimiter='\t')
        if bix % (len(dl)/4) == 0: tl(f'done till: {bix}')
        
tl('finished gen pseudo')