#!/usr/bin/env python
# coding: utf-8
from torch.utils.data import DataLoader, random_split
from datasets import BFMDataset
from utils import weights_init, save_image
from log import telegram_logger as tl
import matplotlib.pyplot as plt
from model_two import SfsNetPipeline
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import torch
import pickle
import matplotlib
matplotlib.use('Agg')


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str)
parser.add_argument('--out-dir', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--val-size', type=float, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=40)
parser.add_argument('--use-pretrained', type=str, default='no')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size

tl(f'sfsnet: {args.data_dir}, {batch_size}, no of epochs: {num_epochs}, out dir: {args.out_dir}')
tl('using bfm dataset')
print(args)
tl(f'lr: {args.lr} with mse loss')
full_dataset = BFMDataset(args.data_dir)
dataset_size = len(full_dataset)
validation_split = args.val_size
validation_count = int(validation_split * dataset_size / 100)
train_count = dataset_size - validation_count

train_dataset, val_dataset = random_split(
    full_dataset, [train_count, validation_count])
syn_train_dl = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

tl((100 - validation_split))

tl(len(train_dataset))

device = args.device
tl(device)
sfs_net = SfsNetPipeline(device)
if args.use_pretrained == 'yes':
    sfs_net.load_state_dict(torch.load(args.model))
else:
    sfs_net.apply(weights_init)
sfs_net = sfs_net.to(device)
model_parameters = sfs_net.parameters()
optimizer = torch.optim.Adam(model_parameters, lr=args.lr)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

normal_loss = nn.MSELoss()
albedo_loss = nn.MSELoss()
sh_loss = nn.MSELoss()
recon_loss = nn.MSELoss()
specular_loss = nn.MSELoss()

sfs_net.train()

lamda_recon = 0
# if args.use_pretrained == 'yes':
#     lamda_recon = 0.5
lamda_albedo = 0.5
lamda_normal = 0.5
lamda_sh = 0.1
syn_train_len = len(syn_train_dl)
loss_dict = {
    'total': [],
    'albedo': [],
    'normal': [],
    'sh': [],
    'specular': [],
    'recon': []
}
for epoch in range(1, num_epochs+1):
    tloss = 0  # Total loss
    nloss = 0  # Normal loss
    aloss = 0  # Albedo loss
    shloss = 0  # SH loss
    sloss = 0  # specular loss
    rloss = 0  # Reconstruction loss
    tl(f'Current Epoch: {epoch}')

    for bix, data in enumerate(syn_train_dl):
        albedo, normal, mask, sh, face = data
        albedo = albedo.to(device)
        normal = normal.to(device)
        mask = mask.to(device)
        sh = sh.to(device)
        face = face.to(device)

        # Apply Mask on input image
        face = face * mask
        predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon = sfs_net(
            face)

        # Loss computation
        # Normal loss
        current_normal_loss = normal_loss(predicted_normal, normal)
        # Albedo loss
        current_albedo_loss = albedo_loss(predicted_albedo, albedo)
        # SH loss
        current_sh_loss = sh_loss(predicted_sh[:, :-1], sh[:, :-1])
        # Specular loss
        current_specular_loss = specular_loss(predicted_sh[:, -1], sh[:, -1])
        # Reconstruction loss
        # Edge case: Shading generation requires denormalized normal and sh
        # Hence, denormalizing face here
        current_recon_loss = recon_loss(out_recon, face)

        total_loss = lamda_normal * current_normal_loss + \
            lamda_albedo * current_albedo_loss + \
            lamda_sh * current_sh_loss + \
            lamda_sh * current_specular_loss + \
            lamda_recon * current_recon_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        exp_lr_scheduler.step()

        # Logging for display and debugging purposes
        tloss += total_loss.item()
        nloss += current_normal_loss.item()
        aloss += current_albedo_loss.item()
        shloss += current_sh_loss.item()
        sloss += current_specular_loss.item()
        rloss += current_recon_loss.item()

    cur_tloss = tloss / syn_train_len
    cur_nloss = nloss / syn_train_len
    cur_aloss = aloss / syn_train_len
    cur_shloss = shloss / syn_train_len
    cur_sloss = sloss / syn_train_len
    cur_rloss = rloss / syn_train_len
    loss_dict['total'].append(cur_tloss)
    loss_dict['normal'].append(cur_nloss)
    loss_dict['albedo'].append(cur_aloss)
    loss_dict['sh'].append(cur_shloss)
    loss_dict['specular'].append(cur_sloss)
    loss_dict['recon'].append(cur_rloss)

    tl(f'Sfsnet: Training set results Data path {args.out_dir}: Total Loss: {cur_tloss}, \
    Normal Loss: {cur_nloss}, Albedo Loss: {cur_aloss}, SH Loss: {cur_shloss}, \
    Specular Loss: {cur_sloss}, Recon Loss: {cur_rloss}')

    save_image(face, f'{args.out_dir}/tmp/face_{epoch}.png')
    save_image(normal, f'{args.out_dir}/tmp/normal_{epoch}.png', True)
    save_image(predicted_normal, f'{args.out_dir}/tmp/predicted_normal_{epoch}.png', True)
    save_image(albedo, f'{args.out_dir}/tmp/albedo_{epoch}.png')
    save_image(predicted_albedo, f'{args.out_dir}/tmp/predicted_albedo_{epoch}.png')
    save_image(out_shading, f'{args.out_dir}/tmp/out_shading_{epoch}.png')
    save_image(out_recon, f'{args.out_dir}/tmp/out_recon_{epoch}.png')

with open(f'{args.out_dir}/loss.pickle', 'wb') as f:
    pickle.dump(loss_dict, f)

for l in loss_dict:
    loss_dict[l] = [np.log(x) for x in loss_dict[l]]

plt.figure(figsize=(10, 6))
data = pd.DataFrame(loss_dict)
ax = sns.lineplot(data=data)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Total Loss', 'Albedo Loss', 'Normal Loss',
            'Shading Loss', 'Specular Loss' 'Reconstruction Loss'])
plt.savefig(f'{args.out_dir}/result_sfs.png')


torch.save(sfs_net.state_dict(), f'{args.out_dir}/sfsnet-bfm.pth')

tl(f'training done bfm {args.out_dir}')

syn_val_dl = DataLoader(val_dataset, batch_size=16, shuffle=True)


sfs_net.eval()
with torch.no_grad():
    i = 0
    for bix, data in enumerate(syn_val_dl):
        albedo, normal, mask, sh, face = data
        albedo = albedo.to(device)
        normal = normal.to(device)
        mask = mask.to(device)
        sh = sh.to(device)
        face = face.to(device)
        face = face * mask
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net(
            face)
        save_image(face, f'{args.out_dir}/face_sfs.png')
        save_image(predicted_normal, f'{args.out_dir}/normal_sfs.png', True)
        save_image(predicted_albedo, f'{args.out_dir}/albedo_sfs.png')
        save_image(predicted_shading, f'{args.out_dir}/shading_sfs.png')
        save_image(predicted_face, f'{args.out_dir}/recon_sfs.png')
        i += 1
        if i > 1:
            break
