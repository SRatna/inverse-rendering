#!/usr/bin/env python
# coding: utf-8
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import PhotoFaceDataset
from model_two import SfsNetPipeline as SfsNetPipelineBFM
from log import telegram_logger as tl
from utils import load_model_from_pretrained
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=str)
parser.add_argument('--data-dir', type=str)
parser.add_argument('--mix-out-dir', type=str)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--batch', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

device = args.device
test_set = PhotoFaceDataset(args.data_dir, args.batch)
dl = DataLoader(test_set, batch_size=1, shuffle=False)
tl(len(dl))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'.  """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


models = [
    {
        'path': f'{args.out_dir}/sfsnet-bfm.pth',
        'type': 'bfm',
        'name': 'Sfsnet BFM'
    },
    {
        'path': f'{args.mix_out_dir}/mix-sfsnet-bfm.pth',
        'type': 'bfm',
        'name': 'Sfsnet BFM Mix'
    },
]

for model in models:
    tl(model['name'])
    net = SfsNetPipelineBFM(device)
    net.load_state_dict(torch.load(
        model['path'], map_location=args.device))
    net = net.to(device)

    errors = []
    net.eval()
    with torch.no_grad():
        for bix, data in enumerate(dl):
            albedo, normal, mask, face = data
            mask = mask.to(device)
            face = face.to(device)
            normal = normal.to(device)
            masked_face = face * mask
            predicted_normal, _, _, _, _ = net(masked_face)

            predicted_normal = predicted_normal.squeeze(
                0).permute(1, 2, 0).cpu().numpy()
            normal = normal.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if model['type'] == 'pretrained':
                predicted_normal = (predicted_normal*2) - 1
            if model['type'] == 'bfm':
                # change normal axis as bfm normals are aligned differently
                x = np.copy(predicted_normal[:, :, 1]*-1)
                y = np.copy(predicted_normal[:, :, 0])
                predicted_normal[:, :, 0], predicted_normal[:, :, 1] = x, y
            err = []
            for i in range(128):
                for j in range(128):
                    if (mask[i, j] == 1).all():
                        err.append(angle_between(
                            normal[i, j], predicted_normal[i, j]))
#                    else: err.append(0)
            if len(err) > 0:
                errors.append(np.array(err).mean())
            if bix % (args.batch/2) == 0:
                tl(f'Done till: {bix}')

    errors = np.array(errors)
    m, s = errors.mean(), errors.std()
    m, s = np.rad2deg(m), np.rad2deg(s)
    model['mean'] = m
    model['std'] = s
    tl(f'{model["name"]}: Mean error: {m} Std: {s}')


with open(f'{args.mix_out_dir}/photoface-eval-sfs-round-2-bfm-{args.batch}-mix-round-{args.round}.out', 'wb') as fp:
    pickle.dump(models, fp)

print(models)
