#!/usr/bin/env python
# coding: utf-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import random
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', type=str)
args = parser.parse_args()

base_dir = args.base_dir


def normalize(image):
    image = np.asarray(image).astype(np.float32) / 255.
    return image


def get_shading(N, L, w_brdf):
    # so Normal is used to calculate spherical harmonics basis
    # which inturn is multiplied by lighting parmeters to obtain shading
    c1 = np.sqrt(1/(4*np.pi)) * 3.141593
    c2 = np.sqrt(3/(4*np.pi)) * 2.094395
    c3 = np.sqrt(5/(16*np.pi)) * 0.785398
    c4 = np.sqrt(15/(4*np.pi)) * 0.785398
    c5 = np.sqrt(15/(16*np.pi)) * 0.785398

    nx = N[:, :, 0]  # * -1
    ny = N[:, :, 1]  # * -1
    nz = N[:, :, 2]  # * -1

    h, w, c = N.shape

    Y1 = c1 * np.ones((h, w))
    Y3 = c2 * nz  # * -1
    Y4 = c2 * nx
    Y2 = c2 * ny
    Y7 = c3 * (2 * nz * nz - nx * nx - ny * ny)
    Y8 = c4 * nx * nz  # * -1
    Y6 = c4 * ny * nz  # * -1
    Y9 = c5 * (nx * nx - ny * ny)
    Y5 = c4 * nx * ny

    # split 27 to 3 different 9 parameters, each for a layer: RGB
    sh = np.split(L, 3)
#     print(sh)
    assert(c == len(sh))
    shading = np.zeros((h, w, c))

    for j in range(c):
        l = sh[j]
#         print(l)
        l = l.reshape((9, 1))
        # Scale to 'h x w' dim: repeat parameters of each layer to scale it up
        l = l.repeat(h*w, axis=1).reshape((9, h, w))
        l = l.transpose([1, 2, 0])
        # Generate shading
        shading[:, :, j] = (Y1 * l[:, :, 0] + Y2 * l[:, :, 1] + Y3 * l[:, :, 2] + Y4 * l[:, :, 3] + Y5 *
                            l[:, :, 4] + Y6 * l[:, :, 5] + Y7 * l[:, :, 6] + Y8 * l[:, :, 7] + Y9 * l[:, :, 8])  # / np.pi

    y = (L[1] + L[10] + L[19]) * c2
    z = (L[2] + L[11] + L[20]) * c2
    x = (L[3] + L[12] + L[21]) * c2

    l = np.array([x, y, z])
    l = l/np.linalg.norm(l)

    # half way vector -> light - view direction
    h = np.array([l[0], l[1], l[2] + 1])
    cos = N.dot(h)  # angle bet normals and half way vector
    brdf = np.maximum(cos, 0.0) ** 1  # shininess -> 1
    brdf = np.expand_dims(brdf, axis=2)
    brdf /= np.maximum(brdf.max(), 1e-04)
    s = w_brdf*brdf + shading  # predict weight for brdf
    return s


faces = []

for img in sorted(glob.glob(base_dir + '/img/*/*')):
    if ('albedo' not in img) and ('normals' not in img):
        faces.append(img)

print(len(faces))

for face in faces:
    weight_specular = round(random.random(), 1)
    rps_path = face.replace('/img/', '/rps/').replace('.png', '.rps')
    with open(rps_path) as json_file:
        data = json.load(json_file)
    pd_sh = np.array(data['environmentMap']['coefficients'])
    l = pd_sh.T.reshape(-1)

    n = Image.open(face.replace('.png', '_normals.png'))
    n = normalize(n)
    n = 1 - 2*n
    n = (n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-4))

    a = Image.open(face.replace('.png', '_albedo.png'))
    a = normalize(a)
    s = get_shading(n, l, weight_specular)
    r = a * s
    plt.imsave(face, r.clip(0, 1))

    data['directionalLight']['specular'] = weight_specular
    with open(rps_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

print(base_dir, 'done')
