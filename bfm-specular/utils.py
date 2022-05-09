import matplotlib
matplotlib.use('Agg')
import torch
from torch.nn import *
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from log import telegram_upload_image as ui

def reconstruct_image(shading, albedo):
    return shading * albedo

def denorm(x):
    out = (1 - x) / 2
    return out.clamp(0, 1)

def get_image_grid(pic, denormalize=False):
    if denormalize:
        pic = denorm(pic)

    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return ndarr

def save_image(pic, path=None, denormalize=False, tl=True):
    ndarr = get_image_grid(pic, denormalize=denormalize)    
    
    if path == None:
        plt.imshow(ndarr)
        plt.show()
    else:
        im = Image.fromarray(ndarr)
        im.save(path)
        if tl: ui(path)

def save_batch_image(pic, bix, out_dir, file_ext):
    images = pic.mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    for idx, img in enumerate(images):
        file_name = f'{out_dir}{bix}_{idx}_{file_ext}'
        Image.fromarray(img).save(file_name)

def get_shading(N, L, device):
    # so Normal is used to calculate spherical harmonics basis
    # which inturn is multiplied by lighting parmeters to obtain shading
    c1 = 0.8862269254527579
    c2 = 1.0233267079464883
    c3 = 0.24770795610037571
    c4 = 0.8580855308097834
    c5 = 0.4290427654048917

    nx = N[:, 0, :, :]
    ny = N[:, 1, :, :]
    nz = N[:, 2, :, :]

    b, c, h, w = N.shape

    Y1 = c1 * torch.ones(b, h, w)
#     Y2 = c2 * nz
#     Y3 = c2 * nx
#     Y4 = c2 * ny
#     Y5 = c3 * (2 * nz * nz - nx * nx - ny * ny)
#     Y6 = c4 * nx * nz
#     Y7 = c4 * ny * nz
#     Y8 = c5 * (nx * nx - ny * ny)
#     Y9 = c4 * nx * ny
#     Y1 = c1 * np.ones((h, w))
    Y3 = c2 * nz #* -1
    Y4 = c2 * nx
    Y2 = c2 * ny
    Y7 = c3 * (2 * nz * nz - nx * nx - ny * ny)
    Y8 = c4 * nx * nz #* -1
    Y6 = c4 * ny * nz #* -1
    Y9 = c5 * (nx * nx - ny * ny)
    Y5 = c4 * nx * ny
    
    L = L[:, :-1]
    
    w_brdf = L[:, -1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    # split 27 to 3 different 9 parameters, each for a layer: RGB
    sh = torch.split(L, 9, dim=1)

    assert(c == len(sh))
    shading = torch.zeros(b, c, h, w)
    
    Y1 = Y1.to(device)
    shading = shading.to(device)

    for j in range(c):
        l = sh[j]
        # Scale to 'h x w' dim: repeat parameters of each layer to scale it up
        l = l.repeat(1, h*w).view(b, h, w, 9)
        # Convert l into 'batch size', 'Index SH', 'h', 'w'
        l = l.permute([0, 3, 1, 2])
        # Generate shading
        shading[:, j, :, :] = Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
            Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
            Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]

    y = (L[:, 1] + L[:, 10] + L[:, 19]) * c2
    z = (L[:, 2] + L[:, 11] + L[:, 20]) * c2
    x = (L[:, 3] + L[:, 12] + L[:, 21]) * c2
    
    l = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), 1)
    l = l/torch.linalg.norm(l)
    l[:, 2] = l[:, 2] + 1 # add view vector (0, 0, 1)
    l = l.unsqueeze(2).unsqueeze(3)
    cos = (N * l).sum(1, keepdim=True) # angle bet normals and half way vector
    brdf = torch.maximum(cos, torch.tensor(0.0).to(device)) ** 1 # shininess -> 1
    
    brdf = brdf / torch.maximum(brdf.max(), torch.tensor(1e-04).to(device))
    s = w_brdf*brdf + shading # predict weight for brdf
    return s


def load_model_from_pretrained(src_model, dst_model):
    dst_model['conv_model.conv1.0.weight'] = src_model['conv1.conv.0.weight']
    dst_model['conv_model.conv1.0.bias'] = src_model['conv1.conv.0.bias']
    dst_model['conv_model.conv1.1.weight'] = src_model['conv1.conv.1.weight']
    dst_model['conv_model.conv1.1.bias'] = src_model['conv1.conv.1.bias']
    dst_model['conv_model.conv1.1.running_mean'] = src_model['conv1.conv.1.running_mean']
    dst_model['conv_model.conv1.1.running_var'] = src_model['conv1.conv.1.running_var']
    dst_model['conv_model.conv2.0.weight'] = src_model['conv2.conv.0.weight']
    dst_model['conv_model.conv2.0.bias'] = src_model['conv2.conv.0.bias']
    dst_model['conv_model.conv2.1.weight'] = src_model['conv2.conv.1.weight']
    dst_model['conv_model.conv2.1.bias'] = src_model['conv2.conv.1.bias']
    dst_model['conv_model.conv2.1.running_mean'] = src_model['conv2.conv.1.running_mean']
    dst_model['conv_model.conv2.1.running_var'] = src_model['conv2.conv.1.running_var']
    dst_model['conv_model.conv3.weight'] = src_model['conv3.weight']
    dst_model['conv_model.conv3.bias'] = src_model['conv3.bias']
    dst_model['normal_residual_model.block1.res.0.weight'] = src_model['nres1.res.0.weight']
    dst_model['normal_residual_model.block1.res.0.bias'] = src_model['nres1.res.0.bias']
    dst_model['normal_residual_model.block1.res.0.running_mean'] = src_model['nres1.res.0.running_mean']
    dst_model['normal_residual_model.block1.res.0.running_var'] = src_model['nres1.res.0.running_var']
    dst_model['normal_residual_model.block1.res.2.weight'] = src_model['nres1.res.2.weight']
    dst_model['normal_residual_model.block1.res.2.bias'] = src_model['nres1.res.2.bias']
    dst_model['normal_residual_model.block1.res.3.weight'] = src_model['nres1.res.3.weight']
    dst_model['normal_residual_model.block1.res.3.bias'] = src_model['nres1.res.3.bias']
    dst_model['normal_residual_model.block1.res.3.running_mean'] = src_model['nres1.res.3.running_mean']
    dst_model['normal_residual_model.block1.res.3.running_var'] = src_model['nres1.res.3.running_var']
    dst_model['normal_residual_model.block1.res.5.weight'] = src_model['nres1.res.5.weight']
    dst_model['normal_residual_model.block1.res.5.bias'] = src_model['nres1.res.5.bias']
    dst_model['normal_residual_model.block2.res.0.weight'] = src_model['nres2.res.0.weight']
    dst_model['normal_residual_model.block2.res.0.bias'] = src_model['nres2.res.0.bias']
    dst_model['normal_residual_model.block2.res.0.running_mean'] = src_model['nres2.res.0.running_mean']
    dst_model['normal_residual_model.block2.res.0.running_var'] = src_model['nres2.res.0.running_var']
    dst_model['normal_residual_model.block2.res.2.weight'] = src_model['nres2.res.2.weight']
    dst_model['normal_residual_model.block2.res.2.bias'] = src_model['nres2.res.2.bias']
    dst_model['normal_residual_model.block2.res.3.weight'] = src_model['nres2.res.3.weight']
    dst_model['normal_residual_model.block2.res.3.bias'] = src_model['nres2.res.3.bias']
    dst_model['normal_residual_model.block2.res.3.running_mean'] = src_model['nres2.res.3.running_mean']
    dst_model['normal_residual_model.block2.res.3.running_var'] = src_model['nres2.res.3.running_var']
    dst_model['normal_residual_model.block2.res.5.weight'] = src_model['nres2.res.5.weight']
    dst_model['normal_residual_model.block2.res.5.bias'] = src_model['nres2.res.5.bias']
    dst_model['normal_residual_model.block3.res.0.weight'] = src_model['nres3.res.0.weight']
    dst_model['normal_residual_model.block3.res.0.bias'] = src_model['nres3.res.0.bias']
    dst_model['normal_residual_model.block3.res.0.running_mean'] = src_model['nres3.res.0.running_mean']
    dst_model['normal_residual_model.block3.res.0.running_var'] = src_model['nres3.res.0.running_var']
    dst_model['normal_residual_model.block3.res.2.weight'] = src_model['nres3.res.2.weight']
    dst_model['normal_residual_model.block3.res.2.bias'] = src_model['nres3.res.2.bias']
    dst_model['normal_residual_model.block3.res.3.weight'] = src_model['nres3.res.3.weight']
    dst_model['normal_residual_model.block3.res.3.bias'] = src_model['nres3.res.3.bias']
    dst_model['normal_residual_model.block3.res.3.running_mean'] = src_model['nres3.res.3.running_mean']
    dst_model['normal_residual_model.block3.res.3.running_var'] = src_model['nres3.res.3.running_var']
    dst_model['normal_residual_model.block3.res.5.weight'] = src_model['nres3.res.5.weight']
    dst_model['normal_residual_model.block3.res.5.bias'] = src_model['nres3.res.5.bias']
    dst_model['normal_residual_model.block4.res.0.weight'] = src_model['nres4.res.0.weight']
    dst_model['normal_residual_model.block4.res.0.bias'] = src_model['nres4.res.0.bias']
    dst_model['normal_residual_model.block4.res.0.running_mean'] = src_model['nres4.res.0.running_mean']
    dst_model['normal_residual_model.block4.res.0.running_var'] = src_model['nres4.res.0.running_var']
    dst_model['normal_residual_model.block4.res.2.weight'] = src_model['nres4.res.2.weight']
    dst_model['normal_residual_model.block4.res.2.bias'] = src_model['nres4.res.2.bias']
    dst_model['normal_residual_model.block4.res.3.weight'] = src_model['nres4.res.3.weight']
    dst_model['normal_residual_model.block4.res.3.bias'] = src_model['nres4.res.3.bias']
    dst_model['normal_residual_model.block4.res.3.running_mean'] = src_model['nres4.res.3.running_mean']
    dst_model['normal_residual_model.block4.res.3.running_var'] = src_model['nres4.res.3.running_var']
    dst_model['normal_residual_model.block4.res.5.weight'] = src_model['nres4.res.5.weight']
    dst_model['normal_residual_model.block4.res.5.bias'] = src_model['nres4.res.5.bias']
    dst_model['normal_residual_model.block5.res.0.weight'] = src_model['nres5.res.0.weight']
    dst_model['normal_residual_model.block5.res.0.bias'] = src_model['nres5.res.0.bias']
    dst_model['normal_residual_model.block5.res.0.running_mean'] = src_model['nres5.res.0.running_mean']
    dst_model['normal_residual_model.block5.res.0.running_var'] = src_model['nres5.res.0.running_var']
    dst_model['normal_residual_model.block5.res.2.weight'] = src_model['nres5.res.2.weight']
    dst_model['normal_residual_model.block5.res.2.bias'] = src_model['nres5.res.2.bias']
    dst_model['normal_residual_model.block5.res.3.weight'] = src_model['nres5.res.3.weight']
    dst_model['normal_residual_model.block5.res.3.bias'] = src_model['nres5.res.3.bias']
    dst_model['normal_residual_model.block5.res.3.running_mean'] = src_model['nres5.res.3.running_mean']
    dst_model['normal_residual_model.block5.res.3.running_var'] = src_model['nres5.res.3.running_var']
    dst_model['normal_residual_model.block5.res.5.weight'] = src_model['nres5.res.5.weight']
    dst_model['normal_residual_model.block5.res.5.bias'] = src_model['nres5.res.5.bias']
    dst_model['normal_residual_model.bn1.weight'] = src_model['nreso.0.weight']
    dst_model['normal_residual_model.bn1.bias'] = src_model['nreso.0.bias']
    dst_model['normal_residual_model.bn1.running_mean'] = src_model['nreso.0.running_mean']
    dst_model['normal_residual_model.bn1.running_var'] = src_model['nreso.0.running_var']
    dst_model['normal_gen_model.conv1.0.weight'] = src_model['nconv1.conv.0.weight']
    dst_model['normal_gen_model.conv1.0.bias'] = src_model['nconv1.conv.0.bias']
    dst_model['normal_gen_model.conv1.1.weight'] = src_model['nconv1.conv.1.weight']
    dst_model['normal_gen_model.conv1.1.bias'] = src_model['nconv1.conv.1.bias']
    dst_model['normal_gen_model.conv1.1.running_mean'] = src_model['nconv1.conv.1.running_mean']
    dst_model['normal_gen_model.conv1.1.running_var'] = src_model['nconv1.conv.1.running_var']
    dst_model['normal_gen_model.conv2.0.weight'] = src_model['nconv2.conv.0.weight']
    dst_model['normal_gen_model.conv2.0.bias'] = src_model['nconv2.conv.0.bias']
    dst_model['normal_gen_model.conv2.1.weight'] = src_model['nconv2.conv.1.weight']
    dst_model['normal_gen_model.conv2.1.bias'] = src_model['nconv2.conv.1.bias']
    dst_model['normal_gen_model.conv2.1.running_mean'] = src_model['nconv2.conv.1.running_mean']
    dst_model['normal_gen_model.conv2.1.running_var'] = src_model['nconv2.conv.1.running_var']
    dst_model['normal_gen_model.conv3.weight'] = src_model['nout.weight']
    dst_model['normal_gen_model.conv3.bias'] = src_model['nout.bias']
    dst_model['albedo_residual_model.block1.res.0.weight'] = src_model['ares1.res.0.weight']
    dst_model['albedo_residual_model.block1.res.0.bias'] = src_model['ares1.res.0.bias']
    dst_model['albedo_residual_model.block1.res.0.running_mean'] = src_model['ares1.res.0.running_mean']
    dst_model['albedo_residual_model.block1.res.0.running_var'] = src_model['ares1.res.0.running_var']
    dst_model['albedo_residual_model.block1.res.2.weight'] = src_model['ares1.res.2.weight']
    dst_model['albedo_residual_model.block1.res.2.bias'] = src_model['ares1.res.2.bias']
    dst_model['albedo_residual_model.block1.res.3.weight'] = src_model['ares1.res.3.weight']
    dst_model['albedo_residual_model.block1.res.3.bias'] = src_model['ares1.res.3.bias']
    dst_model['albedo_residual_model.block1.res.3.running_mean'] = src_model['ares1.res.3.running_mean']
    dst_model['albedo_residual_model.block1.res.3.running_var'] = src_model['ares1.res.3.running_var']
    dst_model['albedo_residual_model.block1.res.5.weight'] = src_model['ares1.res.5.weight']
    dst_model['albedo_residual_model.block1.res.5.bias'] = src_model['ares1.res.5.bias']
    dst_model['albedo_residual_model.block2.res.0.weight'] = src_model['ares2.res.0.weight']
    dst_model['albedo_residual_model.block2.res.0.bias'] = src_model['ares2.res.0.bias']
    dst_model['albedo_residual_model.block2.res.0.running_mean'] = src_model['ares2.res.0.running_mean']
    dst_model['albedo_residual_model.block2.res.0.running_var'] = src_model['ares2.res.0.running_var']
    dst_model['albedo_residual_model.block2.res.2.weight'] = src_model['ares2.res.2.weight']
    dst_model['albedo_residual_model.block2.res.2.bias'] = src_model['ares2.res.2.bias']
    dst_model['albedo_residual_model.block2.res.3.weight'] = src_model['ares2.res.3.weight']
    dst_model['albedo_residual_model.block2.res.3.bias'] = src_model['ares2.res.3.bias']
    dst_model['albedo_residual_model.block2.res.3.running_mean'] = src_model['ares2.res.3.running_mean']
    dst_model['albedo_residual_model.block2.res.3.running_var'] = src_model['ares2.res.3.running_var']
    dst_model['albedo_residual_model.block2.res.5.weight'] = src_model['ares2.res.5.weight']
    dst_model['albedo_residual_model.block2.res.5.bias'] = src_model['ares2.res.5.bias']
    dst_model['albedo_residual_model.block3.res.0.weight'] = src_model['ares3.res.0.weight']
    dst_model['albedo_residual_model.block3.res.0.bias'] = src_model['ares3.res.0.bias']
    dst_model['albedo_residual_model.block3.res.0.running_mean'] = src_model['ares3.res.0.running_mean']
    dst_model['albedo_residual_model.block3.res.0.running_var'] = src_model['ares3.res.0.running_var']
    dst_model['albedo_residual_model.block3.res.2.weight'] = src_model['ares3.res.2.weight']
    dst_model['albedo_residual_model.block3.res.2.bias'] = src_model['ares3.res.2.bias']
    dst_model['albedo_residual_model.block3.res.3.weight'] = src_model['ares3.res.3.weight']
    dst_model['albedo_residual_model.block3.res.3.bias'] = src_model['ares3.res.3.bias']
    dst_model['albedo_residual_model.block3.res.3.running_mean'] = src_model['ares3.res.3.running_mean']
    dst_model['albedo_residual_model.block3.res.3.running_var'] = src_model['ares3.res.3.running_var']
    dst_model['albedo_residual_model.block3.res.5.weight'] = src_model['ares3.res.5.weight']
    dst_model['albedo_residual_model.block3.res.5.bias'] = src_model['ares3.res.5.bias']
    dst_model['albedo_residual_model.block4.res.0.weight'] = src_model['ares4.res.0.weight']
    dst_model['albedo_residual_model.block4.res.0.bias'] = src_model['ares4.res.0.bias']
    dst_model['albedo_residual_model.block4.res.0.running_mean'] = src_model['ares4.res.0.running_mean']
    dst_model['albedo_residual_model.block4.res.0.running_var'] = src_model['ares4.res.0.running_var']
    dst_model['albedo_residual_model.block4.res.2.weight'] = src_model['ares4.res.2.weight']
    dst_model['albedo_residual_model.block4.res.2.bias'] = src_model['ares4.res.2.bias']
    dst_model['albedo_residual_model.block4.res.3.weight'] = src_model['ares4.res.3.weight']
    dst_model['albedo_residual_model.block4.res.3.bias'] = src_model['ares4.res.3.bias']
    dst_model['albedo_residual_model.block4.res.3.running_mean'] = src_model['ares4.res.3.running_mean']
    dst_model['albedo_residual_model.block4.res.3.running_var'] = src_model['ares4.res.3.running_var']
    dst_model['albedo_residual_model.block4.res.5.weight'] = src_model['ares4.res.5.weight']
    dst_model['albedo_residual_model.block4.res.5.bias'] = src_model['ares4.res.5.bias']
    dst_model['albedo_residual_model.block5.res.0.weight'] = src_model['ares5.res.0.weight']
    dst_model['albedo_residual_model.block5.res.0.bias'] = src_model['ares5.res.0.bias']
    dst_model['albedo_residual_model.block5.res.0.running_mean'] = src_model['ares5.res.0.running_mean']
    dst_model['albedo_residual_model.block5.res.0.running_var'] = src_model['ares5.res.0.running_var']
    dst_model['albedo_residual_model.block5.res.2.weight'] = src_model['ares5.res.2.weight']
    dst_model['albedo_residual_model.block5.res.2.bias'] = src_model['ares5.res.2.bias']
    dst_model['albedo_residual_model.block5.res.3.weight'] = src_model['ares5.res.3.weight']
    dst_model['albedo_residual_model.block5.res.3.bias'] = src_model['ares5.res.3.bias']
    dst_model['albedo_residual_model.block5.res.3.running_mean'] = src_model['ares5.res.3.running_mean']
    dst_model['albedo_residual_model.block5.res.3.running_var'] = src_model['ares5.res.3.running_var']
    dst_model['albedo_residual_model.block5.res.5.weight'] = src_model['ares5.res.5.weight']
    dst_model['albedo_residual_model.block5.res.5.bias'] = src_model['ares5.res.5.bias']
    dst_model['albedo_residual_model.bn1.weight'] = src_model['areso.0.weight']
    dst_model['albedo_residual_model.bn1.bias'] = src_model['areso.0.bias']
    dst_model['albedo_residual_model.bn1.running_mean'] = src_model['areso.0.running_mean']
    dst_model['albedo_residual_model.bn1.running_var'] = src_model['areso.0.running_var']
    dst_model['albedo_gen_model.conv1.0.weight'] = src_model['aconv1.conv.0.weight']
    dst_model['albedo_gen_model.conv1.0.bias'] = src_model['aconv1.conv.0.bias']
    dst_model['albedo_gen_model.conv1.1.weight'] = src_model['aconv1.conv.1.weight']
    dst_model['albedo_gen_model.conv1.1.bias'] = src_model['aconv1.conv.1.bias']
    dst_model['albedo_gen_model.conv1.1.running_mean'] = src_model['aconv1.conv.1.running_mean']
    dst_model['albedo_gen_model.conv1.1.running_var'] = src_model['aconv1.conv.1.running_var']
    dst_model['albedo_gen_model.conv2.0.weight'] = src_model['aconv2.conv.0.weight']
    dst_model['albedo_gen_model.conv2.0.bias'] = src_model['aconv2.conv.0.bias']
    dst_model['albedo_gen_model.conv2.1.weight'] = src_model['aconv2.conv.1.weight']
    dst_model['albedo_gen_model.conv2.1.bias'] = src_model['aconv2.conv.1.bias']
    dst_model['albedo_gen_model.conv2.1.running_mean'] = src_model['aconv2.conv.1.running_mean']
    dst_model['albedo_gen_model.conv2.1.running_var'] = src_model['aconv2.conv.1.running_var']
    dst_model['albedo_gen_model.conv3.weight'] = src_model['aout.weight']
    dst_model['albedo_gen_model.conv3.bias'] = src_model['aout.bias']
    dst_model['light_estimator_model.conv1.0.weight'] = src_model['lconv.conv.0.weight']
    dst_model['light_estimator_model.conv1.0.bias'] = src_model['lconv.conv.0.bias']
    dst_model['light_estimator_model.conv1.1.weight'] = src_model['lconv.conv.1.weight']
    dst_model['light_estimator_model.conv1.1.bias'] = src_model['lconv.conv.1.bias']
    dst_model['light_estimator_model.conv1.1.running_mean'] = src_model['lconv.conv.1.running_mean']
    dst_model['light_estimator_model.conv1.1.running_var'] = src_model['lconv.conv.1.running_var']
    dst_model['light_estimator_model.fc.weight'] = src_model['lout.weight']
    dst_model['light_estimator_model.fc.bias'] = src_model['lout.bias']
    return dst_model


def weights_init(m):
    if isinstance(m, Conv2d) or isinstance(m, Conv1d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, Linear):
        init.normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


