#!/usr/bin/env python
# coding: utf-8

# In[9]:


import glob
from matplotlib import pyplot as plt
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import pandas as pd
from PIL import Image
import scipy.io
from log import telegram_logger as tl

tl('photoface started')

# In[40]:


path = '/work/ws-tmp/g051151-sfsnet/g051151-sfsnet-1633129803/Photoface_dist/PhotofaceDB/'
out_path = '/work/ws-tmp/g051151-sfsnet/photoface-padded'


# In[11]:


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# In[36]:


def crop_and_save(img, rect, face_path, is_gray=False):
    tl_corner = rect.tl_corner()
    br_corner = rect.br_corner()
    tl_corner_x = tl_corner.x if tl_corner.x > 0 else 0
    img = img[tl_corner.y:br_corner.y, tl_corner_x:br_corner.x]
    s = max(img.shape[0:2])
    s += np.int(s/3)
    #Creating a dark square with NUMPY  
    face = np.zeros((s,s,3), np.uint8)
    if is_gray: face = np.zeros((s,s), np.uint8)
    #Getting the centering position
    ax, ay = (s - img.shape[1])//2,(s - img.shape[0])//2
    #Pasting the 'image' in a centering position
    face[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
    Image.fromarray(face).resize(size=(128, 128), resample=Image.ANTIALIAS).save(face_path)


# In[21]:


def get_mask(img, rect):
    mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    shape = predictor(img, rect)
    shape = face_utils.shape_to_np(shape)

    left = np.array([[shape[0][0], shape[18][1]]])
    right = np.array([[shape[16][0], shape[25][1]]])
    eye_len = int(np.linalg.norm(shape[22] - shape[26]) * 0.3)
    left_eye_top = np.array([[shape[19][0], shape[19][1] - eye_len]])
    right_eye_top = np.array([[shape[24][0], shape[24][1] - eye_len]])
    # ns = np.concatenate((shape[:17], np.flip(shape[17:27], 0)))
    ns = np.concatenate((left, shape[:17], right, right_eye_top, left_eye_top))
    cv2.fillPoly(mask, [ns.reshape((-1, 1, 2))], 255)
    return mask


# In[24]:


def get_normal(img_path):
    normal_path = img_path.replace(img_path.split('/')[-1], 'sn.mat')
    normal_mat = scipy.io.loadmat(normal_path)
    X, Y, Z = normal_mat['px'], normal_mat['py'], normal_mat['pz']
    normal_img = np.concatenate((X[:,:,np.newaxis], Y[:,:,np.newaxis], Z[:,:,np.newaxis]), axis=2)
    return (((normal_img+1)/2) * 255).astype(np.uint8)


# In[34]:


def get_albedo(img_path):
    albedo_path = img_path.replace(img_path.split('/')[-1], 'albedo.mat')
    albedo_mat = scipy.io.loadmat(albedo_path)
    return (albedo_mat['a']).astype(np.uint8)


# In[39]:


face_paths = glob.glob(path + '*/*/*.bmp')
tl(f'Total faces: {len(face_paths)}')


# In[42]:


failed = 0


# In[43]:


for i, img_path in enumerate(glob.glob(path + '*/*/*.bmp')):
    image = cv2.imread(img_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    albedo = get_albedo(img_path)
    rects = detector(albedo, 1)
    if len(rects) == 0:
        print('failed', img_path)
        failed += 1
        continue
    rect = rects.pop()
    mask = get_mask(albedo, rect)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normal = get_normal(img_path)
    try:
        crop_and_save(image, rect, f'{out_path}/{i}_face.png')
        crop_and_save(normal, rect, f'{out_path}/{i}_normal.png')
        crop_and_save(albedo, rect, f'{out_path}/{i}_albedo.png', is_gray=True)
        crop_and_save(mask, rect, f'{out_path}/{i}_mask.png', is_gray=True)
    except ValueError as e:
        print('Value error', img_path)
        continue
    if i % 2000 == 0: tl(f'done till: {i}')


# In[44]:


tl(f'Total failed count: {failed}')

tl('Photo face done')
# In[ ]:




