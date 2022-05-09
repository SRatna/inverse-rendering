#!/usr/bin/env python
# coding: utf-8
from matplotlib import pyplot as plt
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import pandas as pd
from log import telegram_logger as tl

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


path = './data/celeba/train.csv'
out_path = '/work/ws-tmp/g051151-sfsnet/masked_celeba'

faces = list(pd.read_csv(path)['face'])
tl('mask gen started')
failed = 0
for (i, face) in enumerate(faces):
    image = cv2.imread(face)[40:, :]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        failed += 1
        print('failed', face)
        continue
    rect = rects.pop()
    mask = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    left = np.array([[shape[0][0], shape[18][1]]])
    right = np.array([[shape[16][0], shape[25][1]]])
    eye_len = int(np.linalg.norm(shape[22] - shape[26]) * 0.3)
    left_eye_top = np.array([[shape[19][0], shape[19][1] - eye_len]])
    right_eye_top = np.array([[shape[24][0], shape[24][1] - eye_len]])
    # ns = np.concatenate((shape[:17], np.flip(shape[17:27], 0)))
    ns = np.concatenate((left, shape[:17], right, right_eye_top, left_eye_top))
    
    cv2.fillPoly(mask, [ns.reshape((-1, 1, 2))], 255)
    plt.imsave(f'{out_path}/{i}_mask.jpg', mask, cmap='gray')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imsave(f'{out_path}/{i}_face.jpg', image)
    if i % 10000 == 0: tl(f'done till: {i}')
tl('mask gen done')
tl(f'total fail count: {failed}')
