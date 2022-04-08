import os
import numpy as np
import cv2
import imutils
import time
import random


def random_img(path):
  l = os.listdir(path)
  ll = l[random.randint(0, len(l) - 1)]
  return cv2.imread('{}{}'.format(path, ll), cv2.IMREAD_UNCHANGED)


def random_size(img, s_min=0.8, s_max=1.3):
  out = imutils.resize(img, width = int(img.shape[1] * random.uniform(s_min, s_max)))
  return out


def adjust_gamma(img, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(img, table)


def random_position(x_l, y_l):
  return random.randint(x_l[0], x_l[1]), random.randint(y_l[0], y_l[1])


def combine_imgs(img1, img2, mask, x, y):
  mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  h1, w1 = img1.shape[:2]
  h2, w2 = img2.shape[:2]
  x11, x12 = np.clip(x - w2 // 2, 0, w1 - 1), np.clip(x + w2 // 2, 0, w1 - 1)
  y11, y12 = np.clip(y - h2 // 2, 0, h1 - 1), np.clip(y + h2 // 2, 0, h1 - 1)
  x21 = x11 - (x - w2 // 2)
  y21 = y11 - (y - h2 // 2)
  x22 = np.clip(x21 + x12 - x11, 0, w2)
  y22 = np.clip(y21 + y12 - y11, 0, h2)
  out = img1.copy()

  alpha = mask[y21 : y22, x21 : x22].astype(float) / 255
  foreground = cv2.multiply(alpha, img2[y21 : y22, x21 : x22].astype(float))
  background = cv2.multiply(1.0 - alpha, out[y11 : y12, x11 : x12].astype(float))
  out[y11 : y12, x11 : x12] = cv2.add(foreground, background)
  return out


def combine_masks(img1, mask, x, y):
  h1, w1 = img1.shape[:2]
  h2, w2 = mask.shape[:2]
  x11, x12 = np.clip(x - w2 // 2, 0, w1 - 1), np.clip(x + w2 // 2, 0, w1 - 1)
  y11, y12 = np.clip(y - h2 // 2, 0, h1 - 1), np.clip(y + h2 // 2, 0, h1 - 1)
  x21 = x11 - (x - w2 // 2)
  y21 = y11 - (y - h2 // 2)
  x22 = np.clip(x21 + x12 - x11, 0, w2)
  y22 = np.clip(y21 + y12 - y11, 0, h2)
  out = img1.copy()


  out[y11 : y12, x11 : x12] = cv2.bitwise_or(out[y11 : y12, x11 : x12], mask[y21 : y22, x21 : x22])
  return out


def generate(obj_path, back_path, output_path,
             result_count=1000,
             result_shape_start=(4000, 4000),
             result_shape=(1000, 1000),
             obj_num=(5, 15), obj_size=(0.7, 1.3), obj_gamma=(0.9, 1.1), obj_rotation=(-5, 5),
             back_num=(20, 40), back_size=(0.9, 1.3), back_gamma=(0.9, 1.1), back_rotation=(-5, 5)):

    images_output_path = '{}/images/'.format(output_path)
    annotation_output_path = '{}/annotation/'.format(output_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(images_output_path):
        os.makedirs(images_output_path)

    if not os.path.exists(annotation_output_path):
        os.makedirs(annotation_output_path)
        
    while len(os.listdir(images_output_path)) < result_count:
        result = cv2.bitwise_not(np.zeros((result_shape[1], result_shape[0], 3), np.uint8))
        result_mask = np.zeros((result_shape_start[1], result_shape_start[0]), np.uint8)

        ''' Background '''
        for back_count in range(random.randint(back_num[0], back_num[1])):
            back_img = random_img(back_path)
            back_img = random_size(back_img, back_size[0], back_size[1])
            
            if random.randint(0, 1) == 1:
              back_img = cv2.flip(back_img, random.randint(-1, 1))
              
            back_img = imutils.rotate(back_img, random.randint(back_rotation[0], back_rotation[1]))
            gamma = random.uniform(back_gamma[0], back_gamma[1])
            back_img = adjust_gamma(back_img, gamma)

            o_x, o_y = random_position((0, result_shape[0]),
                                       (0, result_shape[1]))

            result = combine_imgs(result, back_img[:,:,:3], back_img[:,:,3], o_x, o_y)

        ''' Objects'''
        for obj_count in range(random.randint(obj_num[0], obj_num[1])):
            obj_img = random_img(obj_path)
            obj_img = random_size(obj_img, obj_size[0], obj_size[1])
            
            if random.randint(0, 1) == 1:
              obj_img = cv2.flip(obj_img, random.randint(-1, 1))
              
            obj_img = imutils.rotate(obj_img, random.randint(obj_rotation[0], obj_rotation[1]))
            gamma = random.uniform(obj_gamma[0], obj_gamma[1])
            obj_img = adjust_gamma(obj_img, gamma)

            o_x, o_y = random_position((0, result_shape[0]),
                                       (0, result_shape[1]))

            result = combine_imgs(result, obj_img[:,:,:3], obj_img[:,:,3], o_x, o_y)
            result_mask = combine_masks(result_mask, obj_img[:,:,3], o_x, o_y)

        name = str(time.time()).split('.')
        name = '{}.{}'.format(''.join(name[:-1]), name[-1])

        result = cv2.resize(result, result_shape)
        result_mask = cv2.resize(result_mask, result_shape)
        
        cv2.imwrite('{}{}.jpg'.format(images_output_path, name), result)
        cv2.imwrite('{}{}_cancer_0.png'.format(annotation_output_path, name), result_mask)
            
        
    ##    result = cv2.resize(result, (1000, 1000), interpolation = cv2.INTER_AREA)
    ##    cv2.imshow('result', result)
    ##    result_mask = cv2.resize(result_mask, (1000, 1000), interpolation = cv2.INTER_AREA)
    ##    cv2.imshow('result_mask', result_mask)
    ##    cv2.waitKey(0)
    ##    cv2.destroyAllWindows()





obj_path = 'output_objects/'
back_path = 'output_objects/'
output_path = 'results/dataset_1/'

generate(obj_path, back_path, output_path, result_count=2000)
