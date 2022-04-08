import os
import numpy as np
import cv2
import imutils

import random

def save_transparent_img(img, mask, path):

    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    result = np.dstack([img, mask])
    cv2.imwrite(path, result)

    return result

def get_objects(path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in os.listdir(path):
        if 'mask' in file:
            img = cv2.imread('{}{}'.format(path, file))
            img_rgb = cv2.imread('{}{}.jpg'.format(path, file[:-len('_mask.jpg')]))
            name = file[:-len('_mask.jpg')]

            H, W = img.shape[:2]

            kernel = np.ones((5, 5), np.uint8)
            img = cv2.erode(img, kernel, iterations=3)

            contours, h = cv2.findContours(img[:,:,0],
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            
            cnts = imutils.grab_contours((contours, h))
            
            for contour_ind in range(len(contours)):
                if cv2.contourArea(contours[contour_ind]) > 80:
                    (x, y, w, h) = cv2.boundingRect(cnts[contour_ind])

                    img_s = np.zeros((H, W, 3), np.uint8)
                    cv2.drawContours(image=img_s,
                                     contours=contours,
                                     contourIdx=contour_ind,
                                     color=(255, 255, 255),
                                     thickness=-1,
                                     lineType=cv2.LINE_AA)

                    img_s = img_s[y : y + h, x : x + w]
                    img_s = cv2.dilate(img_s, kernel, iterations=3)
                    img_s_rgb = img_rgb[y : y + h, x : x + w].copy()               

                    save_transparent_img(img_s_rgb, img_s,
                                         '{}{}_{}.png'.format(output_path, name,
                                                              contour_ind))



def get_back(path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    kernel = np.ones((5, 5), np.uint8)
    
    for file in os.listdir(path):
        if 'mask' in file:
            img = cv2.imread('{}{}'.format(path, file))
            img_rgb = cv2.imread('{}{}.jpg'.format(path, file[:-len('_mask.jpg')]))
            name = file[:-len('_mask.jpg')]

            ret, mask = cv2.threshold(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY),
                                      230, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.medianBlur(mask, 15)
            mask = cv2.erode(mask, kernel, iterations=3)
            mask = cv2.medianBlur(mask, 15)

            img = cv2.dilate(img, kernel, iterations=6)
            img = cv2.medianBlur(img, 5)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print(mask.shape, img.shape)
            img = mask - img

            H, W = img.shape[:2]

            
            img = cv2.erode(img, kernel, iterations=10)
            img = cv2.dilate(img, kernel, iterations=6)

            contours, h = cv2.findContours(img,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            
            cnts = imutils.grab_contours((contours, h))
            
            for contour_ind in range(len(contours)):
                if cv2.contourArea(contours[contour_ind]) > 80:
                    (x, y, w, h) = cv2.boundingRect(cnts[contour_ind])

                    img_s = np.zeros((H, W, 3), np.uint8)
                    cv2.drawContours(image=img_s,
                                     contours=contours,
                                     contourIdx=contour_ind,
                                     color=(255, 255, 255),
                                     thickness=-1,
                                     lineType=cv2.LINE_AA)

                    img_s = img_s[y : y + h, x : x + w]
                    img_s = cv2.dilate(img_s, kernel, iterations=3)
                    img_s_rgb = img_rgb[y : y + h, x : x + w].copy()               

                    save_transparent_img(img_s_rgb, img_s,
                                         '{}{}_{}.png'.format(output_path, name,
                                                              contour_ind))

##                    img_s_rgb = cv2.bitwise_and(img_s_rgb, img_s)
##                    cv2.imshow('{}'.format(contour_ind), img_s_rgb)
           

##            img = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_AREA)
##            img_rgb = cv2.resize(img_rgb, (1920, 1080), interpolation = cv2.INTER_AREA)
##            mask = cv2.resize(mask, (1920, 1080), interpolation = cv2.INTER_AREA)
##            cv2.imshow('mask', img)
##            cv2.imshow('rgb', img_rgb)
##            cv2.imshow('mask1', mask)
##            cv2.waitKey(0)
##            break
##    cv2.destroyAllWindows()


path = '../tissue-segment/'
output_path = 'output_objects/'
get_objects(path, output_path)

output_path = 'output_back/'
get_back(path, output_path)
