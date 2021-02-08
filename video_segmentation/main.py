
from model import get_testing_model_resnet101
import os
import cv2
import config
import sys
import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from keras.models import load_model
import code
import copy
import scipy.ndimage as sn
from PIL import Image
from function import human_seg_combine_argmax, human_seg_combine_argmax_rgb

human_part = [0,1,2,3,4,5,6]
human_ori_part = [0,1,2,3,4,5,6]
seg_num = 7 # current model supports 7 parts only

def recover_flipping_output(oriImg, part_ori_size):
    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    return part_flip_size

def part_thresholding(seg_argmax):
    background = 0.6
    head = 0.5
    torso = 0.8 
    part_th = [background, head, torso, 0.55, 0.55, 0.55, 0.55]
    th_mask = np.zeros(seg_argmax.shape)
    for indx in range(seg_num):
        part_prediction = (seg_argmax==indx)
        part_prediction = part_prediction*part_th[indx]
        th_mask += part_prediction

    return th_mask


def process (input_image):
    oriImg = cv2.imread(input_image)
    flipImg = cv2.flip(oriImg, 1)
    oriImg = (oriImg / 256.0) - 0.5
    flipImg = (flipImg / 256.0) - 0.5
    multiplier = [x for x in config.scale_search]

    seg_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [ 0,
                0, 
                (imageToTest.shape[0] - config.stride) % config.stride,
                (imageToTest.shape[1] - config.stride) % config.stride
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))

        input_img = imageToTest[np.newaxis, ...]
        
        print( "\t[Original] Actual size fed into NN: ", input_img.shape)

        output_blobs = model.predict(input_img)

        seg = np.squeeze(output_blobs[0])
        seg = cv2.resize(seg, (0, 0), fx=config.stride, fy=config.stride,
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)


        if m==0:
            segmap_scale1 = seg
        elif m==1:
            segmap_scale2 = seg         
        elif m==2:
            segmap_scale3 = seg
        elif m==3:
            segmap_scale4 = seg


    # flipping
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pad = [ 0,
                0, 
                (imageToTest.shape[0] - config.stride) % config.stride,
                (imageToTest.shape[1] - config.stride) % config.stride
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        input_img = imageToTest[np.newaxis, ...]
        print( "\t[Flipping] Actual size fed into NN: ", input_img.shape)
        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
       
        seg = np.squeeze(output_blobs[0])
        seg = cv2.resize(seg, (0, 0), fx=config.stride, fy=config.stride,
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        seg_recover = recover_flipping_output(oriImg, seg)

        if m==0:
            segmap_scale5 = seg_recover
        elif m==1:
            segmap_scale6 = seg_recover         
        elif m==2:
            segmap_scale7 = seg_recover
        elif m==3:
            segmap_scale8 = seg_recover
    
    segmap_a = np.maximum(segmap_scale1,segmap_scale2)
    segmap_b = np.maximum(segmap_scale4,segmap_scale3)
    segmap_c = np.maximum(segmap_scale5,segmap_scale6)
    segmap_d = np.maximum(segmap_scale7,segmap_scale8)
    seg_ori = np.maximum(segmap_a, segmap_b)
    seg_flip = np.maximum(segmap_c, segmap_d)
    seg_avg = np.maximum(seg_ori, seg_flip)

    return seg_avg


if  __name__ == '__main__':
    keras_weights_file='cdcl_pascal_model\model_simulated_RGB_mgpu_scaling_append.0024.h5'
    model = get_testing_model_resnet101() 
    model.load_weights(keras_weights_file)
    # scale_list = []

    # for item in args.scale:
    #     scale_list.append(float(item))

    # config.scale_search = scale_list

    input_folder='input_image'
    output_folder='output_image'
    # generate image with body parts
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            print(input_folder+'/'+filename)
            seg = process(input_folder+'/'+filename)
            seg_argmax = np.argmax(seg, axis=-1)
            seg_max = np.max(seg, axis=-1)
            th_mask = part_thresholding(seg_argmax)
            seg_max_thres = (seg_max > 0.1).astype(np.uint8)
            seg_argmax *= seg_max_thres
            seg_canvas = human_seg_combine_argmax_rgb(seg_argmax)
            cur_canvas = cv2.imread(input_folder+'/'+filename)
            canvas = cv2.addWeighted(seg_canvas, 0.6, cur_canvas, 0.4, 0)
            filename = '%s/%s.jpg'%(output_folder,'seg_'+filename)
            cv2.imwrite(filename, canvas) 