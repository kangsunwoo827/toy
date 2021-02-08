import glob
img_files = glob.glob('.\\result_10400\\*.jpg')
# for f in img_files:
#     print(f)
# import os
# path='result_10400'
# # for f in img_files:
# #     os.rename(f, os.path.join(path, os.path.basename(f)[:-4]+'.jpg' ))
import cv2
from time import sleep
cnt = len(img_files)
idx = 0

img_array = []
l=list(range(30))+[40,50,60,70,80,90,99]
for i in l:
    idx = i+1
    img = cv2.imread('result_10400\sunwoo_image_{}.jpg'.format(idx))
    img_resize = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    height, width, layers = img_resize.shape
    size = (width,height)
    img_array.append(img_resize)

    # if img is None: # 이미지가 없는 경우
    #     print('Image load failed!')
    #     break

    # cv2.imshow('image', img_resize)
    # if cv2.waitKey(1000) >= 0: # 1초 동안 사진보여주는데 만약에 키보드 입력이 있으면 종료
    #     break
    # print(idx)

    # 사진을 다 보면 첫번째 사진으로 돌아감
    # idx += 1
    # if idx >= cnt:
    #     idx = 0
 
out = cv2.VideoWriter('project_5초.avi',cv2.VideoWriter_fourcc(*'DIVX'), 6, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
