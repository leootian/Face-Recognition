# -*- coding: UTF-8 -*- 
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
# plt显示灰度图片
def plt_show(img):
    plt.imshow(img,cmap='gray')
    plt.show()
# 读取一个文件夹下的所有图片，输入参数是文件名，返回文件地址列表
def read_directory(directory_name):
    faces_addr = []
    for filename in os.listdir(directory_name):
        faces_addr.append(directory_name + "/" + filename)
    return faces_addr
# 读取所有人脸文件夹,保存图像地址在faces列表中
faces = []
for filename in os.listdir('./Dataset'):
    faces_addr = read_directory('./Dataset/'+str(filename))
    for addr in faces_addr:
        faces.append(addr)
# 读取图片数据,生成列表标签
images = []
labels = []
for index,face in enumerate(faces):
    # enumerate函数可以同时获得索引和值
    image = cv2.imread(face,0)
    images.append(image)
    labels.append(int(index/8+1))   
# 画出最后20组人脸图像
# 创建画布和子图对象
fig, axes = plt.subplots(5,8
                       #,figsize=(20,10)
                       ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
                       )
# 图片x行y列，画布x宽y高
# 填充图像
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i],cmap="gray")

plt.show()