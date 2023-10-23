# -*- coding: UTF-8 -*- 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter.messagebox
import tkinter.simpledialog
import tkinter.filedialog
import time
import os
import math
from enum import Enum

import torch.nn as nn
import torch

#import sv_ttk


dataset_path="./Dataset/"
lib_path="./Libs/"

single_face_filter_factorw=0.12
single_face_filter_factorh=0.12
vedio_faces_filter_factorw=0.1
vedio_faces_filter_factorh=0.1
image_faces_filter_factorw=0.1
image_faces_filter_factorh=0.1
dataset_width=48
dataset_height=48

sys_state=0
sys_state_pre=0
time_stamp=0
kernel_index=0
kernel_list=[]
input_count=0
input_name=[]
FaceMat = []
FaceLabel=[]
name_list=[]
picture_path=[]
picture_temp=[]
picture_path_temp=[]
face_detecter = cv2.CascadeClassifier(lib_path+"haarcascade_frontalface_default.xml")
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
#for filename in os.listdir('./Dataset/chenjiwei'):
faces_addr = read_directory('./Dataset/litingfeng')
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
fig, axes = plt.subplots(5,10
                       #,figsize=(20,10)
                       ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
                       )
# 图片x行y列，画布x宽y高
# 填充图像
'''
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i],cmap="gray")
plt.show()
'''

def face_detection(image):
	# 转成灰度图像
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
  faces = face_detecter.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=5)
  return faces

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        lib_path+"simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def splitFrame(ix,iy,ih,iw,frame_input):
  if ix>=10:
    oxl=ix-10
  else:
    oxl=ix
  if iy>=10:
    oyl=iy-10
  else:
    oyl=iy
  if ix+iw+10<frame_input.shape[1]:
    oxh=ix+iw+10
  else:
    oxh=frame_input.shape[1]
  oyh=iy+ih
  return frame_input[oyl:oyh,oxl:oxh]

def recordImage():
  global image_faces_filterw
  global image_faces_filterh
  global picture_temp
  global sys_state
  frame=picture_temp
  frame_temp=frame
  image_faces_filterw=image_faces_filter_factorw*frame.shape[1]
  image_faces_filterh=image_faces_filter_factorw*frame.shape[0]
  face_location=face_detection(frame_temp)#查找人脸位置
  valid_face_count=0
  if len(face_location)>0:
    for x, y, w, h in face_location:
      if w>image_faces_filterw or h>image_faces_filterh:
        valid_face_count+=1#有效人脸数
        face_temp=splitFrame(x, y, w, h,frame_temp)
        face_temp = cv2.resize(face_temp,(dataset_width,dataset_height))
        face_temp = cv2.cvtColor(face_temp, cv2.COLOR_BGR2GRAY)
    if valid_face_count==1:
      correct_name=[]
      correct_name.append(str("litingfeng1"))#输入正确姓名
      if not correct_name is None:
        if not os.path.exists(dataset_path+correct_name[0]):
          os.makedirs(dataset_path+correct_name[0])
        filename_list = os.listdir(dataset_path+correct_name[0])#扩充没有数据集
        index_list=[]
        for name in filename_list:
          index_list.append(int(name[1:4]))
        filename="s"+str(len(index_list)+1).zfill(3)+".png"
        cv2.imwrite(dataset_path+correct_name[0]+"/"+filename,face_temp)#扩充已有数据集
        return 1
      else:
        return 0
    else:
      #tkinter.messagebox.showinfo( "提示", "无效输入，未识别到人脸或识别到多张人脸")
      return 0
for faces_addr in faces:
    picture_temp=cv2.imread(faces_addr,cv2.IMREAD_UNCHANGED)
    recordImage()