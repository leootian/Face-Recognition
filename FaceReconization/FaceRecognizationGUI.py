# -*- coding: UTF-8 -*- 
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

class sys_state_enum(Enum):
  RECOGNIZE_VEDIO=0
  RECORD_FACE=1
  TRAIN_MODEL=2
  LOAD_IMAGE=3
  RECOGNIZE_IMAGE=4
  RECORD_IMAGE=5
  SYS_QUIT=6

class recognizer_kernel_enum(Enum):
  CASCADE_EIGENFACE=0
  FAST_RCNN=1
  

cap = cv2.VideoCapture(0)#创建摄像头对象
face_detecter = cv2.CascadeClassifier(lib_path+"haarcascade_frontalface_default.xml")
expression_model = torch.load(lib_path+"model.pkl", map_location=torch.device('cpu'))
expression_model = expression_model.cpu()

################################################################表情识别库#################################################################

def gaussian_weights_init(m):
    
    '''
    Generate numbers with gaussian distribution
    '''
    
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)

class ExpressionCNN(nn.Module):
    
    '''
    Model for expression recognition
    '''
    
    def __init__(self):
        
        super(ExpressionCNN, self).__init__()
        
        # layer1(conv + relu + pool)
        # input:(batch_size, 1, 48, 48), output(batch_size, 64, 24, 24)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # layer2(conv + relu + pool)
        # input:(batch_size, 64, 24, 24), output(batch_size, 128, 12, 12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # layer3(conv + relu + pool)
        # input: (batch_size, 128, 12, 12), output: (batch_size, 256, 6, 6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Initialization
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)
        
        self.fullc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256*6*6, 4096),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.RReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.RReLU(inplace=True),
            nn.Linear(256, 7)
        )
        
    def forward(self, x):
        
        '''
        Forward propagation
        '''
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)  # 数据扁平化
        y = self.fullc(x)

        return y
    
expression_dict = {0:"生气", 1:"厌恶", 2:"恐慌", 3:"高兴", 4:"难过", 5:"惊讶", 6:"中立"}

def expression_recognition(image,exp_model):
    model=exp_model
    model.eval()
    
    outputs = model.forward(image)

    max_index = torch.argmax(outputs.float())
    max_index = max_index.numpy()
    max_index = max_index.reshape(1,1)
    max_index = max_index[0][0]
    # print("Result:", max_index)
    # print("Expression:", expression_dict[max_index])
    
    return expression_dict[max_index]

########################################################################################################################################



##############################################################GUI功能函数################################################################

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

def display_result(result_text,text_font,text_color,location,pic):
  cv2.putText(pic,result_text,location,text_font,1,text_color,2)

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

def recognizeExpression(image):
  #image=cv2.resize(image,(48,48))
  #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_hist = cv2.equalizeHist(image)
  image_normalized = image_hist.reshape(1, 48, 48) / 255.0
  image_tensor = torch.from_numpy(image_normalized)
  image_tensor = image_tensor.type('torch.FloatTensor')
  image_tensor = image_tensor.unsqueeze(0)

  return expression_recognition(image_tensor,expression_model)

def recognizeFace(image):
  label, confidence = recognizer.predict(image)
  return [label, confidence]

def recognizeVedio():
  global name_list
  ref,frame=cap.read()
  frame = cv2.flip(frame, 1) #摄像头翻转
  frame_temp=frame
  vedio_faces_filterw=vedio_faces_filter_factorw*frame.shape[1]
  vedio_faces_filterh=vedio_faces_filter_factorh*frame.shape[0]
  face_location=face_detection(frame_temp)#获取人脸位置
  if len(face_location)>0:
    for x, y, w, h in face_location:
      if w>vedio_faces_filterw or h>vedio_faces_filterh:
        # 在原图像上绘制矩形标识
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)
        face_temp=splitFrame(x, y, w, h,frame_temp)
        face_temp = cv2.recognizer(face_temp,(dataset_width,dataset_height))
        face_temp=cv2.cvtColor(face_temp, cv2.COLOR_BGR2GRAY)
        str_exp=recognizeExpression(face_temp)#表情识别
        [label,confidence]=recognizeFace(face_temp)#人脸识别
        frame=cv2AddChineseText(frame, str(name_list[label-1])+"/"+str_exp, (x,y), textColor=(0, 255, 255), textSize=30)#结果输出
  cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
  pilImage=Image.fromarray(cvimage)
  pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
  tkImage =  ImageTk.PhotoImage(image=pilImage)
  return tkImage

def recordFace():
  global input_count
  global input_name
  global dataset_width
  global dataset_height
  ref,frame=cap.read()
  frame = cv2.flip(frame, 1) #摄像头翻转
  frame_temp=frame
  face_input_filterw=single_face_filter_factorw*frame.shape[1]
  face_input_filterh=single_face_filter_factorh*frame.shape[0]
  face_location=face_detection(frame_temp)
  if len(face_location)==1 and time.time()-time_stamp>input_count:
    if face_location[0][2]>face_input_filterw or face_location[0][3]>face_input_filterh:
      input_count+=1
      for x, y, w, h in face_location:
        frame_temp=splitFrame(x, y, w, h,frame_temp)
        frame_temp = cv2.resize(frame_temp,(dataset_width,dataset_height))
        frame_temp=cv2.cvtColor(frame_temp, cv2.COLOR_BGR2GRAY)
      filename="s"+str(input_count).zfill(3)+".png"
      cv2.imwrite(dataset_path+input_name+"/"+filename,frame_temp)
  for x, y, w, h in face_location:
      # 在原图像上绘制矩形标识
      cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)
  cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
  pilImage=Image.fromarray(cvimage)
  pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
  tkImage =  ImageTk.PhotoImage(image=pilImage)
  return tkImage

def recognizeImage():
  global image_faces_filterw
  global image_faces_filterh
  global name_list
  global picture_temp
  pic_msg=[]
  frame=picture_temp
  frame_temp=frame
  image_faces_filterw=image_faces_filter_factorw*frame.shape[1]
  image_faces_filterh=image_faces_filter_factorw*frame.shape[0]
  face_location=face_detection(frame_temp)
  if len(face_location)>0:
    for x, y, w, h in face_location:
      if w>image_faces_filterw or h>image_faces_filterh:
        # 在原图像上绘制矩形标识
        cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)
        face_temp=splitFrame(x, y, w, h,frame_temp)
        face_temp = cv2.resize(face_temp,(dataset_width,dataset_height))
        face_temp = cv2.cvtColor(face_temp, cv2.COLOR_BGR2GRAY)
        str_exp=recognizeExpression(face_temp)
        [label,confidence]=recognizeFace(face_temp)
        frame=cv2AddChineseText(frame, str(name_list[label-1])+"/"+str_exp, (x,y), textColor=(0, 255, 255), textSize=30)
  cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
  pilImage=Image.fromarray(cvimage)
  if frame.shape[0]/frame.shape[1]>image_height/image_width:
    target_width=math.floor(frame.shape[1]*image_height/frame.shape[0])
    target_height=image_height
  else:
    target_width=image_width
    target_height=math.floor(frame.shape[0]*image_width/frame.shape[1])
  pilImage = pilImage.resize((target_width, target_height),Image.ANTIALIAS)
  pic_x=(image_width-target_width)/2
  pic_y=(image_height-target_height)/2
  tkImage =  ImageTk.PhotoImage(image=pilImage)
  pic_msg.append(tkImage)
  pic_msg.append(pic_x)
  pic_msg.append(pic_y)
  return pic_msg

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
      correct_name = tkinter.simpledialog.askstring(title = '提示',prompt='请输入正确的姓名：')#输入正确姓名
      if not correct_name is None:
        if not os.path.exists(dataset_path+correct_name):
          os.makedirs(dataset_path+correct_name)
        filename_list = os.listdir(dataset_path+correct_name)#扩充没有数据集
        index_list=[]
        for name in filename_list:
          index_list.append(int(name[1:4]))
        filename="s"+str(max(index_list)+1).zfill(3)+".png"
        cv2.imwrite(dataset_path+correct_name+"/"+filename,face_temp)#扩充已有数据集
        return 1
      else:
        return 0
    else:
      tkinter.messagebox.showinfo( "提示", "无效输入，未识别到人脸或识别到多张人脸")
      return 0

def recordFacesCallBack():
  global sys_state
  global time_stamp
  global input_count
  global input_name
  if sys_state!=sys_state_enum.RECORD_FACE:
    input_count=0
    input_name=[]
    input_name = tkinter.simpledialog.askstring(title = '提示',prompt='请输入姓名的汉语拼音：')
    if len(input_name)>0:
      sys_state=sys_state_enum.RECORD_FACE
      if not os.path.exists(dataset_path+input_name):
        os.makedirs(dataset_path+input_name)
      tkinter.messagebox.showinfo( "提示", "开始录入人脸信息，请正对摄像头并略微转动头部")
      time_stamp=time.time()

def is_Chinese(word):
  for ch in word:
    if '\u4e00' <= ch <= '\u9fff':
        return True
  return False

def chooseImageCallBack():
  global picture_path
  global picture_path_temp
  global sys_state
  picture_path=[]
  picture_path = tkinter.filedialog.askopenfilename()  # 选择目录，返回目录名
  if len(picture_path)>0:
    if is_Chinese(picture_path):
      tkinter.messagebox.showinfo( "提示", "cv2不支持中文路径，请选择英文路径")
    else:
      picture_path_temp=picture_path
      sys_state=sys_state_enum.LOAD_IMAGE

def recognizeVedioCallBack():
  global sys_state
  sys_state=sys_state_enum.RECOGNIZE_VEDIO

def recordImageCallBack():
  global sys_state
  if sys_state==sys_state_enum.RECOGNIZE_IMAGE:
    tkinter.messagebox.showinfo( "警告", "此操作会扩充训练集")
    sys_state=sys_state_enum.RECORD_IMAGE
  else:
    tkinter.messagebox.showinfo( "提示", "仅支持图片的勘误")

def kernelChooseCallBack(*args):
  global kernel_index
  kernel_index=kernel_list.index(kernel_mode.get())

def _quit():
  global sys_state
  sys_state=sys_state_enum.SYS_QUIT

########################################################################################################################################



###############################################################界面初始化################################################################

top = tk.Tk()
#sv_ttk.set_theme("dark")
top.protocol("WM_DELETE_WINDOW", _quit)
top.title('模式识别课程作业-人脸识别demo')
top.geometry('1600x800')
image_width = 800
image_height = 600
canvas_input = Canvas(top,bg = 'black',width = image_width,height = image_height )#绘制输入
Label(top,text = '图像输入',font = ("黑体",14),width =15,height = 1).place(x =700,y = 20,anchor = 'nw')
canvas_input.place(x = 380,y = 50)


Button1 = Button(top, text ="录入人脸", command = recordFacesCallBack,relief=RAISED,height=2,width=20)
Button1.place(x=250, y=680)
Button2 = Button(top, text ="载入图片", command = chooseImageCallBack,relief=RAISED,height=2,width=20)
Button2.place(x=550, y=680)
Button2 = Button(top, text ="摄像头实时识别", command = recognizeVedioCallBack,relief=RAISED,height=2,width=20)
Button2.place(x=850, y=680)
Button3 = Button(top, text ="手动勘误", command = recordImageCallBack,relief=RAISED,height=2,width=20)
Button3.place(x=1150, y=680)
for item in recognizer_kernel_enum:
  kernel_list.append(item.name)
kernel_mode=tkinter.StringVar(top)
kernel_mode.set(kernel_list[0])
OptionMenu1 = OptionMenu(top, kernel_mode, *kernel_list)
OptionMenu1.place(x=100, y=50)
kernel_mode.trace("w", kernelChooseCallBack)
Label(top,text = '选择内核',font = ("黑体",10),height = 1).place(x=30, y=58)

########################################################################################################################################



#################################################################主循环##################################################################

sys_state=sys_state_enum.TRAIN_MODEL

while True:
  if sys_state==sys_state_enum.RECOGNIZE_VEDIO:
    pic = recognizeVedio()
    canvas_input.create_image(0,0,anchor = 'nw',image = pic)

  elif sys_state==sys_state_enum.RECORD_FACE:
    pic = recordFace()
    canvas_input.create_image(0,0,anchor = 'nw',image = pic)
    if time.time()>time_stamp+8:
      sys_state_pre=sys_state_enum.RECORD_FACE
      sys_state=sys_state_enum.TRAIN_MODEL
      tkinter.messagebox.showinfo( "提示", "录入成功！")
      input_name=[]

  elif sys_state==sys_state_enum.TRAIN_MODEL:
    train_msg=Label(top,text = '模型训练中。。。',font = ("黑体",10),width =15,height = 1)
    train_msg.place(x =700,y = 800,anchor = 'nw')
    name_list = os.listdir(dataset_path)
    i=0
    if len(name_list)>0:
      FaceMat=[]
      FaceLabel=[]
      for name in name_list:
        i+=1
        for dir_item in os.listdir(dataset_path+name):
          FaceMat.append(cv2.imread(dataset_path+name+"/"+dir_item, cv2.IMREAD_GRAYSCALE))
          FaceLabel.append(i)
      recognizer = cv2.face.EigenFaceRecognizer_create()
      recognizer.train(FaceMat, np.array(FaceLabel))
      train_msg.destroy()
      if sys_state_pre==sys_state_enum.RECORD_IMAGE:
        sys_state=sys_state_enum.RECOGNIZE_IMAGE
      else:
        sys_state=sys_state_enum.RECOGNIZE_VEDIO
    else:
      tkinter.messagebox.showinfo( "提示", "缺少训练集，无法训练模型，请录入人脸！")
      recordFacesCallBack()

  elif sys_state==sys_state_enum.LOAD_IMAGE:
    picture_temp=cv2.imread(picture_path,cv2.IMREAD_UNCHANGED)
    sys_state=sys_state_enum.RECOGNIZE_IMAGE

  elif sys_state==sys_state_enum.RECOGNIZE_IMAGE:
    if len(picture_path)>0:
      pic = recognizeImage()
      canvas_input.create_image(pic[1],pic[2],anchor = 'nw',image = pic[0])
      picture_path=[]

  elif sys_state==sys_state_enum.RECORD_IMAGE:
    picture_temp=cv2.imread(picture_path_temp,cv2.IMREAD_UNCHANGED)
    if recordImage():
      picture_path=picture_path_temp
      sys_state_pre=sys_state_enum.RECORD_IMAGE
      sys_state=sys_state_enum.TRAIN_MODEL
    else:
      sys_state=sys_state_enum.RECOGNIZE_IMAGE

  else:
    top.quit()
    cap.release()
    top.destroy()
    break

  top.update()
  top.after(25)