# -*- coding: utf-8 -*-

from os.path import splitext
from PIL import Image
import numpy as np
import os
import urllib
import gzip
import struct
import matplotlib.pyplot as plt
import time

tStart = time.time()#計時開始

dataPath='~/user_data/cr/nistsd19v2/by_class'   #sd-19 data path
#image(PngImageFile mode=RGB size=128x128 (0, 255))  resized into 28x28 with grayscale value between 0 and 254
rows, cols = 28, 28

def prepareData(dataPath,rows, cols):
        #將sd-19之train_,hsf_4資料夾png分別讀至一np array並存成壓縮檔sd-19train,sd-19val 
    train_img = np.zeros((rows, cols), dtype=np.float32)  #traning image for save npz file       
    train_lbl= np.zeros(1, dtype=np.int8)  #training label
    val_img= np.zeros((rows, cols), dtype=np.float32)   #validation image for save npz file 
    val_lbl= np.zeros(1, dtype=np.uint8)    #validation label 

    def rgb2gray(rgb):  #png 3chanel-> gray 1chanel
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  #sd19白底黑字的二值(254.975,0)影像28*28，floating
        #gray = abs(int(0.2989 * r + 0.5870 * g + 0.1140 * b))  #改成同mnist,黑底灰字影像28*28，int
        return gray  
    
    #如果png影像之npz已存在
    try:
        data = np.load("train.npz") 
        train_img, train_lbl = data['train_img'], data['train_lbl'] 
        data = np.load("val.npz")
        val_img, val_lbl = data['val_img'], data['val_lbl'] 
        if train_img==[] or train_lbl==[] or val_img==[] or val_lbl==[]:
            print("+npz壓縮檔有問題，將由sd19_by_clas夾讀取，並製作npz file")  
        else:  
            print('成功讀取npz file, training image：'+ str(len(train_img))+', val_img:'+str(len(val_img)))
            return train_img, train_lbl, val_img, val_lbl
    except: 
        print("npz壓縮檔不存在，將由sd19_by_clas夾讀取，並製作npz file")   
    

        #creat np array of image and label
    
    i=0  
    pnglist = glob( dataPath+"*/train*/*.png" )   #the suggested training set for OCR studie
    for png in pnglist: 
        i +=1 
        if((i%8)==0):  #原有731668張，每8張取一個
            #if "train" in png:  #the suggested training set for OCR studies
            im1=Image.open(png) 
            im2 = im1.resize((rows, cols), Image.NEAREST)      # use nearest neighbour filter to resize the image(28*28)
            im2=np.array(im2) 
            im2=rgb2gray(im2)  #png 3chanel-> gray 1chanel 
            train_img=np.vstack((train_img,im2))  #is need:(( ));im2 加在 train_img後面
            i1=png[len(dataPath):len(dataPath)+2] #取出label like '47'   
            #train_lbl = np.vstack((train_lbl, chr(int(i1, 16))))  #16進位'47' --> 'G';;training label
            train_lbl = np.append(train_lbl, int(i1, 16)) #training label;i1 加在train_lbl後面;16進位'47'(string) 代表'G'--> 轉成10進位的71,chr(71)-->'G'
            if((i%500)==0):
                print('約已讀取：'+ str(int(i/5)) + '---' + png[len(dataPath):len(png)] + '\r', end='')
    #            print('約已讀取：'+ str(int(i/5)) + '---' + png[len(dataPath):len(png)] + '\r', end='')

    pnglist = glob( dataPath+"*/hsf_4/*.png" )  #validation data; hsf_4 standard testing set
    for png in pnglist:
        i +=1 
        if((i%8)==0): #原有82587，每8個取一個
            #elif "hsf_4" in png:    #validation data; hsf_4 standard testing set
            im1=Image.open(png) 
            im2 = im1.resize((rows, cols), Image.NEAREST)      # use nearest neighbour filter to resize the image(28*28)
            im2=np.array(im2)
            im2=rgb2gray(im2)  #png 3chanel-> gray 1chanel
            val_img=np.vstack((val_img,im2))  #validation image;im2加在val_img後面
            i1=png[len(dataPath):len(dataPath)+2] #取出label like '47'    
            #val_lbl = np.vstack((val_lbl, chr(int(i1, 16))))  #16進位'47' --> 'G'
            val_lbl = np.append(val_lbl, int(i1, 16)) #validation label;i1加在val_lbl後面;16進位'47'(string) 代表'G'--> 轉成10進位的71,chr(71)-->'G'
            if((i%500)==0):
                print('約已讀取：'+ str(int(i/5)) +'---'+png[len(dataPath):len(png)]+'\r',end='')
 
    if train_img==[] or train_lbl==[] or val_img==[] or val_lbl==[]:
        print("無資料")
        return 0
    #儲存影像,label,image mode & size
    print(len(train_lbl),train_lbl.shape, len(train_img),train_img.shape)
    #im={'mode':im2.mode,'size':im2.size}
    train_img=train_img.reshape((len(train_lbl),28, 28))
    train_img=np.delete(train_img, 0,0)  #del 空的初值
    train_lbl=np.delete(train_lbl, 0,0)  #del 空的初值
    val_img=val_img.reshape((len(val_lbl),28, 28))
    val_img=np.delete(val_img, 0,0)  #del 空的初值
    val_lbl=np.delete(val_lbl, 0,0)  #del 空的初值
    
    #mxnet label要從0開始起，-48,48-57(0-9)-->0-9,65-122(a-z)-->10-67
    train_lbl=np.array([x-48 if x < 59 else x-65+10 for x in train_lbl])
    val_lbl=np.array([x-48 if x < 59 else x-65+10 for x in val_lbl])
    #abs(val_img-254)-->改成同mnist,黑底灰字影像28*28，int;
    train_img=np.int_(abs(train_img-254))
    val_img=np.int_(abs(val_img-254))    
    
    np.savez_compressed("train.npz", train_img=train_img, train_lbl=train_lbl)    
    np.savez_compressed("val.npz", val_img=val_img, val_lbl=val_lbl) 
    #np.savez_compressed("train.npz", train_img=np.int_(abs(train_img-254)), train_lbl= np.int8(train_lbl))
    #np.savez_compressed("val.npz", val_img=np.int_(abs(val_img-254)), val_lbl=np.int8(val_lbl))
    print('成功讀取npz file, training image：'+ str(len(train_img))+', val_img1:'+str(len(val_img)))
    return train_img, train_lbl, val_img, val_lbl #sd19白底黑字的二值(254.975,0)影像28*28，floatin 
    #return np.int_(abs(train_img-254)), np.int8(train_lbl), np.int_(abs(val_img-254)), np.int8(val_lbl)   

#將sd19之png images、labe、image mode & size等匯集成list
train_img, train_lbl, val_img, val_lbl = prepareData(dataPath, rows, cols) 
   
tEnd = time.time()#計時結束
m, s = divmod((tEnd - tStart), 60)  
h, m = divmod(m, 60) 
print('')
print ("it cost: %02d:%02d:%02d" % (h, m, s))
