#!/usr/bin/env python
# coding: utf-8

# In[102]:


import os
from collections import defaultdict
import cv2, os
import numpy as np
import sklearn.cluster as skc
import matplotlib.pyplot as plt
from PIL import Image
import math
path='/ai/zhuhong/jupyter_dir/pix/pre_rotate_image_gen_bbox_nodir2'
#path='/ai/zhuhong/jupyter_dir/pix/pre_rotate_image_gen_bbox'
#path='/ai/zhuhong/jupyter_dir/pix/result/tatami_version_1pix2pix/test_latest/images/'



def show_image(data,p,save=False,name="ss.jpg",ax=None):
    sign_create_ax = True if not ax else False
    if sign_create_ax:
        fig, ax = plt.subplots()
        
    #print("data.shape:",data.shape)
    ax.imshow(data)
    p1=[]
    p2=[]
    for key, data in p.items():
        for value in data:
            p1.append(value[0])
            p2.append(value[1])
    ax.scatter(np.array(p1),np.array(p2),c='r') 
    #ax.axis('off')
    plt.show()
#     if sign_create_ax:
#         plt.show()
    if save:
        fig.savefig(name)
        plt.close()
#     if sign_create_ax:
#         fig.clf()
    return fig



def get_image_bbox(image,zonedicts,ax = None):

    data= Image.fromarray(image, mode='RGB')
#     data = Image.open(file).convert('RGB')
    w,h=data.size
#     image = np.array(data)
    thread = 20
    boxes = defaultdict(list)
    p=defaultdict(list)
    boxes_origin = defaultdict(list)
    directions = defaultdict(list)
    for key, values in zonedicts.items():
        Binary = np.zeros(image.shape[:2])
        
        Lower = np.array([values[1][0] - thread, values[1][1] - thread, values[1][2] - thread])
        Upper = np.array([values[1][0] + thread, values[1][1]+ thread, values[1][2] + thread])
        #import pdb;pdb.set_trace()    #print(Lower,Upper)
        Binary = np.add(cv2.inRange(image, Lower, Upper),Binary)
        horizontal_indicies, vertical_indicies = np.where(Binary)
        Coords = np.stack([horizontal_indicies, vertical_indicies], axis=-1)
        if horizontal_indicies.shape[0]:
            # 聚类、去除异常值
            db = skc.DBSCAN(eps=8, min_samples=2).fit(Coords)  # DBSCAN聚类方法 还有参数，matric = ""距离计算方法
            labels = db.labels_  # 获取类别标签：-1为异常值。
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            areamax=float('-inf')
            for n in range(n_clusters_):
                x1, x2 = horizontal_indicies[labels == n].min(), horizontal_indicies[labels == n].max()
                y1, y2 = vertical_indicies[labels == n].min(), vertical_indicies[labels == n].max()
                x2 += 1
                y2 += 1
                wheather_append=True
                if x2 - x1 < 10 or y2 - y1 < 10:
                    wheather_append=False
                    continue
                boxes_origin[key].append([y1, x1, y2, x2]) #原始产生的所有框
                dx,dy=y2-y1,x2-x1
                if dx*dy>areamax:
                    areamax=dx*dy
                    xmin,ymin,xmax,ymax= y1, x1, y2, x2  
                    dx_,dy_=xmax-xmin,ymax-ymin
                    x_,y_=(xmax+xmin)/2,(ymax+ymin)/2
                    x_min_y_num = np.count_nonzero(np.isin(horizontal_indicies,[x1,x1+1]))
                    x_max_y_num =  np.count_nonzero(np.isin(horizontal_indicies,[x2,x2-1]))
                    y_dis = abs(x_min_y_num - x_max_y_num)
                    y_min_x_num = np.count_nonzero(np.isin(vertical_indicies,[y1,y1+1]))
                    y_max_x_num =  np.count_nonzero(np.isin(vertical_indicies,[y2,y2-1]))
                    x_dis = abs(y_min_x_num - y_max_x_num)
                    if y_dis > x_dis:
                        if x_min_y_num > x_max_y_num:
                            direction = 0#[1,0]
                        else:
                            direction = 180#[-1,0]
                    else:
                        if y_min_x_num > y_max_x_num :
                            direction = 270
                        else:
                            direction = 90
                    for i in range(data.size[0]):
                        for j in range(data.size[1]):
                            pixdata = (data.getpixel((i,j)))#
                            if ((Lower[0]<=pixdata[0]<=Upper[0] and Lower[1]<=pixdata[1]<=Upper[1] and Lower[2]<=pixdata[2]<=Upper[2])):
                                data.putpixel((i, j),tuple(values[0]))
#                     continue
                    directions[key].append([y1, x1, y2, x2,direction*math.pi/180])
                    #directions[key].append([y1, x1, y2, x2,direction])
                    
                    #print(directions.keys(),directions.values())
            last_image=np.array(data.convert('RGB'))
            if  wheather_append:   
                boxes[key].append([xmin,ymin,xmax,ymax])  #现在产生的最大框
                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), values[0], -1)
                p[key].append([x_,y_ ])
        for key,values in boxes_origin.items():
            for value in values:
                if value not in boxes[key]:
                    cv2.rectangle(last_image,  (value[0], value[1]), (value[2], value[3]),(255,255,255), -1)

    for key,data1 in directions.items():
        
        for value in data1: 
            x,y = value[0],value[1]
            x1,y1 = value[2],value[3]
            dx = x1 -x
            dy= y1 - y
            x_,y_ = (x1+x)/2,(y1+y)/2
            rot = int(value[-1]/90)
            print(rot)
           
            if rot == 0:
                points = [[x_ - 0.5 * dx, y_ -0.5 * dy], [x_, y_ + 0.5 * dy], [x_ + 0.5 * dx, y_ - 0.5 * dy]]
            elif rot == 1:
                points = [[x_ + 0.5 * dx, y_ - 0.5 * dy], [x_ - 0.5 * dx, y_], [x_ + 0.5 * dx, y_ + 0.5 * dy]]
            elif rot == 2:
                
                points = [[x_ - 0.5 * dx, y_ + 0.5 * dy], [x_, y_ - 0.5 * dy], [x_ + 0.5 * dx, y_ + 0.5 * dy]]
            elif rot == 3:
                points = [[x_ - 0.5 * dx, y_ - 0.5 * dy], [x_ + 0.5 * dx, y_], [x_ - 0.5 * dx, y_ + 0.5 * dy]]
            points = np.array(points,dtype=np.int)
            #print(key,points)
               
            cv2.fillConvexPoly(last_image,points, (255, 255,0))
            #points = np.array(points,dtype=np.int)
           
            #print(boxes,boxes_origin)
            
    for key, data in boxes.items():
        for value in data:
            cv2.rectangle(last_image, (value[0], value[1]), (value[2], value[3]), (0, 0, 0), 2)
    path='/ai/zhuhong/the_last_result/{}'.format('1.jpg')
    #path='/ai/zhuhong/the_last_result/{}'.format(file.split('/')[-1])
    os.makedirs('/ai/zhuhong/the_last_result',exist_ok=True)
#     path='/ai/zhuhong/pre_ratate_bbox_result3/{}'.format(file.split('/')[-1])
#     os.makedirs('/ai/zhuhong/pre_ratate_bbox_result3',exist_ok=True)
    fig=show_image(last_image,p,True,path)
    return directions




# data = Image.open(test_pngs[1])
# image = np.array(data)
# boxes = get_image_bbox(image,{"48":[[ 253, 200, 127],[128,168,93]],"51":[[125,236,254],[0,206,220]],"38":[[121,254,159],[0,223,135]]})
# print(boxes)









