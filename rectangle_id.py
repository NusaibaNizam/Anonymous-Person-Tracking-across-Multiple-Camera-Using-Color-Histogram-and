# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:53:47 2020

@author: HP
"""

import cv2
#def draw_labels(img):
#img=cv2.imread("tanvir.PNG")
#img2=cv2.imread("kobir.PNG")
def put_ID(img,id_number):
    x=0
    y= 0
    #id_number=1
    #color=[255,0,0]

    height,width,color=img.shape
    font = cv2.FONT_HERSHEY_PLAIN
    
    cv2.rectangle(img, (x,y), (x+width, y+height), [255,0,0], 5)
    cv2.putText(img, 'ID:'+str(id_number), (int(width/8),int(height/10)), font, 1, [0,255,0], 1)
    return img
cv2.waitKey(0)