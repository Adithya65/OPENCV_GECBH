#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv


# In[ ]:


img=cv.imread("R.jpg")
cv.imshow("image",img)
cv.waitKey(5000)
cv.destroyAllWindows


# In[ ]:


video1=cv.VideoCapture(0)
while True:
    isTrue,frame =video1.read()
    img_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    img_blur=cv.blur(img_gray,(3  ,3  ))
    img_canny=cv.Canny(img_blur, 130,150)
    cv.imshow("result2",img_canny)
    if cv.waitKey(1) & 0xFF ==ord('d'):
        break
video1.release()
cv.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
 
image = cv2.imread('nw.jpg')
 


kernel1 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
 
identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
cv2.imshow('Identity', identity)
cv2.waitKey(5000)
cv2.destroyAllWindows()
 


# In[ ]:


# Apply blurring kernel
kernel2 = np.ones((5, 5), np.float32) / 25
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
 
 

     
cv2.imshow('Kernel Blur', img)
cv2.waitKey(5000)

cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
 
image = cv2.imread('nw.jpg')
 


kernel1 = np.array([[0.0625, 0.125, 0.0625],
                    [0.125, 0.25, 0.125],
                    [0.0625, 0.125, 0.0625]])
 
identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
cv2.imshow('sobel', identity)
cv2.waitKey(20000)
cv2.destroyAllWindows()


# In[ ]:


# importing libraries
import cv2
import numpy as np

img = cv2.imread('nw.jpg')
width = int(img.shape[1] * 40 / 100)
height = int(img.shape[0] * 40 / 100)
dim = (width, height)

image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


cv2.imshow('Original Image', image)
cv2.waitKey(500)

# Gaussian Blur
Gaussian = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(500)

# Median Blur
median = cv2.medianBlur(image, 5)
cv2.imshow('Median Blurring', median)
cv2.waitKey(500)


# Bilateral Blur
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows(500)


# In[ ]:


import cv2
import matplotlib.pyplot as plt
 
img = cv2.imread('nw.jpg')
width = int(img.shape[1] * 40 / 100)
height = int(img.shape[0] * 40 / 100)
dim = (width, height)

image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
plt.imshow(image,cmap="gray")
plt.show()


# In[ ]:


get_ipython().system('pip install matplotlib')


# In[1]:


x=0
y=0

import cv2 as cv
img=cv.imread("nw.jpg")
gr=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gr.shape


# In[ ]:


for i in range(1000):
    for j in range(1000):
        if (int(gr[i,j])>100):
            gr[i,j]=0
            x=x+1
        else:
            gr[i,j]=255
            y=y+1
cv.imshow("gr",gr)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:


import matplotlib


# In[ ]:




