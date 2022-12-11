#!/usr/bin/env python
# coding: utf-8

# In[ ]:


motion_list=[]
import cv2, time,pandas
static_back = None
video = cv2.VideoCapture(0)
while True:
	check, frame = video.read()
	motion = 0
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	if static_back is None:
		static_back = gray
		continue
	diff_frame = cv2.absdiff(static_back, gray)
	thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
	cnts,_ = cv2.findContours(thresh_frame.copy(),
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in cnts:
		if cv2.contourArea(contour) < 10000:
			continue
		motion = 1
		(x, y, w, h) = cv2.boundingRect(contour)
		
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        
	motion_list.append(motion)
	cv2.imshow("Gray Frame", gray)
	cv2.imshow("Difference Frame", diff_frame)
	cv2.imshow("Threshold Frame", thresh_frame)
	cv2.imshow("Color Frame", frame)
	key = cv2.waitKey(100)
	if key == ord('q'):
		break
video.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np

image1 = cv2.imread('bp1.jpg')

img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
										cv2.THRESH_BINARY, 199, 5)

thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
										cv2.THRESH_BINARY, 199, 5)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
 
cv2.imshow('Adaptive Mean', img)
cv2.imshow('Adaptive Gaussian', thresh2)
cv2.imshow('n', thresh3)

 
if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()


# In[ ]:





# In[ ]:


import cv2
import numpy as np
 
img1 = cv2.imread('1.png') 
img2 = cv2.imread('input2.png')
dim = (500, 500)

dest_and = cv2.bitwise_and(img2, img1, mask = None)

cv2.imshow('Bitwise And', dest_and)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# In[4]:


import cv2
import numpy as np
 
img1 = cv2.imread('11.png') 
img2 = cv2.imread('12.png')
dim = (500, 500)

 
dest_and = cv2.bitwise_or(img2, img1, mask = None)
 
cv2.imshow('Bitwise or', dest_and)
  
 
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# In[3]:


import cv2
import numpy as np

img1 = cv2.imread('11.png') 
img2 = cv2.imread('12.png')
dim = (500, 500)

dest_and = cv2.bitwise_not(img2, mask = None)

cv2.imshow('Bitwise not', dest_and)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# In[5]:


import cv2
import numpy as np

img1 = cv2.imread('11.png')
img2 = cv2.imread('12.png')

dest_xor = cv2.bitwise_xor(img1, img2, mask = None)

cv2.imshow('Bitwise XOR', dest_xor)

if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()


# In[ ]:




