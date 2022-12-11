#!/usr/bin/env python
# coding: utf-8

# Morphological Transformation

# In[13]:


import cv2 as cv

img=cv.imread("8.png")
cv.imshow("image",img)
cv.waitKey(5000)
cv.destroyAllWindows


# In[16]:


#erode

import numpy as np
img=cv.imread("8.png")
kernel=np.ones((7,7),np.uint8)
erosion=cv.erode(img,kernel,iterations=1)
cv.imshow("image",erosion)
cv.waitKey(5000)
cv.destroyAllWindows


# In[20]:


#dilate

import numpy as np
kernel=np.ones((5,5),np.uint8)
dilate=cv.dilate(img,kernel,iterations=1)
cv.imshow("image",dilate)
cv.waitKey(5000)
cv.destroyAllWindows


# In[21]:


#open

import numpy as np
img=cv.imread("8n.png")
kernel=np.ones((1,1),np.uint8)
opening=cv.morphologyEx(img,cv.MORPH_OPEN,kernel)
cv.imshow("image",opening)
cv.waitKey(5000)
cv.destroyAllWindows
#erosion +dilation


# In[22]:


#close
img=cv.imread("8n.png")
import numpy as np
kernel=np.ones((11,11),np.uint8)
closing=cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)
cv.imshow("image",closing)
cv.waitKey(5000)
cv.destroyAllWindows


# In[25]:


#gradient

import numpy as np
kernel=np.ones((3,3),np.uint8)
gradient=cv.morphologyEx(img,cv.MORPH_GRADIENT,kernel)
cv.imshow("image",gradient)
cv.waitKey(5000)
cv.destroyAllWindows


# In[35]:


import numpy as np
blank=np.zeros(img.shape[:2],dtype='uint8')
cv.imshow("balnk",blank)
contours,h=cv.findContours(blank)
cv.drawContours(blank,contours,-1,(0,0,255),2)
cv.imshow("blank",blank)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:


import cv2 as cv
img = cv.imread('nw.jpg')
width = int(img.shape[1] * 40 / 100)
height = int(img.shape[0] * 40 / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
cv.imshow("norma",img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
cv.imshow("gray",gray)
ret,thresh=cv.threshold(gray,100,255,cv.THRESH_BINARY)
cv.imshow("thresh",thresh)
cv.waitKey(5000)
cv.destroyAllWindows


# Canny edge

# In[33]:


import cv2 as cv 
import numpy as np
img = cv.imread('nw.jpg')
width = int(img.shape[1] * 40 / 100)
height = int(img.shape[0] * 40 / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_blur=cv.blur(img_gray,(3  ,3  ))
canny=cv.Canny(img,60,255)
cv.imshow("canny",canny)
cv.imshow("ORIGINAL",img)
cv.waitKey(5000)
cv.destroyAllWindows()


# In[2]:


import cv2 as cv
video1=cv.VideoCapture(0)
while True:
    isTrue,frame =video1.read()
    img_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    img_blur=cv.blur(img_gray,(3  ,3  ))
    img_canny=cv.Canny(img_blur, 0,255)
    cv.imshow("result2",img_canny)
    print(isTrue)
    if cv.waitKey(1) & 0xFF ==ord('d'):
        break
video1.release()
cv.destroyAllWindows()


# sobel

# In[40]:


import cv2

img = cv2.imread('nw.jpg')
width = int(img.shape[1] * 40 / 100)
height = int(img.shape[0] * 40 / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
 
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(8000)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(8000)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(8000)
cv2.destroyAllWindows()


# Figures

# In[50]:


test1=np.zeros((500,500,3),dtype='uint8')
#cv.rectangle(test1,(0,0),(500,250),(255,0,0),thickness =cv.FILLED)
cv.rectangle(test1,(0,0),(test1.shape[0]//2,test1.shape[1]//2),(0,0,255),thickness=-1)
cv.imshow('rectangle',test1)
cv.waitKey(5000)
cv.destroyAllWindows()


 



# In[48]:


test2=np.zeros((500,500,3),dtype='uint8')
cv.circle(test2,(400,250),10,thickness=-1,color=(0,0,255))
cv.imshow("circle",test2)
cv.waitKey(5000)
cv.destroyAllWindows()




# In[53]:


test3=np.zeros((500,500,3),dtype='uint8')

cv.line(test3,(0,0),(250,250),(0,255,0),thickness=4)
cv.line(test3,(400,400),(300,300),(100,28,90),thickness=4) #2nd line
cv.line(test3,(100,400),(200,300),(100,28,90),thickness=4) 


cv.imshow("line1",test3)

cv.waitKey(5000)
cv.destroyAllWindows()




# In[55]:


test4=np.zeros((500,500,3),dtype='uint8')
cv.putText(test4,"hello World",(225,225),cv.FONT_HERSHEY_TRIPLEX,6.0,(0,255,0),thickness=3)
cv.imshow("text",test4)
cv.waitKey(5000)
cv.destroyAllWindows()


# Matpkotlib operations

# In[57]:


import matplotlib.pyplot as plt
  
# initializing the data
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
  
# plotting the data
plt.plot(x, y)
  
# Adding the title
plt.title("Simple Plot")
  
# Adding the labels
plt.ylabel("y-axis")
plt.xlabel("x-axis")
plt.show()


# In[58]:


import matplotlib.pyplot as plt 
# data to display on plots 
x = [3, 1, 3] 
y = [3, 2, 1] 
z = [1, 3, 1] 
  
# Creating figure object 
plt.figure() 
  
# addind first subplot 
plt.subplot(121) 
plt.plot(x, y) 
  
# addding second subplot 
plt.subplot(122) 
plt.plot(z, y)


# In[ ]:


import matplotlib.pyplot as plt

# Creating the figure and subplots
# according the argument passed
fig, axes = plt.subplots(1, 2)

# plotting the data in the 1st subplot
axes[0].plot([1, 2, 3, 4], [1, 2, 3, 4])

# plotting the data in the 1st subplot only
axes[0].plot([1, 2, 3, 4], [4, 3, 2, 1])

# plotting the data in the 2nd subplot only
axes[1].plot([1, 2, 3, 4], [1, 1, 1, 1])


# In[59]:


import matplotlib.pyplot as plt

# data to display on plots
x = [3, 1, 3]
y = [3, 2, 1]
plt.plot(x, y)
plt.plot(y, x)

# Adding the legends
plt.legend(["blue", "orange"])
plt.show()


# In[61]:


import matplotlib.pyplot as plt
# data to display on plots
x = [0,255]
y = [100,20]

# This will plot a simple bar chart
plt.bar(x, y)

# Title to the plot
plt.title("Bar Chart")

# Adding the legends
plt.legend(["bar"])
plt.show()


# In[66]:


import matplotlib.pyplot as plt 
import matplotlib.image as img 
# reading the image 
testImage = img.imread('8n.png') 
# displaying the image 
plt.imshow(testImage) 


# In[64]:


import matplotlib.pyplot as plt
# data to display on plots
x = [3, 1, 3, 12, 2, 4, 4]
y = [3, 2, 1, 4, 5, 6, 7]

# This will plot a simple scatter chart
plt.scatter(x, y)

# Adding legend to the plot
plt.legend("A")

# Title to the plot
plt.title("Scatter chart")
plt.show()


# In[67]:


import cv2 
import numpy as np
def nothing(x):
    pass


cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 180, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
cap=cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)
    key =cv2.waitKey(1)
    if key ==27:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
  
img = cv2.imread('red.jpg')
width = int(img.shape[1] * 40 / 100)
height = int(img.shape[0] * 40 / 100)
dim = (width, height)
imagergb = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
imagehsv=cv2.cvtColor(imagergb, cv2.COLOR_BGR2HSV)
cv2.imshow("norma",imagehsv)
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([357 ,97,100])
#masking the image using inRange() function
imagemask = cv2.inRange(imagehsv, lower_bound, upper_bound)
cv2.imshow("circle",imagemask)
cv2.waitKey(5000)
cv2.destroyAllWindows()
 
 


# In[63]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('j.png',0)
width = int(img.shape[1] * 40 / 100)
height = int(img.shape[0] * 40 / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
plt.hist(img.ravel(),256)
plt.show()


# In[5]:


import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('nw.jpg')
width = int(img.shape[1] * 40 / 100)
height = int(img.shape[0] * 40 / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
hsv = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
cv.imshow("circle",hsv)
cv.waitKey(5000)
cv.destroyAllWindows()


# In[ ]:




