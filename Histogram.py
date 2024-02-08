import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img=cv.imread('cats.jpg')
cv.imshow('cats',img)

blank= np.zeros(img.shape[:2],dtype='uint8')
b,g,r= cv.split(img)

blue= cv.merge([b, blank, blank])
green= cv.merge([b, g, blank])
red= cv.merge([blank, blank, r])

cv.imshow("B",blue)
plt.figure()
plt.title('color histogram')
plt.xlabel('pixel intensity')
plt.ylabel('number of pixel')
blue_hist = cv.calcHist([blue],[0], None, [256], [0,256])
plt.plot(blue_hist,color='blue')

red_hist = cv.calcHist([red],[2], None, [256], [0,256])
plt.plot(red_hist,color='red')

green_hist = cv.calcHist([green],[1], None, [256], [0,256])
plt.plot(green_hist,color='green')

cv.imshow("G",green)
cv.imshow("R",red)
plt.xlim([0,256])
plt.show()

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)
cv.waitKey(0)
