import numpy as np
import cv2

im = cv2.imread('new.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,127,255,0)

kernel = np.ones((10,15), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

#image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#im = cv2.drawContours(im, contours, -1, (0,255,0), 3)


im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

idx=0
for i, ctr in enumerate(sorted_ctrs):
    idx+=1
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
 
    # Getting ROI
    roi = im[y-25:y+h+25, x-25:x+w+25]

    # show ROI
    #cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(im,(x-25,y-25),( x + w+25, y + h+25 ),(0,255,0),2)
    #cv2.imwrite('roi1.png'.format(i), roi)
    

    if w > 15 and h > 15:
        roi=cv2.resize(roi,(28,28))
        cv2.imwrite(str(idx) + '.jpg', roi)
        #cv2.imwrite('roi.png'.format(i), roi)



cv2.imshow("q",im)
cv2.waitKey(0)
