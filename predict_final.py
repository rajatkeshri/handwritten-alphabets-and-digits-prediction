import numpy as np
import cv2
import os
from keras.models import load_model
import numpy as np
import PIL
from PIL import ImageOps
import cv2
import numpy as np
import win32com.client as wincl

import glob


def predict(counter,image):
    #--------------------SEPERATE EACH LETTER--------------------------
    
    im = cv2.imread(image)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(imgray,127,255,0)

    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    '''image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    '''

    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    foldername = "CustomInput{}".format(counter)
    os.mkdir(foldername)

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
    

        if w > 100 and h > 100:
            path = 'C:/Anaconda codes/Handwritting/CustomInput{}'.format(counter)
        
            roi=cv2.resize(roi,(28,28))
         
            cv2.imwrite(os.path.join(path , str(idx) + '.jpg'), roi)
            #cv2.imwrite(str(idx) + '.jpg', roi)
            #cv2.imwrite('roi.png'.format(i), roi)


    
    cv2.imshow("q",im)
    cv2.waitKey(0)

    #---------------------PREDICT--------------------------------------

    model = load_model('weights.model')
    test_image = []
    name=foldername
    for img in glob.glob(foldername+ "/*.jpg"):
        n= cv2.imread(img)
        test_image.append(n)

    finalstring=""


    alphabet=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


    for i in range(0,len(test_image))   :
        #test_image[i] = cv2.imread('5.jpg')
        test_image[i] = cv2.cvtColor(test_image[i], cv2.COLOR_BGR2GRAY)
        #test_image=cv2.resize(test_image,(28,28))
        test_image[i] = test_image[i].reshape(1, 28, 28, 1)
        test_image[i] = np.array(test_image[i])
        test_image[i] = test_image[i].astype('float32')
        test_image[i] /= 255
        #print (test_image[i].shape)

        probablity=(model.predict(test_image[i]))
        #print(probablity)

        xyz=model.predict_classes(test_image[i])
        #print(xyz)

        alphaeach=alphabet[int(xyz)]
        finalstring=finalstring+alphaeach
        #print(alphaeach)
        
    print("FINAL OUTPUT= "+finalstring)
    speak = wincl.Dispatch("SAPI.SpVoice")
    #speak.Speak(finalstring)
    return
#--------------------------------------------------------------------------

image='ABC.jpg'

#image = PIL.ImageOps.invert(image)
counter=1
predict(counter=counter,image=image)











    
