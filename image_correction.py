import common.detectorClient as detectorClient      
import numpy as np    
import cv2 

'''This function takes in image
rotates in clockwise and anticlockwise direction,
does Object detection inference and returns the image
with maximum object detections which will be in most cases
the straight image
'''

class image_rectifier():
    def __init__(self,config={}):
        config={"endpoint":"http://localhost","port":9000,"route":'detect'}
        self.detector=detectorClient(config)
        
    def run(self,path):
        
        img=cv2.imread(path)
        max=-1
        maxImage=None
        
        # Inference on clockwise rotated image
        dets= self.detector.detect(img,threshold=0.3)
        print("dets",len(dets))
        if len(dets)>max:
            maxImage=img
            max=len(dets)
        
        # Inference on clockwise rotated image
        img1=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        dets1= self.detector.detect(img1,threshold=0.3)
        print("dets1",len(dets1))
        if len(dets1)>max:
            maxImage=img1
            max=len(dets1)
    
        # Inference on counter clockwise rotated image
        img2=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        dets2=self.detector.detect(img2,threshold=0.3)
        print("dets2",len(dets2))
        if len(dets2)>max:
            maxImage=img2
            max=len(dets2)
            
        return maxImage 
    
    
if __name__ == '__main__':
    config={"endpoint":"http://localhost","port":9000,"route":'detect'}
    detector=detectorClient(config)
    i=5
    while i<7:
        i+=1
        
        max=-1
        maxImage=None
        img=cv2.imread('data/t{}.jpeg'.format(str(i)))
        dets= detector.detect(img,threshold=0.3)
        print("dets",len(dets))
        if len(dets)>max:
            maxImage=img
            max=len(dets)
        
        
        img1=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        dets1= detector.detect(img1,threshold=0.3)
        print("dets1",len(dets1))
        if len(dets1)>max:
            maxImage=img1
            max=len(dets1)
        
        img2=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        dets2=detector.detect(img2,threshold=0.3)
        print("dets2",len(dets2))
        if len(dets2)>max:
            maxImage=img2
            max=len(dets2)        
        
        cv2.imwrite('data/out{}.jpeg'.format(str(i)),maxImage)
        
        
        
        