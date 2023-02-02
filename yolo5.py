import torch
import cv2
import numpy as np
from tracker import *



def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)



model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
count=0
cap=cv2.VideoCapture('tvid.mp4')

tracker=Tracker()

area1=[(424,257),(410,276),(779,293),(770,271)]
area1_c=set()
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,600))
    results = model(frame)
    list=[]
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
#        print(d)
     
        if 'motorcycle' in d:

            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x,y,w,h,id=bbox
        cx=int(x+w)//2
        cy=int(y+h)//2
        cv2.rectangle(frame,(x,y),(w,h),(255,255,0),2)
        results=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if results>=0:
           cv2.rectangle(frame,(x,y),(w,h),(255,0,0),2)
           cv2.putText(frame,str(id),(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
           cv2.circle(frame,(cx,cy),3,(0,0,255),-1) 
           area1_c.add(id) 
               
                   

    b=len(area1_c)
    cv2.putText(frame,str(b),(50,60),cv2.FONT_HERSHEY_PLAIN,5,(0,0,255),3)

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
    cv2.imshow("ROI",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
