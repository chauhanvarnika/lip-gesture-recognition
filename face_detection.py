import cv2;
import numpy as np;
import dlib;
import math;

face_detector = dlib.get_frontal_face_detector();
landmark_pred = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat');

cv2.namedWindow("Landmark Detection")
cap = cv2.VideoCapture(0);


def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=True):
#   points = []
  for i in range(startpoint, endpoint+1):
    # point = [faceLandmarks[i][0], faceLandmarks[i][1]];
    # points.append(point);

    #For mouth open and close gesture
    lip_left_coord = (int (faceLandmarks[60][0]), int(faceLandmarks[60][1]))
    lip_right_coord = (int (faceLandmarks[64][0]), int(faceLandmarks[64][1]))
    lip_wd = (lip_left_coord[0] - lip_right_coord[0]) ** 2 +( lip_left_coord[1] - lip_right_coord[1]) ** 2 

    lip_top_coord = (int (faceLandmarks[62][0]), int(faceLandmarks[62][1]))
    lip_btm_coord = (int (faceLandmarks[66][0]), int(faceLandmarks[66][1]))
    lip_ht = (lip_top_coord[0] - lip_btm_coord[0]) ** 2 +( lip_top_coord[1] - lip_btm_coord[1]) ** 2

    # line_from_left = (int(faceLandmarks[54][0]) - int(faceLandmarks[3][0])) ** 2 + (int(faceLandmarks[54][1]) - int(faceLandmarks[3][1])) ** 2
    # line_from_right = (int(faceLandmarks[48][0]) - int(faceLandmarks[13][0])) ** 2 + (int(faceLandmarks[48][1]) - int(faceLandmarks[13][1])) ** 2
    # smile_ratio = round((line_from_left/line_from_right), 2)
    # print(line_from_left, line_from_right, )
    # cv2.line(image, (int(faceLandmarks[54][0]),int(faceLandmarks[54][1])), (int(faceLandmarks[3][0]),int(faceLandmarks[3][1])), (255,150,0))
    # cv2.line(image, (int(faceLandmarks[48][0]),int(faceLandmarks[48][1])), (int(faceLandmarks[13][0]),int(faceLandmarks[13][1])), (255,150,0))

    lip_left_corner = (int (faceLandmarks[48][0]), int(faceLandmarks[48][1]))
    lower_right_corner = (int (faceLandmarks[55][0]), int(faceLandmarks[55][1]))
    lip_right_corner = (int (faceLandmarks[54][0]), int(faceLandmarks[54][1]))
    lower_left_corner = (int (faceLandmarks[59][0]), int(faceLandmarks[59][1]))
    left_line = (lip_left_corner[0] -lower_right_corner[0]) ** 2 +( lip_left_corner[1] -lower_right_corner[1]) ** 2
    right_line = (lower_left_corner[0] -lip_right_corner[0]) ** 2 +( lower_left_corner[1] -lip_right_corner[1]) ** 2

    # print(left_line, right_line, round((left_line/right_line),2))
    # cv2.line(image, lip_left_corner, lower_right_corner, (255,150,0))
    # cv2.line(image, lower_left_corner, lip_right_corner, (255,150,0))

    ulip_top_coord = (int (faceLandmarks[51][0]), int(faceLandmarks[51][1]))
    ulip_btm_coord = (int (faceLandmarks[57][0]), int(faceLandmarks[57][1]))
    ulip_ht = (ulip_top_coord[0] - ulip_btm_coord[0]) ** 2 +( ulip_top_coord[1] - ulip_btm_coord[1]) ** 2

    temp_lip_ht = (lip_top_coord[0] - ulip_btm_coord[0]) ** 2 +( lip_top_coord[1] - lip_btm_coord[1]) ** 2
    temp_ratio = round((temp_lip_ht/lip_wd),2)
    pout_ratio = round((ulip_ht/lip_wd),2)
    
    # cv2.line(image, lip_left_coord, lip_right_coord, (0,0,255))
    # cv2.line(image, lip_top_coord, lip_btm_coord, (0,0,255))
    aspect_ratio = round((lip_ht/lip_wd), 2)
    # print(temp_ratio)
    cv2.putText(image, str(pout_ratio), (50,200), cv2.FONT_ITALIC, 2, (10,10,10))
    if aspect_ratio > 0.08:                                                                             #adding aspect ratio threshold for mouth open and close state
        cv2.putText(image, "open mouth", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (10,0,255))
    elif aspect_ratio == 0.0:
        if pout_ratio > 0.03 and pout_ratio < 0.06:
            cv2.putText(image, "lip folded inwards", (50,50), cv2.FONT_ITALIC, 2, (10,0,255))
        elif pout_ratio > 0.075 and pout_ratio < 0.09:
            cv2.putText(image, "smile", (50,50), cv2.FONT_ITALIC, 2, (10,0,255))
        elif pout_ratio > 0.1 and pout_ratio < 0.25 :
            cv2.putText(image, "close mouth", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (10,0,255))
        elif pout_ratio > 0.25 and pout_ratio < 0.5:
            cv2.putText(image, "pout gesture", (50,50), cv2.FONT_ITALIC, 2, (10,0,255))
         
#   points = np.array(points, dtype=np.int32)                                                           //converting array to a numpy array 
#   cv2.polylines(image, [points], isClosed, (0, 255, 0), thickness=1, lineType=cv2.LINE_8)             // creating lip contour 



while True:
    ret, image = cap.read()
    resized_img = cv2.resize(image, (640, 360))
    gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    face_detected = face_detector(gray_image, 1)
    
    for face in face_detected:
        shape = landmark_pred(gray_image, face)                                                         #Getting landmarks by passing the frame to the landmark detector
        shape_np = np.zeros((68, 2), dtype="int")                                                       #Creating a 68 X 2 numpy array to store (x,y) coordiates of each landmark.
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np
        
        # Display the landmarks
        for i, (x, y) in enumerate(shape):
            if i == 47 :
                drawPoints(resized_img, shape, 48, 59)
            elif i == 59:
                drawPoints(resized_img, shape, 60, 67)

    # Display the image
    cv2.imshow('Landmark Detection', resized_img)

    # escape button to terminate the code
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()