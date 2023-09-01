

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:49:47 2023
@author: 3mai1
"""
import time
start_time=time.time()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the current user's home directory
user_home = os.path.expanduser("~")

# Specify the rest of the file path but without needing to rewrite username
# Use must ensure that these file names are the same as on file explorer

ef10file = "EF10Front-06152023124024-0000_106kPa.mp4"
ef10 = os.path.join(user_home, "Downloads", ef10file)
ef20file= "EF20Front-06152023130308-0000_112kPa.mp4"
ef20=os.path.join(user_home, "Downloads", ef20file)
ef30file="EF30Front-06152023132505-0000_180kPa.mp4"
ef30=os.path.join(user_home, "Downloads", ef30file)
ef50file="EF50Front-06152023134551-0000_240kPa.mp4"
ef50=os.path.join(user_home, "Downloads", ef50file)
efgelfile="EFGelFront-06152023121715-0000_37kPa.mp4"
efgel=os.path.join(user_home, "Downloads", efgelfile)
ms30file="MS30Front-06152023140708-0000_663kPa.mp4"
ms30=os.path.join(user_home, "Downloads", ms30file)

xvals = [6.63, 2.40, 1.80, 1.12, 1.06, .37]
#xvals=[0]
averages = []
#substrates=[ef20]
substrates=[ms30, ef50, ef30, ef20, ef10, efgel]

for s in substrates:
    distances = []
    frames = []
    f=0
    #Can adjust this to 300 to analyze stable frames
    cap = cv2.VideoCapture(s)
    #Adjustment of f is the frame range desired
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel = 5
            blur_gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)

            low = 50
            high = 100

            edges = cv2.Canny(blur_gray, low, high)

            rho = 1 #distance resolution in pixels of the Hough grid
            theta = np.pi/1000 #angular resolution in radians of the Hough grid
            threshold = 30 #minimum number of votes (intersections in Hough grid cell)
            min_line_length = 50 #minimum number of pixels making up a line
            max_line_gap = 50 #maximum gap in pixels between connectable line segments
            line_image = np.copy(frame) * 0 # creates a copy of black pixels

            #run Hough on edge detected image 
            lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

            if lines is not None:
                listforlimits=[]
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        if abs(y1-y2) <3:
                            #draws green lines over image
                            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            listforlimits.append(y1)
                            listforlimits.append(y2)
                            upperbound=min(listforlimits)
                            lowerbound=max(listforlimits)

            #draws a red vertical line in the center
      
                #Uppperbound, refers to the higher number
            
                cv2.line(line_image, (int(frame.shape[0]/2), upperbound), (int(frame.shape[0]/2), lowerbound), (0, 0, 255), 2)
                lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)

            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 600, 600)
            cv2.imshow('frame', lines_edges)
            cv2.waitKey(1)
            
            #if lowerbound-upperbound<0:
                #print(f)
                #print("WARNING: Distance below 0 detected")

            frames.append(f)
            f+=1
            #Merged /26 to obtain single line 
            distance_mm = abs(upperbound-lowerbound)/26
            #if distance_mm<0:
                #distances.append(np.nan)
        
            distances.append(distance_mm)
        else:
            break
    cv2.destroyAllWindows()
    '''
    frames_adjusted = frames[300:1060]
    distances_adjusted = distances[300:1060]
    plt.plot(frames_adjusted, distances_adjusted, 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    plt.title('Distance vs Time (ef10)')
    
    '''
    print(s)
    distanceadj=distances[600:1000]

    average = np.nanmean(distanceadj)
    averages.append(average)
print(averages)
plt.bar(xvals, averages, color="green")
plt.xlabel('112 kPa')
plt.ylabel('Average distance (mm)')
plt.title('Avg distance at 112 kPa')
end_time=time.time()
print(end_time-start_time)