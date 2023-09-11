# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:01:43 2023
@author: thean
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad

# Get the current user's home directory
user_home = os.path.expanduser("~")

# Specify the rest of the file path but without needing to rewrite username
ef10file = "EF10Side-06152023124023-0000_106kPa.mp4" #Only one with white text on top
ef10 = os.path.join(user_home, "Downloads", ef10file)
ef20file= "EF20Side-06152023130307-0000_112kPa.mp4"
ef20=os.path.join(user_home, "Downloads", ef20file)
ef30file="EF30Side-06152023132504-0000_180kPa.mp4"
ef30=os.path.join(user_home, "Downloads", ef30file)
ef50file="EF50Side-06152023134549-0000_240kPa.mp4"
ef50=os.path.join(user_home, "Downloads", ef50file)
efgelfile="EFGelSide-06152023121714-0000_37kPa.mp4"
efgel=os.path.join(user_home, "Downloads", efgelfile)
ms30file="MS30Side-06152023140706-0000_663kPa.mp4"
ms30=os.path.join(user_home, "Downloads", ms30file)
substrates = [ef30] #Enter one by one so its easier to analyze
averages=[]
areas = []
frames = []
for s in substrates:
    cap = cv2.VideoCapture(s)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            xvals = []
            yvals = []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            kernel = 5 #higher kernel means more blurring
            blur_gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)

            low = 50
            high = 150

            edges = cv2.Canny(gray, low, high)

            rho = 1 #distance resolution in pixels of the Hough grid
            theta = np.pi/700 #angular resolution in radians of the Hough grid
            threshold = 50 #minimum number of votes (intersections in Hough grid cell)
            min_line_length = 20 #minimum number of pixels making up a line
            max_line_gap =20 #maximum gap in pixels between connectable line segments
            line_image = np.copy(frame) * 0
            lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
         
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        #Create  rotated rectangles. We take the detected line, calculate its length and angle, and given a specified height
                        #, we map out the 4 vertices (Two of them are default endpoints of the line, the other two are mapped above)
                        length= np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)#default length of line
                        theta = np.arctan2(y2 - y1, x2 - x1) #can also do arccosi, just need hypotenuse and one of x/y lenghts
                        height=25 #Adjustable
                        deltax=height*np.cos(1.57-theta)
                        deltay=height*np.sin(1.57-theta)
                        rectx1=x1
                        recty1=y1
                        rectx2=int(x1+deltax)
                        recty2=int(y1-deltay)
                        rectx3=int(x2+deltax)
                        recty3=int(y2-deltay)
                        rectx4=x2
                        recty4=y2
                        vertices=np.array([(rectx1,recty1),(rectx2,recty2),(rectx3,recty3),(rectx4,recty4)])
                        #cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Uncomment to see lines detected
                        roi = frame[min(recty1, recty2, recty3, recty4):max(recty1, recty2, recty3, recty4), min(rectx1, rectx2, rectx3, rectx4):max(rectx1, rectx2, rectx3, rectx4)]
                        average_intensity = np.mean(roi)
                        if average_intensity<100 and abs(x1-x2)>50 and y1-y2>0:
                            xvals.append(x1)
                            xvals.append(x2)
                            yvals.append(y1)
                            yvals.append(y2)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 2) # lines detected after filter
                #cv2.polylines(line_image, [vertices], isClosed=True, color= (0,0,255), thickness=2) #Uncomment to see rectangles 
                try: #Try is recommended in case of error of polynomial mapping (common in beginning of vid)
                    coeffs=np.polyfit(xvals,yvals,2) #Polynomial of 2nd degree (quadratric)
                    curve_x = np.linspace(min(xvals), max(xvals), num=200)
                    curve_y = np.polyval(coeffs, curve_x)
                    curve_coords = np.column_stack((curve_x, curve_y)).astype(np.int32)
                    cv2.polylines(line_image, [curve_coords], isClosed=False, color=(0, 0, 255), thickness=2) #Map out polynomial
                    leftbound=min(xvals)
                    rightbound=max(xvals)
                    a=coeffs[0] 
                    b=coeffs[1]
                    c=coeffs[2]
                    def quadratic_function(x):
                        return a * x**2 + b * x + c
                    area, _ = quad(quadratic_function, leftbound, rightbound) #Integral inside of polynomial, accounts for everything below
                    areas.append(area)
                except:
                    pass
                #Filter out areas by standard deviation
                mean_area = np.mean(areas[int((4/6)*len(areas)):int((5/6)*len(areas))]) #Changed to fraction of list instead of flat number
                std_deviation = np.std(areas[int((4/6)*len(areas)):int((5/6)*len(areas))])
                std_threshold = 0.5 #Adjustable
                range_threshold=std_deviation*std_threshold
                filtered_areas = [x for x in areas if abs(x - mean_area) <= range_threshold]
                frames = list(range(1, len(filtered_areas)+1))
                line_image = cv2.addWeighted(line_image, 1, frame, 1, 0)
            else:
                break
            avrgfiltered=np.nanmean(filtered_areas)
            averages.append(avrgfiltered)
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 600, 600)
            cv2.imshow('frame', line_image)
            cv2.waitKey(1)
    cv2.destroyAllWindows()  
    #Collected areas of following substrates
    #ms30=229412 
    #ef20=735989 lower moduli had the higher side view 
    #ef30=562398 (reasonable)
    #ef50=168781 (reasonable)
    '''
    xvals=[list of moduli]
    plt.plot(xvals, averages, "-b"
    plt.xlabel('Moduli (s)')
    plt.ylabel('Areas (mm)')
    plt.title('Distance vs Time (ef10)')
    
    '''
    #Note: User can manually plot after desired variables are collected. Above is an example.
   
        
        







