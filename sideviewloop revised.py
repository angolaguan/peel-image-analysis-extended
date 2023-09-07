# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:01:43 2023

@author: 3mai1
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the current user's home directory
user_home = os.path.expanduser("~")

# Specify the rest of the file path but without needing to rewrite username
# Use must ensure that these file names are the same as on file explorer
ef10file = "EF10Side-06152023124023-0000_106kPa.mp4"
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

xvalssss = [0, 1, 2, 3, 4, 5]
substrates = [ms30, ef50, ef30, ef20, ef10, efgel]
averages = []
#SOLELY FOR TESTING


for s in substrates:
    
    areas = []
    frames = []
    
    f = 0
#allows video file to be open
    cap = cv2.VideoCapture(s)
    #initializes frame reading
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            xvals = []
            yvals = []
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # kernel = 5
            # blur_gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
            
            low = 40
            high = 120
            
            edges = cv2.Canny(gray, low, high)
            
            rho = 1 #distance resolution in pixels of the Hough grid
            theta = np.pi/700 #angular resolution in radians of the Hough grid
            threshold = 50 #minimum number of votes (intersections in Hough grid cell)
            min_line_length = 20 #minimum number of pixels making up a line
            max_line_gap =20 #maximum gap in pixels between connectable line segments
            line_image = np.copy(frame) * 0
            
            lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
            
            upperboundx=0
            upperboundy=0
            lowerboundx=100000
            lowerboundy=100000
            
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        if abs(x1-x2)>50 and y1-y2>0:
                            #checks for a criteria of length and orientation, and adds to list of values
                            xvals.extend([x1, x2])
                            yvals.extend([y1, y2])
                            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                            if y1>upperboundy:
                                upperboundx = x1
                                upperboundy = y2
                            if y1<lowerboundy:
                                lowerboundx = x2
                                lowerboundy = y1
                      
                       
                #overlays on original fram
                line_image = cv2.addWeighted(frame, 1, line_image, 1, 0)
             
                try:
                    #diagonal line
                    cv2.line(line_image, (lowerboundx, lowerboundy), (upperboundx, upperboundy), (255, 255, 255), 2)
                    
                    lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)
                    x1, y1 = lowerboundx, lowerboundy
    
                    x2, y2 = upperboundx, upperboundy
                    
                    A = (y2-y1)*-1
                    B = x1-x2
                    C = x2*y1-x1*y2
                    
                    distances = np.abs(A*xvals+B*yvals+C)/np.sqrt(A**2+B**2)
                    mean_distance = np.mean(distances)
                    std_deviation = np.std(distances)
                    
                    std_threshold = 1.0
                    filtered_xvals = []
                    filtered_yvals = []
                    
                    for x, y, distance in zip(xvals, yvals, distances):
                        if np.abs(distance - mean_distance) <= std_threshold * std_deviation:
                            filtered_xvals.append(x)
                            filtered_yvals.append(y)
                            x = int(x)
                            y = int(y)
                            cv2.circle(line_image, (x, y), 2, (0, 255, 0), 2)
                            

                    coeffs = np.polyfit(filtered_xvals, filtered_yvals, 2)
                    #numpy approach
                    #coeffs = np.polyfit(xvals, yvals, 2)
                    
                    curve_x = np.linspace(min(xvals), max(xvals), num = line_image.shape[1])
                    curve_y = np.polyval(coeffs, curve_x)
                    
                    mask = np.zeros_like(line_image, dtype = np.uint8)
                    points = np.column_stack((curve_x, curve_y)).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(mask, [points], isClosed = False, color=(0, 0, 255), thickness = 2)
                    
                    line_image = cv2.addWeighted(line_image, 1, mask, 1, 0)
                    
                    f+=1
                    frames.append(f)
                    coeffs = coeffs/100
                    upperboundx = upperboundx/100
                    lowerboundx = lowerboundx/100
                    
                    area_upperbound = coeffs[0]*upperboundx + coeffs[1]*upperboundx*upperboundx/2 + coeffs[2]*upperboundx*upperboundx*upperboundx
                    area_lowerbound = coeffs[0]*lowerboundx + coeffs[1]*upperboundx*upperboundx/2 + coeffs[2]*lowerboundx*lowerboundx*lowerboundx
                    area = area_lowerbound-area_upperbound
                    areas.append(area)
                except:
                    pass
            
            '''
            #diagonal line
            cv2.line(line_image, (lowerboundx, lowerboundy), (upperboundx, upperboundy), (255, 255, 255), 2)
            lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)
            '''
            '''
            #vertical line
            cv2.line(line_image, (lowerboundx, lowerboundy), (lowerboundx, upperboundy), (0, 0, 255), 2)
            image = cv2.addWeighted(image, 1, line_image, 1, 0)
            #horizontal line
            cv2.line(line_image, (lowerboundx, upperboundy), (upperboundx, upperboundy), (255, 0, 255), 2)
            image = cv2.addWeighted(image, 1, line_image, 1, 0)
            '''
            '''
            height = abs(upperboundy-lowerboundy)/100
            halfbase = abs(upperboundx-lowerboundx)/100
            totalarea = height*halfbase
            print("height = " + str(height) + "mm")
            print("halfbase = "  + str(halfbase) + "mm")
            print("area (assuming symmetric) = " +str(totalarea) + " sq mm")
            areas.append(totalarea)
            '''
        else:
            break
            
        # triangle= findtriangle(edges)
        # if triangle is not None:
        #     cv2.drawContours(edges, [triangle], 0, (255, 0, 0), 5)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 600, 600)
        cv2.imshow('frame', line_image)
        cv2.waitKey(1)


    cv2.destroyAllWindows()
    if s == ef50:
        average = np.nanmean(areas[400:500])
    else:
        average = np.nanmean(areas[700:800])
    averages.append(average)
print(xvalssss)    
print(averages)
plt.bar(xvalssss, averages, color="green")
plt.xlabel('Moduli')
plt.ylabel('Average distance (mm)')
plt.title('Avg distance vs ???')
