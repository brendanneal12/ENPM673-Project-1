## Brendan Neal
## ENPM673 Spring 2023
## Project 1 Code

##Note, this script is only for question 1. There is a differnt script for question 2.

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

##---------------------------------------------------------------##
##------------------Defining Least Squares Function--------------##
#This is a function that will output the least squares estimated values and constants from x and y data
def Calc_Least_Squares_Estimate(xdata, ydata):
    x_squared = np.power(xdata,2) #Square the X data
    # Quadratic Equation: ax^2+bx+c
    A = np.stack((x_squared, xdata, np.ones((len(xdata)), dtype = int)), axis=1) #Create my A Matrix using a stack
    A_trans = A.transpose() #Transpose
    ATA = A_trans.dot(A)
    ATY = A_trans.dot(ydata)
    LS_Est = (np.linalg.inv(ATA)).dot(ATY)
    LS_Value = A.dot(LS_Est)
    return LS_Est, LS_Value


##----------------------------------------------------------------##
##----------------------Problem 1.1-------------------------------##

#Loading the Video File into my Workspace and Getting Frame Count
original_video = cv.VideoCapture('ball.mov')
num_frames = int(original_video.get(cv.CAP_PROP_FRAME_COUNT))

#Setting Up my Red Channel Color Min and Max Threshold
channelRMin = 123.000
channelRMax = 254.000

channelGMin = 0.000
channelGMax = 70.000

channelBMin = 0.000
channelBMax = 94.000

Min_Thresh = np.array([channelBMin,channelGMin,channelRMin])
Max_Thresh = np.array([channelBMax,channelGMax,channelRMax])

#Setting up Counters and Empty Arrays
count = 0
pointsx= []
pointsy = []
BallHistory = []



#Checking if Video Successfully Opened
if (original_video.isOpened() == False):
    print("Error Opening File!")

# While Video is successfully loaded:
while(original_video.isOpened()):
    count = count + 1 #Increase Counter
    success, img = original_video.read() #Read Image
    if success: #If Successfully Read Image:
        #cv.imshow("Original", img); cv.waitKey(1)
        Thresholded = cv.inRange(img, Min_Thresh, Max_Thresh) #Threshold Image Based on Above Channels
        Thresholded[:,:100] = 0 # Cropping Hand out of frame because I didn't perfectly threshold it.
        Ball_Extract = np.where(Thresholded != 0) #I extract all the nonzero indicies from binary image, which is where the ball would be found per frame.
        if (Ball_Extract[0].size > 0) and (Ball_Extract[1].size > 0): #If ball is found:
            pointsx.append(int(np.mean(Ball_Extract[1]))) #Appending x locations of ball into matrix
            pointsy.append(int(np.mean(Ball_Extract[0]))) #Appending y locations of ball into matrix
        cv.imshow("Thresholded Image", Thresholded); cv.waitKey(1) #Display Black and White Thresholded Image
    else:
        original_video.release() #release video
        break #break out of loop

#Code for Plotting Ball Path
#Note: ball path is reflected since the coordinate frame of an image is different than a regular coordinate frame.
#Negating the Data to mimic proper path
plot_y_original = []
for i in range(len(pointsy)):
    plot_y_original.append(-1*pointsy[i])


##-----------Problem 1.2-----------------##
LS_Constants, y_LS = Calc_Least_Squares_Estimate(pointsx, plot_y_original) #Use the Function above to calculate the least squares constants and output estimates
print("The equation of the Least Squares Fit Curve is: %fx^2 + %fx + %f" % (LS_Constants[0],LS_Constants[1],LS_Constants[2])) #Print the Least Squares Equation

##------------Problem 1.3----------------##
#Remembering that I have flipped the y coordinates, I subtract 300 rather than add 300.
y_final = plot_y_original[0] - 300 # Getting the final position of the ball
x_final = np.roots([LS_Constants[0], LS_Constants[1], LS_Constants[2]-y_final]) #Use the roots function adjusting for the ending y position of the ball.
print("The final x position of the ball in pixels is %f" % x_final[0]) #Printing the final position of the ball.


##---------------Plotting-------------##

fig = plt.figure()
plt.title('Plotting Found Center Points vs Least Squares Fit')
plt.xlabel('X Coordinates (Pixels)')
plt.ylabel('Y Coordinates (Pixels)')
plt.plot(pointsx,plot_y_original, 'b*', label = 'Extracted Y Points')
plt.plot(pointsx,y_LS, 'r-', label = 'Least Squares Fit')
plt.legend()
plt.show()















