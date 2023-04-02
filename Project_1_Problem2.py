## Brendan Neal
## ENPM673 Spring 2023
## Project 1 Code

##Note, this script is only for question 2. There is a differnt script for question 1.

#Importing Proper Libraries
import numpy as np
from matplotlib import pyplot as plt
import math
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


##--------------------------Defining Necessary Functions to solve problems------------------------------##

# Defining my 3D Covariance Matrix Function
#Want a 3-3 Matrix
#Sx COV(y,x) COV(y,y)
#COV(x,y) Sy COV(z,y)
#COV(x,z) COV(y,z) Sz

def Cov_Matrix_Calc(Xdata,Ydata,Zdata):
    xmean = Xdata.mean() # Mean of Xdata
    ymean = Ydata.mean() # Mean of Ydata
    zmean = Zdata.mean() # Mean of Zdata
    n = len(Xdata) # Get n for known dataset
    Sx = (1/n)*np.sum((Xdata-xmean)*(Xdata-xmean).T) #Calculate Variance of X
    Sy = (1/n)*np.sum((Ydata-ymean)*(Ydata-ymean).T) #Calculate Variance of Y
    Sz = (1/n)*np.sum((Zdata-zmean)*(Zdata-zmean).T) #Calculate Variance of Z
    COVXY = (1/n)*np.sum((Xdata-xmean)*(Ydata-ymean)) #Calculate Covariance of X and Y
    COVXZ = (1/n)*np.sum((Xdata-xmean)*(Zdata-zmean)) #Calculate Covariance of X and Z
    COVYZ = (1/n)*np.sum((Ydata-ymean)*(Zdata-zmean)) #Calculate Covarience of Y and Z
    COVMATRIX = np.array([[Sx, COVXY, COVXZ],[COVXY, Sy, COVYZ],[COVXZ, COVYZ,Sz]]) #Format calculated values into proper matrix format.

    return COVMATRIX

## Defining my 3-D Surface Normal Calculation Function

def Surface_Normal_Calc(covmatrix):
    evalues, evectors = np.linalg.eig(covmatrix) #Calculate the eigenvalues and eigenvectors of the covariance matrix
    min_idx = np.argmin(evalues) #Get the index of the minimum eigenvalue
    evec_min = evectors[:,min_idx] #separate the minimum eigenvector
    surf_normal = np.divide(evec_min,np.linalg.norm(evec_min)) #calculate the surface normal by normalizing the eigenvector of the min eigenvalue
    return surf_normal

## Defining my Standard Least Squares Function for 3D Surfaces
def Standard_Least_Squares_Calculation(xdata,ydata,zdata):
    #Data: (x1,y1,z1)
    #Equation of a plane: ax + by + cz + d = 0
    #Thus, our X matrix should be [xvals, yvals, 1]
    #Our Y matrix is [zvals]
    #Our B matrix is [a, b, c, d]
    X = np.stack((xdata, ydata, np.ones((len(xdata)), dtype = int)), axis=1) #Create my X Matrix using a stack
    Y = zdata
    X_trans = X.transpose() #Transpose
    XTX = X_trans.dot(X)
    XTY = X_trans.dot(Y)
    StdLS_Est = (np.linalg.inv(XTX)).dot(XTY) #Using the Standard Least Squares to Calculate Constants and Output Values
    StdLS_Value = X.dot(StdLS_Est)
    return StdLS_Est, StdLS_Value


# Defining my Total Least Squares function for 3D Surfaces

def Total_Least_Squares_Calculation(xdata,ydata,zdata):
    mean_x = np.mean(xdata)
    mean_y = np.mean(ydata) #Calculating the mean
    mean_z = np.mean(zdata)
    n = len(xdata)
    U = np.vstack(((xdata-mean_x), (ydata-mean_y), (zdata-mean_z))).T #Setting up my U matrix
    UTU = np.dot(U.transpose(),U) #Performing eigen decomposition to get the coefficients
    beta = np.dot(UTU.transpose(),UTU)
    eigvals, eigvecs = np.linalg.eig(beta)
    idx = np.argmin(eigvals)
    coeffs = eigvecs[:,idx]
    a,b,c = coeffs
    D = a*mean_x + b*mean_y + c*mean_z #Calculating the Distance Variable
    return coeffs, D

#Defining my RANSAC function for 3D Surfaces
def RANSAC_Fit(ransac_array, zdata, sample_size, threshold):
    iter_max = math.inf #Generate Temporary Max Iteration. This will change later
    iteration = 0 #Init First Iterationj
    max_inliers = 0 #Init max_inliers
    best_model = None #create best model variable
    prob_outlier = 0 #I want 0 outlier probability
    prob_des = 0.95 #I wanta  95% Accuracy Rate

    ransac_data = np.column_stack((ransac_array,zdata)) #Initialize Data Structure
    n = len(ransac_data) 
    while iteration < iter_max: #While iteration number is less than calculated max
        np.random.shuffle(ransac_data) #Shuffle the data randomly
        samples = ransac_data[:sample_size,:] #Take out random data points
        temp_matrix = samples[:,:-1] #Get the X and Y values from the random sample
        temp_z = samples[:,-1:] #Get the Z values from the random sample
        iteration_model, _ = Standard_Least_Squares_Calculation(temp_matrix[:,0], temp_matrix[:,1], temp_z) #Selected Test model; Standard Least Squares
        inliers = ransac_array.dot(iteration_model) #Calculate the Inliers
        error = np.abs(zdata-inliers.T) #Calculate the Error compared to the zdata
        inlier_count = np.count_nonzero(error < threshold) #Count the number of inliers
        #print("Inlier Count is", inlier_count, "for iteration", iteration)
        if inlier_count > max_inliers: #If the number of inliers is greater than the current max:
            max_inliers = inlier_count #Update the Max Inliers
            #print("Max Inliers:", max_inliers)
            best_model = iteration_model #Update the current best model

        prob_outlier = 1-(inlier_count/n) #Calculate the probability of an outlier
        if prob_outlier > 0: #If the probability of an outlier is greater than 0:
            iter_max = math.log(1-prob_des)/math.log(1-(1-prob_outlier)**sample_size) #Recalculate the new number of max iteration number
        print("Max Iterations:", iter_max)

        iteration+=1 #Increase Iteration Number

    return best_model


def Calculate_RANSAC(xdata, ydata, zdata):
    data_arr = np.stack((xdata, ydata, np.ones(len(xdata))),axis=1) #Arranging my data into required data structure
    threshold = np.std(zdata)/3 #Defining my threshold
    ransac_model = RANSAC_Fit(data_arr, zdata, 3, threshold) #applying my ransac fit function from above
    ransac_soln = data_arr.dot(ransac_model) #get the outputtez z values

    return ransac_model, ransac_soln


    

## Definining my "Generate Surface" Function. This aids in plotting Later
def Generate_Surface_StdLS(LS_Const, LS_Vals):
    sampx = np.linspace(-10,10,len(LS_Vals)) #Generate Sample X Data for Surface
    sampy = np.linspace(-10,10,len(LS_Vals)) #Generate Sample Y Data for Surface
    xx, yy = np.meshgrid(sampx, sampy) #Creating a Meshgrid for Surface
    zz = LS_Const[0]*xx + LS_Const[1]*yy + LS_Const[2] # Calculating the output Z Based on the Constants Derived from LS Method
    return xx, yy, zz

def Generate_Surface_TotLS(LS_Const, d, lengthdata):
    sampx = np.linspace(-10,10,len(lengthdata)) #Generate Sample X Data for Surface
    sampy = np.linspace(-10,10,len(lengthdata)) #Generate Sample Y Data for Surface
    xx, yy = np.meshgrid(sampx, sampy) #Creating a Meshgrid for Surface
    zz = (-1/LS_Const[2])*(LS_Const[0]*xx + LS_Const[1]*yy - d) # Calculating the output Z Based on the Constants Derived from LS Method
    return xx, yy, zz


        




##--------------------Importing, Reading Files, and Making Data useful.----------------------##
data_pc1 = np.genfromtxt('pc1.csv',delimiter=',', dtype=None, names =True)
data_pc2 = np.genfromtxt('pc2.csv',delimiter=',', dtype=None, names =True)
xpc1 = []
ypc1 = [] 
zpc1 = []
xpc2 = [] 
ypc2 = []
zpc2 = []

for i in range(len(data_pc1)):
    xpc1.append(data_pc1[i][0])
    ypc1.append(data_pc1[i][1])
    zpc1.append(data_pc1[i][2])
xpc1data = np.array(xpc1)
ypc1data = np.array(ypc1)
zpc1data = np.array(zpc1)

for i in range(len(data_pc2)):
    xpc2.append(data_pc2[i][0])
    ypc2.append(data_pc2[i][1])
    zpc2.append(data_pc2[i][2])
xpc2data = np.array(xpc2)
ypc2data = np.array(ypc2)
zpc2data = np.array(zpc2)


##---------------Problem 2.1---------------------##
##----------------part a-------------------------##
COVMATRIX_PC1 = Cov_Matrix_Calc(xpc1data,ypc1data,zpc1data) #I used my function from above to calculate the covariance matrix.
print("The covariance matrix for PC1 is:") #Printing
print(COVMATRIX_PC1) #Printing

##----------------part b-------------------------##
SurfNorm = Surface_Normal_Calc(COVMATRIX_PC1) #I used my function from above to calculate the surface normal vector
SurfNormMag = math.sqrt(sum(pow(element,2) for element in SurfNorm)) #Calculate the margnitude of the surface normal
print("The surface normal vector is:")
print(SurfNorm)
print("The magnitude of the surface normal is: %f" % SurfNormMag)
print("The direction of the surface normal vector is %f in the x, %f in the y, and %f in the z direction" % (SurfNorm[0], SurfNorm[1], SurfNorm[2]))


##-------------Problem 2.2----------------------##

#Setting Up Standard Least Squares for Plotting
Std_LS_Constants_PC1, Std_LS_Values_PC1 = Standard_Least_Squares_Calculation(xpc1data,ypc1data,zpc1data)
Std_LS_Constants_PC2, Std_LS_Values_PC2 = Standard_Least_Squares_Calculation(xpc2data,ypc2data,zpc2data)
PC1X_Std, PC1Y_Std, PC1Z_Std = Generate_Surface_StdLS(Std_LS_Constants_PC1, Std_LS_Values_PC1)
PC2X_Std, PC2Y_Std, PC2Z_Std = Generate_Surface_StdLS(Std_LS_Constants_PC2, Std_LS_Values_PC2)

#Setting Up Total Least Squares for Plotting
Tot_LS_Constants_PC1, D_PC1 = Total_Least_Squares_Calculation(xpc1data,ypc1data,zpc1data)
Tot_LS_Constants_PC2, D_PC2 = Total_Least_Squares_Calculation(xpc2data,ypc2data,zpc2data)
PC1X_Tot, PC1Y_Tot, PC1Z_Tot = Generate_Surface_TotLS(Tot_LS_Constants_PC1, D_PC1, xpc1data)
PC2X_Tot, PC2Y_Tot, PC2Z_Tot = Generate_Surface_TotLS(Tot_LS_Constants_PC2, D_PC2, xpc2data)

## Setting up RANSAC for Plotting
RANSAC_Const_PC1, Ransac_Soln_PC1 = Calculate_RANSAC(xpc1data,ypc1data,zpc1data)
RANSAC_Const_PC2, Ransac_Soln_PC2 = Calculate_RANSAC(xpc2data,ypc2data,zpc2data)
PC1X_RANSAC, PC1Y_RANSAC, PC1Z_RANSAC = Generate_Surface_StdLS(RANSAC_Const_PC1, Ransac_Soln_PC1)
PC2X_RANSAC, PC2Y_RANSAC, PC2Z_RANSAC = Generate_Surface_StdLS(RANSAC_Const_PC2, Ransac_Soln_PC2)




#Setting Up RANSAC for Plotting

fig = plt.figure()
#----------PC1 Plot-------------#
ax = fig.add_subplot(1, 2, 1, projection='3d') 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('PC1 Data')

#Plotting Raw Data
ax.scatter(xpc1,ypc1,zpc1, label = 'Raw Data')

#Plotting Standard LS Fit
ax.plot_surface(PC1X_Std,PC1Y_Std,PC1Z_Std, color = 'r', label = 'Std LS Fit')

#Plotting Total LS Fit
ax.plot_surface(PC1X_Tot,PC1Y_Tot,PC1Z_Tot, color = 'g', label = 'Tot LS Fit')

#Plotting RANSAC Fit
ax.plot_surface(PC1X_RANSAC, PC1Y_RANSAC, PC1Z_RANSAC, color = 'y', label = 'RANSAC Fit')


#-----------PC2 Plot------------#
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('PC2 Data')

#Plotting Raw Data
ax.scatter(xpc2,ypc2,zpc2, label = 'Raw Data')

#Plotting Standard LS Fit
ax.plot_surface(PC2X_Std,PC2Y_Std,PC2Z_Std, color = 'r', label = 'Std LS Fit')

#Plotting Total LS Fit
ax.plot_surface(PC2X_Tot,PC2Y_Tot,PC2Z_Tot, color = 'g', label = 'Tot LS Fit')

#Plotting RANSAC Fit
ax.plot_surface(PC2X_RANSAC, PC2Y_RANSAC, PC2Z_RANSAC, color = 'y', label = 'RANSAC Fit')


plt.show()


