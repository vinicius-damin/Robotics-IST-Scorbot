import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

# Open the image inside "PutTheImageHere" folder in grayscale mode and save it to csv and xlsx
def readImage():
    # path: directory where the git rep is located
    path = os.getcwd() + r'\PutTheImageHere'
    imgFilePath = path + fr'\{os.listdir(path)[0]}'

    # read image in gray (_,0)
    img = cv2.imread(imgFilePath,0)
    imgDataFrame = pd.DataFrame(img[0:2500,0:2000])

    #cv2.imshow('image', img)
    #cv2.waitKey()

    return img

##############################################################################################
################################## IMG PROCESSING FUNCTIONS ##################################
##############################################################################################
def expandBoundary(image, length = 100):
    # Add n = lenght white pixels to each side so no out of bounds error
    a = np.zeros((image.shape[0], length))
    a[a==0] = 255
    image = np.concatenate((image, a), axis=1)
    image = np.concatenate((a, image), axis=1)

    b = np.zeros((length, image.shape[1]))
    b[b==0] = 255
    image = np.concatenate((image, b), axis=0)
    image = np.concatenate((b, image), axis=0)
    return image

def reduceBoundary(image):
    image = image[50:-50,50:-50]
    return image

def dilateImage(image, numIter=1):
    imageCopy = image.copy()
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((2, 2), np.uint8) ## 6x6 good
    dilated = cv2.dilate(imageCopy, kernel, iterations=numIter)
    return dilated
    #4375 1800

def blackenTheImage(image):
    # All not almost white pixels are turned black
    result = image.copy()
    result[result<230] = 0
    return result

########################################################################################
################################## PLOTTING FUNCTIONS ##################################
########################################################################################
def pyplotImage(image):
    fig = plt.figure()
    plt.imshow(image, interpolation='none', cmap='gray')
    plt.colorbar()
    plt.show()
    return

def createFigureWithOpenCV(image):
    # Create window with freedom of dimensions
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)  
    imS = cv2.resize(image, (960, 540)) # Resize image
    cv2.imshow('output',imS)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


#############################################################################################
################################## OUR ALGORITHM FUNCTIONS ##################################
#############################################################################################
def findCorners(image):
    gray = image
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,30,3,0.05)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # make all corners white in dst
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    # calculate the centroids of the corners
    _, _, _, centroids = cv2.connectedComponentsWithStats(dst)
    centroids = (np.rint(centroids)).astype(int)
    centroids = centroids[:, [1, 0]]
    centroids = centroids[1:]
    centroids = [tuple(element) for element in centroids]

    # Threshold for an optimal value, it may vary depending on the image.
    cornersImg = image.copy()
    cornersImg[dst==255]=[127]

    return cornersImg, centroids

# create a square with L = 2*radius, center = centerPoint and find the intersection with the refined image
def findIntersection(refinedImage, centerPoint, radius, epsilon):
    intersections = []

    imageDrawed = refinedImage.copy()
    centerPointX = centerPoint[0]
    centerPointY = centerPoint[1]
    radius = int(radius)

    x = centerPointX-radius
    y = centerPointY-radius
    for i, color in np.ndenumerate(imageDrawed[x, y : y+2*radius]):
        # if the cell is black
        if color == 0:
            imageDrawed[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 150
            intersections.append((x,y))
        # color the path
        imageDrawed[x,y] = 120
        y += 1

    x = centerPointX-radius
    y = centerPointY+radius
    for i, color in np.ndenumerate(imageDrawed[x : x+2*radius, y]):
        # if the cell is black
        if color == 0:
            imageDrawed[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 150
            intersections.append((x,y))
        # color the path
        imageDrawed[x,y] = 120
        x += 1
 
    x = centerPointX+radius
    y = centerPointY+radius
    for i, color in np.ndenumerate(imageDrawed[x, y : y-2*radius :-1]):
        # if the cell is black
        if color == 0:
            imageDrawed[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 150
            intersections.append((x,y))
        # color the path
        imageDrawed[x,y] = 120
        y -= 1

    x = centerPointX+radius
    y = centerPointY-radius
    for i, color in np.ndenumerate(imageDrawed[x:x-2*radius : -1, y]):
        # if the cell is black
        if color == 0:
            imageDrawed[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 150
            intersections.append((x,y))
        # color the path
        imageDrawed[x,y] = 120
        x -= 1

    return imageDrawed, intersections
        

def findStartingPoints(refinedImage, centerOfCorners, radius, epsilon):
    # Find out how many lines are leaving each 'corner'
    cornersIntersections = []
    for centroid in centerOfCorners:
        _, intersections = findIntersection(refinedImage, centroid, radius, epsilon)
        cornersIntersections.append(intersections)

    # Iterate trough all corners, if there is one with just 1 line leaving it, then it is the starting point
    for idx, element in enumerate(cornersIntersections):
        if len(element) == 1:
            return centerOfCorners[idx]

    # In the case where there is not a good starting point, start it where 2 lines are connected
    for idx, element in enumerate(cornersIntersections):
        if len(element) == 2:
            return centerOfCorners[idx]
    
    # If there aren't any good starting points, start with the first.
    print(f"NO GOOD STARTING POINT FOUND: \n Search Radius = {radius} \n Epsilon = {epsilon}")
    return centerOfCorners[0]

# defines features
def findFeatures(refinedFeatureImg, startPoint, centroids, stepRadius, epsilon):
    stepImg = refinedFeatureImg.copy()
    stepsList = [[startPoint]]
    stepRadius = int(stepRadius)
    centroidsCopy = centroids.copy()
    centroidsCopy.remove(startPoint)


    # While there is a next step to go:
    while stepsList[-1] != []:
        # current coords
        x = stepsList[-1][0][0]
        y = stepsList[-1][0][1]
        
        # Centroid Stpe: if there is a centroid near, go directly to it
        centroidSearchingArea = []
        centroidRadius = 3*stepRadius
        for ix in range(x-centroidRadius,x+centroidRadius+1):
            for jy in range(y-centroidRadius,y+centroidRadius+1):
                centroidSearchingArea.append((ix,jy))
        
        if (any((match := item) in centroidsCopy for item in centroidSearchingArea)):
            # 👇️ this runs
            stepsList.append([match])
            centroidsCopy.remove(match)
            # Make the little gray square, epsilon must be way bigger to cover for corner errors
            stepImg[int(stepsList[-1][0][0]-epsilon*2):int(stepsList[-1][0][0]+epsilon*2)+1 , int(stepsList[-1][0][1]-epsilon*2):int(stepsList[-1][0][1]+epsilon*2)+1] = 150
            thickness = 15
        
        
        # Normal step: check available black points inside the given radius
        else:
            stepImg, stepIntersection = findIntersection(stepImg, stepsList[-1][0], stepRadius, epsilon)
            stepsList.append(stepIntersection)
            thickness = 7
        
        if stepsList[-1] == []:
            return stepImg, stepsList[:-1]
            
        # remove path travelled (paint it almost white)
        start_point = (stepsList[-2][0][1],stepsList[-2][0][0])
        end_point = (stepsList[-1][0][1], stepsList[-1][0][0])
        color = (200,200,200)
        stepImg = cv2.line(stepImg, start_point, end_point, color, thickness)
            
    return stepImg, stepsList[:-1]


#############################################################################################
################################## OUR ALGORITHM CLASS ######################################
#############################################################################################

class ImageST:
    def __init__(self, featureNumber, startingFeature, FeatureList):
        self.featureNumber = featureNumber
        self.startingFeature = startingFeature
        self.FeatureList = FeatureList

class ImgFeatureNode:
    def __init__(self, id, startingPoint, endingPoint, middlePoints=[]):
        self.id = id
        self.startingPoint = startingPoint
        self.endingPoint = endingPoint
        self.middlePoints = middlePoints
 

#############################################################################################
################################## MAIN CODE ################################################
#############################################################################################
def main(stepRadius, epsilon, heightFinalImageScorbot, startingPointScorbot):
    # Read Image
    img = readImage()

    # For better search of good starting point
    radiusStartingPoint = stepRadius*2 

    # Expand boundary according to the parameter to avoid out of bounds error
    img = expandBoundary(img, radiusStartingPoint+1) 

    # Make a copy to preserve original img
    imgCopy = img.copy()

    # Find the corners of the image
    cornersImg, centroids = findCorners(imgCopy)

    # Dilate to lower the number of black pixels
    refinedImage = dilateImage(imgCopy,4)

    # Turn most non white pixels to black
    refinedImage = blackenTheImage(refinedImage)

    # Find a good starting point with refined image
    startingPoint = findStartingPoints(refinedImage, centroids, radiusStartingPoint, epsilon)

    # Step through all the image to find points for the robot
    stepImg, stepsList = findFeatures(refinedImage, startingPoint, centroids, radiusStep, epsilon) # [[(173, 806)], 

    # Turn into numpy array so it is easy to escale into the real world
    npStepsList = np.array([i[0] for i in stepsList])

    # The final image will have 29,7cm of height:
    height = heightFinalImageScorbot
    scaleFactor = height / (stepImg.shape[0]-2*(radiusStartingPoint+1))
    npStepsList = npStepsList * scaleFactor + startingPointScorbot

    # Calculation of the minimum step made by the robot
    min_step = np.min([np.sqrt((npStepsList[i][0] - npStepsList[i+1][0])**2 + (npStepsList[i][1] - npStepsList[i+1][1])**2)  for i in range(npStepsList.shape[0]-1)])

    print(f'Virtual data:')
    print(f'\t First points:\n{stepsList[0:5]}')
    print(f'\tCentroids of corners: {centroids}')
    print(f'\tStarting point: {startingPoint}')
    print(f'\tStep size: {radiusStep} pixels')
    print(f'\tNumber of steps: {len(stepsList)}')

    print(f'\nReal world data:')
    print(f'\t First points:\n{npStepsList[0:5]/10}mm')
    print(f'\tStarting point: {npStepsList[0]/10}mm')
    print(f'\tStep size: {radiusStep/10:.3f}mm')
    print(f'\tThe minimum step was {min_step/10:.3f}mm')

    print(f"\nFinished analyzing the image")

    return stepImg, npStepsList


###############################################################################################################
############################## You can change the parameters bellow! ##########################################
###############################################################################################################

if __name__ == "__main__":

    # Algorithm parameters:
    radiusStep = 40 #
    epsilon = 15 # since the stroke is 5p, with 15p we have room for error

    # Set size of the image to be drawn by SCORBOT height: 29,7cm
    height = 2970

    # Put the coords of the top left corner of the paper (like you are facing towards the front of the robot)
    startingPointScorbot = [6000, -3000] # In tenths of milimiters

    # Run Main code
    stepImg, stepsList = main(radiusStep, epsilon, height, startingPointScorbot)

    # Plot graphs to see if it worked:
    # 1. Colorbar with steps
    pyplotImage(stepImg)
    # 2. Final points in world coords
    plt.figure()
    plt.scatter(stepsList[:,0], stepsList[:,1], marker='.')
    plt.title('Points in the vector given to Scorbot')
    plt.show()



