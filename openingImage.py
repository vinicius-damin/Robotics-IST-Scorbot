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

    # Threshold for an optimal value, it may vary depending on the image.
    cornersImg = image.copy()
    cornersImg[dst==255]=[127]

    return cornersImg, centroids[:, [1, 0]]

# create a square with L = 2*radius, center = centerPoint and find the intersection with the refined image
def findIntersection(refinedImage, centerPoint, radius, epsilon):
    intersections = []

    imageDrawed = refinedImage.copy()
    centerPointX = centerPoint[0]
    centerPointY = centerPoint[1]

    x = centerPointX-radius
    y = centerPointY-radius
    for i, color in np.ndenumerate(imageDrawed[x, y : y+2*radius]):
        # if the cell is black
        if color == 0:
            imageDrawed[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 150
            intersections.append((x,y))
        # color the path
        imageDrawed[x,y] = 50
        y += 1

    x = centerPointX-radius
    y = centerPointY+radius
    for i, color in np.ndenumerate(imageDrawed[x : x+2*radius, y]):
        # if the cell is black
        if color == 0:
            imageDrawed[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 160
            intersections.append((x,y))
        # color the path
        imageDrawed[x,y] = 70
        x += 1
 
    x = centerPointX+radius
    y = centerPointY+radius
    for i, color in np.ndenumerate(imageDrawed[x, y : y-2*radius :-1]):
        # if the cell is black
        if color == 0:
            imageDrawed[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 170
            intersections.append((x,y))
        # color the path
        imageDrawed[x,y] = 90
        y -= 1

    x = centerPointX+radius
    y = centerPointY-radius
    for i, color in np.ndenumerate(imageDrawed[x:x-2*radius : -1, y]):
        # if the cell is black
        if color == 0:
            imageDrawed[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 180
            intersections.append((x,y))
        # color the path
        imageDrawed[x,y] = 110
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

def iterThroughFeature(refinedFeatureImg, stepRadius, epsilon):
    return


#############################################################################################
################################## OUR ALGORITHM CLASS ######################################
#############################################################################################

class Image:
    def __init__(self, featureNumber, startingFeature, FeatureList):
        self.featureNumber = featureNumber
        self.startingFeature = startingFeature
        self.FeatureList = FeatureList

class ImgFeature:
    def __init__(self, id, startingPoint, endingPoint, middlePoints):
        self.id = id
        self.startingPoint = startingPoint
        self.endingPoint = endingPoint
        self.middlePoints = middlePoints
 

#############################################################################################
################################## MAIN CODE ################################################
#############################################################################################
def main():
    # Read Image
    img = readImage()

    # Manually define parameters of the algorithm
    radiusStartingPoint = 80 # large enough radius
    radiusStep = radiusStartingPoint/2
    epsilon = 15 # since the stroke is 5p width, with 15p we make sure
    
    # Expand boundary according to the parameter to avoid out of bounds error
    img = expandBoundary(img, radiusStartingPoint) 

    # Make a copy to preserve original img
    imgCopy = img.copy()

    # Find the corners of the image
    cornersImg, centroids = findCorners(imgCopy)
    centroids = centroids[1:]

    # Dilate to lower the number of black pixels
    refinedImage = dilateImage(imgCopy,4)

    # Turn most non white pixels to black
    refinedImage = blackenTheImage(refinedImage)

    print('Centroids and corners image:')
    print(centroids)
    pyplotImage(cornersImg)

    # Find a good starting point with refined image
    startingPoint = findStartingPoints(refinedImage, centroids, radiusStartingPoint, epsilon)
    print('startingPoint:')
    print(startingPoint)

    
    # Step through all the image to find points for the robot

    print('Will show all intersections in all centroids')
    for centroid in centroids:
        imgDrawed, intersections = findIntersection(refinedImage,centroid, radiusStartingPoint, epsilon)
        print(intersections)
        pyplotImage(imgDrawed)
    

    # Create Figure with opencv
    #createFigureWithOpenCV(img)

    print(f"Finished showing the image")



if __name__ == "__main__":
    main()



