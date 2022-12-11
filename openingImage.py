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

def dilateImage(image, numIter):
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((2, 2), np.uint8) ## 6x6 good
    eroded = cv2.dilate(image, kernel, iterations=numIter)
    return eroded
    #4375 1800

def blackenTheImage(image):
    # All not almost white pixels are turned black
    result = image.copy()
    result[result<230] = 0
    return result

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

def findIntersection(image, centerPoint, radius, epsilon):
    intersections = []

    imageCopy = image.copy()
    centerPointX = centerPoint[0]
    centerPointY = centerPoint[1]

    x = centerPointX-radius
    y = centerPointY-radius
    for i, color in np.ndenumerate(imageCopy[x, y : y+2*radius]):
        # if the cell is black
        if color == 0:
            print("Found a black pixels")
            imageCopy[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 150
            pyplotImage(imageCopy)
        
        print(f'Orig color = {color} changed x = {x}, y = {y} to 150')
        imageCopy[x,y] = 50
        #pyplotImage(imageCopy)

        y += 1

    x = centerPointX-radius
    y = centerPointY+radius
    for i, color in np.ndenumerate(imageCopy[x : x+2*radius, y]):
        
        # if the cell is black
        if color == 0:
            print("Found a black pixels")
            imageCopy[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 160
            pyplotImage(imageCopy)

        print(f'changed x = {x}, y = {y} to 160')
        imageCopy[x,y] = 70
        x += 1
    pyplotImage(imageCopy)


    x = centerPointX+radius
    y = centerPointY+radius
    for i, color in np.ndenumerate(imageCopy[x, y : y-2*radius :-1]):
        
        # if the cell is black
        if color == 0:
            print("Found a black pixels")
            imageCopy[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 170
            pyplotImage(imageCopy)
        
        print(f'changed x = {x}, y = {y} to 170')
        imageCopy[x,y] = 90
        
        y -= 1
    
    pyplotImage(imageCopy)

    x = centerPointX+radius
    y = centerPointY-radius
    for i, color in np.ndenumerate(imageCopy[x:x-2*radius : -1, y]):
        # if the cell is black
        if color == 0:
            print("Found a black pixels")
            imageCopy[int(x-epsilon/2):int(x+epsilon/2)+1 , int(y-epsilon/2):int(y+epsilon/2)+1] = 180
            pyplotImage(imageCopy)

        print(f'changed x = {x}, y = {y} to 180')
        imageCopy[x,y] = 110

        x -= 1
    
    pyplotImage(imageCopy)
        




def findStartingPoints(refinedImage, centerOfCorners):
    startingImg = refinedImage.copy()
    #check if there is only 1 black pixel in a radius epsilon which doesn't neighbor another
    #black one withing a range of 5 pixels
    radius = 40 # large enough radius
    epsilon = 10 # since the stroke is 5p width, with 10p we make sure
    for centroid in centerOfCorners:
        counter = 0

        pass

        # Blue color in BGR
        color = (0, 0, 150)
        # Line thickness of 2 px
        thickness = 1
        image = cv2.circle(startingImg, tuple(centroid), radius, color, thickness)
        # Displaying the image
        
    #pyplotImage(image)
    #pyplotImage(refinedImage)

def expandBoundary(image):
    # Add 50p to each side so no out of bounds error
    a = np.zeros((image.shape[0], 50))
    a[a==0] = 255
    image = np.concatenate((image, a), axis=1)
    image = np.concatenate((a, image), axis=1)

    b = np.zeros((50, image.shape[1]))
    b[b==0] = 255
    image = np.concatenate((image, b), axis=0)
    image = np.concatenate((b, image), axis=0)
    return image

def reduceBoundary(image):
    image = image[50:-50,50:-50]
    return image



def main():
    # Read Image
    img = readImage()
    


    # Expand boundary by 50p to avoid out of bounds error
    img = expandBoundary(img)  

    # Make a copy to preserve original img
    imgCopy = img.copy()

    # Find the corners
    cornersImg, centroids = findCorners(imgCopy)
    centroids = centroids[1:]

    # Dilate
    imgCopy = dilateImage(imgCopy,4)

    # Turn black
    imgCopy = blackenTheImage(imgCopy)

    print(centroids)
    pyplotImage(cornersImg)

    radius = 40 # large enough radius
    epsilon = 10 # since the stroke is 5p width, with 10p we make sure

    print('started findIntersection')
    findIntersection(imgCopy,centroids[4], radius, epsilon)
    

    # Colormap Pyplot
    pyplotImage(img)
    pyplotImage(imgCopy)

    pyplotImage(cornersImg)

    findStartingPoints(img, centroids)

    # Create Figure with opencv
    #createFigureWithOpenCV(img)

    print(f"Finished showing the image")



if __name__ == "__main__":
    main()



