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

    return cornersImg, centroids

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

def isStartingPoint(refinedImage, centerOfCorners):
    pass

def main():
    # Read Image
    img = readImage()
    
    # Find the corners
    cornersImg, centroids = findCorners(img)

        # Dilate
    img = dilateImage(img,4)

    # Turn black
    img = blackenTheImage(img)

    # Colormap Pyplot
    pyplotImage(img)
    pyplotImage(cornersImg)

    # Create Figure with opencv
    #createFigureWithOpenCV(img)

    print(f"Finished showing the image")



if __name__ == "__main__":
    main()



