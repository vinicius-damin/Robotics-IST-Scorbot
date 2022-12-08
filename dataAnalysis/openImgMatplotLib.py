import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

def analyze():
    # Open the image in grayscale mode
    path = os.getcwd()
    pathToFolder = path + r'\PutTheImageHere'
    imgFilePath = pathToFolder + fr'\{os.listdir(pathToFolder)[0]}'
    print(imgFilePath)
    img = cv2.imread(imgFilePath,0)

    fig = plt.figure()
    plt.imshow(img, interpolation='none', cmap='gray')
    plt.colorbar()
    plt.show()


    # Numeric data decided after looking at the figure above
    imgZoom = img[1610:1643, 1078:1107]

    # Save to .csv
    imgDataFrame = pd.DataFrame(imgZoom)
    imgDataFrame.to_csv(fr"{path}\dataAnalysis\csvImgZoom.csv", index=False)

if __name__ == "__main__":
    analyze()