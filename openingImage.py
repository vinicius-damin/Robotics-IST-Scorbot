import numpy as np
import pandas as pd
import cv2
import os

# Open the image inside "PutTheImageHere" folder in grayscale mode and save it to csv and xlsx
def main():
    # path: directory where the git rep is located
    path = os.getcwd() + r'\PutTheImageHere'
    imgFilePath = path + fr'\{os.listdir(path)[0]}'

    # read image
    img = cv2.imread(imgFilePath,0)
    imgDataFrame = pd.DataFrame(img[0:2500,0:2000])

    cv2.imshow('image', img)
    cv2.waitKey()

    print(f"Finished showing the image")

if __name__ == "__main__":
    main()



