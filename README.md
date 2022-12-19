# Robotics-IST-Scorbot
Send commands to a serial robot (SCORBOT-ER VII) in serial form. The aim of the project is to read an image in Python and then make the robot draw the same image on a sheet of paper



# Requirements

There is a requirements.txt with all the libraries needed to run the code.

# How to run the code

1. Put the image to be drawn by the robot inside the folder  **PutTheImageHere**

2. Open the terminal in the folder **Robotics-IST-Scorbot** (otherwise Python won't find the path to the image)
Ex: Your terminal has to show something like this "*C:\somePath\Robotics-IST-Scorbot>*"

3. Run (ourMainCode.py) and the robot will draw the image mentioned above.


# Example:
If the image inside the folder **PutTheImageHere** is this:
![Alt text](../test_draw_2.png)

The **simpleScorbotCoords.Py** will return a list with two vectors (represented by the blue and orange dots), each containing the ordered points that will be send to the robot.
![Alt text](../output-test-draw-2.png)

And, finally, the **CHARLIE CODE.py** will send these points in serial form to the robot draw the image.

PUT THE PICTURE CHARLIE WILL TAKE