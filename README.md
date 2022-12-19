# Robotics-IST-Scorbot
Send commands to a serial robot (SCORBOT-ER VII) in serial form. The aim of the project is to read an image in Python and then make the robot draw the same image on a sheet of paper.



# Requirements

There is a requirements.txt with all the libraries needed to run the code.

# How to run the code

1. Put the image to be drawn by the robot inside the folder  **PutTheImageHere** (there must be only one image in the folder)

2. Open the terminal in the folder **Robotics-IST-Scorbot** (otherwise Python won't find the path to the image)
Ex: Your terminal has to show something like this "*C:\somePath\Robotics-IST-Scorbot>*"

3. Run (ourMainCode.py) and the robot will draw the image mentioned above.


## Example of it working:
1. If the image inside the folder **PutTheImageHere** is this:

![Alt text](For%20readme%20(don't%20change)/test_draw_2.png)

2. Then, the **simpleScorbotCoords.Py** will return a list with two vectors (represented by the blue and orange dots), each containing the ordered points that will be send to the robot.

![Alt text](For%20readme%20(don't%20change)/output-test-draw-2.png)

3. And, finally, the **CHARLIE CODE.py** will send these points in serial form to the robot draw the image. The image bellow is a picture of the drawing made by the robot:

PUT THE PICTURE CHARLIE WILL TAKE

*Points that were sent to robot to make the image above:*

stepsList = [array([[ 8192.90499088, -2075.66075278],
       [ 8208.1711159 , -1977.16962361],
       [ 8245.59774498, -1898.37672028],
       [ 8297.79804344, -1841.74432101],
       [ 8376.59094677, -1801.8554137 ],
       [ 8475.08207594, -1796.93085724],
       [ 8612.96965677, -1819.58381695],
       [ 8711.46078594, -1857.99535732],
       [ 8790.25368927, -1913.15038965],
       [ 8843.43889902, -2011.64151882],
       [ 8828.172774  , -2267.71845465],
       [ 8773.51019731, -2484.39893882],
       [ 8694.22483834, -2598.156193  ],
       [ 8556.3372575 , -2604.5581164 ],
       [ 8359.35499917, -2567.62394296],
       [ 8221.46741834, -2509.51417675],
       [ 8123.96120046, -2432.691096  ],
       [ 8088.01193832, -2373.5964185 ],
       [ 8077.17791411, -2314.501741  ],
       [ 8087.02702703, -2137.21770851],
       [ 8122.97628917, -1940.23545017],
       [ 8184.04078926, -1762.95141768],
       [ 8260.86387   , -1625.06383684],
       [ 8366.24937821, -1511.7990383 ],
       [ 8464.74050738, -1459.1062842 ],
       [ 8582.92986238, -1448.76471564],
       [ 8720.81744321, -1491.11590118],
       [ 8839.00679821, -1580.74282872],
       [ 8925.18653623, -1698.93218372],
       [ 8981.32647985, -1836.81976455],
       [ 9005.4568065 , -1974.70734538],
       [ 8999.0548831 , -2230.78428121],
       [ 8966.06035483, -2467.16299121],
       [ 8905.4883104 , -2683.84347538],
       [ 8839.9917095 , -2821.73105621],
       [ 8759.22898358, -2917.75990715],
       [ 8680.43608025, -2955.67899188],
       [ 8641.03962859, -2949.27706848],
       [ 7428.12137291, -2375.07378544],
       [ 6797.2856906 , -2377.53606367],
       [ 6385.59277068, -1960.9185873 ],
       [ 6365.89454485, -1957.47139778],
       [ 6247.70518985, -1990.95838169],
       [ 6134.93284696, -2089.94196651],
       [ 6066.48151219, -2227.82954734],
       [ 6044.81346377, -2376.55115238],
       [ 6163.00281877, -2410.53059194],
       [ 6284.63936329, -2536.10678163],
       [ 6366.3870005 , -2733.08903996],
       [ 6394.45697231, -2951.73934671],
       [ 6769.21571879, -2419.88724921]]), array([[ 6385.59277068, -1960.9185873 ],
       [ 6764.78361797, -1985.54136959],
       [ 7116.88940474, -2077.63057536],
       [ 7333.56988891, -2200.74448682],
       [ 7402.02122368, -2281.01475709],
       [ 7428.12137291, -2375.07378544]])]