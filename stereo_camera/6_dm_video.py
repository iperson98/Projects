import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from datetime import datetime


def stereo_depth_map(sbm, rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    local_max = disparity.max()
    local_min = disparity.min()
    disparity_grayscale = (disparity - local_min) * (65535.0 / (local_max - local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    cv2.imshow("Image", disparity_color)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        quit();
    return disparity_color


if __name__ == '__main__':
    try:

        # Camera settimgs
        cam_width = 320
        cam_height = 240

        # Final image capture settings
        scale_ratio = 0.5
        # Camera resolution height must be dividable by 16, and width by 32
        cam_width = int((cam_width + 31) / 32) * 32
        cam_height = int((cam_height + 15) / 16) * 16
        print("Used camera resolution: " + str(cam_width) + " x " + str(cam_height))

        # Buffer for captured image settings
        img_width = int(cam_width * scale_ratio)
        img_height = int(cam_height * scale_ratio)
        capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        print("Scaled image resolution: " + str(img_width) + " x " + str(img_height))

        camera1 = cv2.VideoCapture(1)
        print("Setting the custom Width and Height")
        camera1.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        camera2 = cv2.VideoCapture(2)
        print("Setting the custom Width and Height")
        camera2.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        # Implementing calibration data
        print('Read calibration data and rectifying stereo pair...')
        calibration = StereoCalibration(input_folder='calib_result')

        # Initialize interface windows
        cv2.namedWindow("Image")
        cv2.moveWindow("Image", 50, 100)
        cv2.namedWindow("left")
        cv2.moveWindow("left", 450, 100)
        cv2.namedWindow("right")
        cv2.moveWindow("right", 850, 100)

        disparity = np.zeros((img_width, img_height), np.uint8)
        sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)

        while True:
            # Read Camera 1
            check1, frame1 = camera1.read()
            # Convert frame to gray scale image
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            # Read Camera 2
            check2, frame2 = camera2.read()
            # Convert frame to gray scale image
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            t1 = datetime.now()
            pair_img_L = frame1
            pair_img_R = frame2
            imgLeft = pair_img_L
            imgRight = pair_img_R
            rectified_pair = calibration.rectify((imgLeft, imgRight))
            disparity = stereo_depth_map(sbm, rectified_pair)

            # Replacement Code for Disparity
            dmLeft = rectified_pair[0]
            dmRight = rectified_pair[1]
            disparity = sbm.compute(dmLeft, dmRight)

            f = 1
            b = 54

            distance = b*f/disparity

            print(distance)

            # show the frame
            cv2.imshow("left", imgLeft)
            cv2.imshow("right", imgRight)

            t2 = datetime.now()
            print("DM build time: " + str(t2 - t1))

    except KeyboardInterrupt:
        camera1.release()
        camera2.release()
        exit()
