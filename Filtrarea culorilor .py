import numpy as np
import cv2 as cv

def find_color_values_using_custom_trackbar(frame):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Initial values
    l_h, l_s, l_v = 0, 0, 0
    u_h, u_s, u_v = 255, 255, 255

    def trackbar_callback(value):
        pass

    cv.namedWindow("Trackbar")
    cv.createTrackbar("LH", "Trackbar", 0, 255, trackbar_callback)
    cv.createTrackbar("LS", "Trackbar", 0, 255, trackbar_callback)
    cv.createTrackbar("LV", "Trackbar", 0, 255, trackbar_callback)
    cv.createTrackbar("UH", "Trackbar", 255, 255, trackbar_callback)
    cv.createTrackbar("US", "Trackbar", 255, 255, trackbar_callback)
    cv.createTrackbar("UV", "Trackbar", 255, 255, trackbar_callback)

    while True:
        l_h = cv.getTrackbarPos("LH", "Trackbar")
        l_s = cv.getTrackbarPos("LS", "Trackbar")
        l_v = cv.getTrackbarPos("LV", "Trackbar")
        u_h = cv.getTrackbarPos("UH", "Trackbar")
        u_s = cv.getTrackbarPos("US", "Trackbar")
        u_v = cv.getTrackbarPos("UV", "Trackbar")

        l = np.array([l_h, l_s, l_v])
        u = np.array([u_h, u_s, u_v])
        mask_table_hsv = cv.inRange(frame_hsv, l, u)

        res = cv.bitwise_and(frame, frame, mask=mask_table_hsv)
        cv.imshow("Frame", cv.resize(frame, (640, 480)))  # Adjust the size as needed
        cv.imshow("Mask", cv.resize(mask_table_hsv, (640, 480)))  # Adjust the size as needed
        cv.imshow("Res", cv.resize(res, (640, 480)))  # Adjust the size as needed

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cv.destroyAllWindows()

img = cv.imread("imagini_auxiliare/02.jpg")
cv.namedWindow('img_initial', cv.WINDOW_NORMAL)
cv.imshow('img_initial', img)
cv.waitKey(0)
cv.destroyAllWindows()

find_color_values_using_custom_trackbar(img)

low_yellow = (15, 105, 105)
high_yellow = (90, 255, 255)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
cv.namedWindow('mask_yellow_hsv', cv.WINDOW_NORMAL)
cv.imshow('mask_yellow_hsv', mask_yellow_hsv)
cv.waitKey(0)
cv.destroyAllWindows()
