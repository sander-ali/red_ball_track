
# import the necessary packages
from collections import deque
import numpy as np
import cv2
import imutils
import time
from imutils.video import VideoStream
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "red"
# ball in the HSV color space
lower_red1 = np.array([0,120,70])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([170,120,70])
upper_red2 = np.array([180,255,255])
#Buffer size
tr_pt = deque(maxlen=args["buffer"])

if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the video file to warm up
time.sleep(2.0)

# keep looping
while True:
    # grab the current frame
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    
    if frame is None:
        break
    
    # Converting the frame to HSV color space and resizing the frame size
    frame = imutils.resize(frame, width=600)
    filt_blur = cv2.GaussianBlur(frame, (11, 11), 0)
    frame_hsv = cv2.cvtColor(filt_blur, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "red", then remove the blobs using
    #dilations and erosion
    mask0 = cv2.inRange(frame_hsv, lower_red1, upper_red1)
    mask1 = cv2.inRange(frame_hsv, lower_red2, upper_red2)
    mask = mask0+mask1
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    
    # find contours in the mask
    cont = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)
    cent = None
    
    # only proceed if at least one contour was found
    if len(cont) > 0:
        # find the largest contour to compute the centroid
        centroid = max(cont, key=cv2.contourArea)
        ((xt, yt), rad) = cv2.minEnclosingCircle(centroid)
        temp = cv2.moments(centroid)
        cent = (int(temp["m10"] / temp["m00"]), int(temp["m01"] / temp["m00"]))
        
        # only proceed if the radius meets a minimum size
        if rad > 9:
            # draw the circle and update the tracked points
            cv2.circle(frame, (int(xt), int(yt)), int(rad),
                       (0, 255, 255), 2)
            cv2.circle(frame, cent, 5, (0, 0, 255), -1)
    # update the points queue
    tr_pt.appendleft(cent)
    
    # loop over the set of tracked points
    for k in range(1, len(tr_pt)):
        # if either of the tracked points are None, ignore
        # them
        if tr_pt[k - 1] is None or tr_pt[k] is None:
            continue
        
        # compute the line thickness
        line_th = int(np.sqrt(64 / float(k + 1)) * 2.5)
        cv2.line(frame, tr_pt[k - 1], tr_pt[k], (0, 0, 255), line_th)
        
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()
    