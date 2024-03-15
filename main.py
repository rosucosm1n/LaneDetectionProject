import cv2
import numpy as np
import functie as fct

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

w = 360
h = 250

upper_left = (w * 0.43, h * 0.77)     #colt stg sus trapez
upper_right = (w * 0.55, h * 0.77)    #colt drt sus trapez
lower_left = (w * 0.02, h)
lower_right = (w, h)
trapezoid = np.array([upper_right, upper_left, lower_left, lower_right], np.int32)

top_right = (w, h * 0)                #colt stg sus fullscreen
top_left = (w * 0, h * 0)             #colt drt sus fullscreen
fullscreen = np.array([top_right, top_left, lower_left, lower_right], np.int32)

trapezoidfloat = np.float32(trapezoid)
fullscreenfloat = np.float32(fullscreen)

sobel_vertical = np.float32([[-1, -2, -1],
                             [0, 0, 0],
                             [+1, +2, +1]])
sobel_horizontal = np.transpose(sobel_vertical)

while True:
    ret, frame = cam.read()

    if ret is False:
        break

    frame = cv2.resize(frame, (w, h))
    cv2.imshow('Original', frame)

    frame2 = frame
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', frame2)

    trapez = np.zeros((h, w), np.uint8)
    trapez = cv2.fillConvexPoly(trapez, trapezoid, 1)
    road = trapez * frame2
    cv2.imshow('Road', road)

    stretched = cv2.getPerspectiveTransform(trapezoidfloat, fullscreenfloat)
    stretched = cv2.warpPerspective(road, stretched, (w, h))
    cv2.imshow('Stretched', stretched)

    blurred = cv2.blur(stretched, (5, 5))
    cv2.imshow('Blurred', blurred)

    blurred1 = np.float32(blurred)          #transformare frame uint->float pt aplicare filtre
    blurred2 = np.float32(blurred)          #same

    verticalfilter = cv2.filter2D(blurred1, -1, sobel_vertical)
    horizontalfilter = cv2.filter2D(blurred2, -1, sobel_horizontal)

    sobelfilter = np.sqrt((verticalfilter)**2 + (horizontalfilter)**2)  #suprapunere filtre vert+hor
    sobelfilter = cv2.convertScaleAbs(sobelfilter)      #conversie din float inapoi in uint
    cv2.imshow('Sobel Aplicat', sobelfilter)

    ret, binarized = cv2.threshold(sobelfilter, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binarizata', binarized)

    binarcopy = binarized.copy()

    #############pasul 9:

    binarcopy[0:h, 0:round(w*0.04)] = 0
    binarcopy[0:h, round(w*0.95):w] = 0
    binarcopy[round(h*0.96):h, 0:w] = 0

    lefthalf = binarcopy[:, 0:round(w*0.5)]
    righthalf = binarcopy[:, round(w*0.5):w]

    cv2.imshow('Binarizata finala', binarcopy)

    lefthalf = np.argwhere(lefthalf > 1)
    righthalf = np.argwhere(righthalf > 1)

    left_xs = lefthalf[:, 1]
    left_ys = lefthalf[:, 0]
    right_xs = righthalf[:, 1] + round(w * 0.5)
    right_ys = righthalf[:, 0]

    #############pasul 10:

    (b1, a1) = np.polynomial.polynomial.polyfit(left_xs, left_ys, 1)
    (b2, a2) = np.polynomial.polynomial.polyfit(right_xs, right_ys, 1)

    left_top_y = 0
    left_top_x = round((left_top_y - b1) / a1)
    left_bottom_y = h
    left_bottom_x = round((left_bottom_y - b1) / a1)

    right_top_y = 0
    right_top_x = round((right_top_y - b2) / a2)
    right_bottom_y = h
    right_bottom_x = round((right_bottom_y - b2) / a2)

    frame_with_left_line = binarcopy.copy()
    frame_with_right_line = binarcopy.copy()
    # Draw left line
    cv2.line(frame_with_left_line, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), 120, 2)

    # Draw right line
    cv2.line(frame_with_right_line, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), 120, 2)

    cv2.imshow('Left Lines', frame_with_left_line)
    cv2.imshow('Right Lines', frame_with_right_line)

    ####################pasul 11:
    almost_final_left_frame = np.zeros((h,w), np.uint8)
    almost_final_right_frame = np.zeros((h,w), np.uint8)
    cv2.line(almost_final_left_frame, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), 120, 3)
    cv2.line(almost_final_right_frame, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), 120, 2)

    (pct1, pct2) = fct.getLinePoints(almost_final_left_frame, trapezoidfloat, h, w)
    (pct3, pct4) = fct.getLinePoints(almost_final_right_frame, trapezoidfloat, h, w)

    final_frame = frame.copy()
    cv2.line(final_frame, pct1, pct2, (50, 50, 250), 2)
    cv2.line(final_frame, pct3, pct4, (50, 250, 50), 2)

    cv2.imshow('Rezultat', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindow()