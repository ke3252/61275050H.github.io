import cv2
import numpy as np

cap = cv2.VideoCapture(0)
kernel = np.ones((3, 3), np.uint8)

def empty(v):
    pass


cv2.namedWindow('TrackBar')
cv2.resizeWindow('TrackBar', 640, 320)

cv2.createTrackbar('Hue Min','TrackBar', 0, 179,empty)
cv2.createTrackbar('Hue Max','TrackBar', 179, 179,empty)
cv2.createTrackbar('Sat Min','TrackBar', 0, 255,empty)
cv2.createTrackbar('Sat Max','TrackBar', 255, 255,empty)
cv2.createTrackbar('Val Min','TrackBar', 0, 255,empty)
cv2.createTrackbar('Val Max','TrackBar', 255, 255,empty)


while True:

    ret, img = cap.read()
    results = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos('Hue Min', 'TrackBar')
    h_max = cv2.getTrackbarPos('Hue Max', 'TrackBar')
    s_min = cv2.getTrackbarPos('Sat Min', 'TrackBar')
    s_max = cv2.getTrackbarPos('Sat Max', 'TrackBar')
    v_min = cv2.getTrackbarPos('Val Min', 'TrackBar')
    v_max = cv2.getTrackbarPos('Val Max', 'TrackBar')

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # img = cv2.resize(img, (1200, 1200))
    # img = img[700:1200,400:1000]

    imgCountour = result.copy()
    img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(result, 200, 250)
    dilate = cv2.dilate(canny, kernel, iterations=3)
    erode = cv2.erode(dilate, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if area > 10000 and area < 12000 and len(approx) >= 4 and len(approx) <= 12:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(results, (x, y), (x + w, y + h), (0, 0, 0), 4)
            cv2.putText(results, 'dice', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('result', result)
    #cv2.imshow('mask', mask)
    cv2.imshow('results', results)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()