import cv2
import numpy as np
import mediapipe as mp
import math
def empty(v):
    pass
kernel = np.ones((3, 3), np.uint8)
cap = cv2.VideoCapture(0)
color_clubs = cv2.imread('clubs.jpg')
color_diamonds = cv2.imread('diamonds.jpg')
color_hearts = cv2.imread('hearts.jpg')
color_spade = cv2.imread('spade.jpg')

cv2.namedWindow('TrackBar')
cv2.resizeWindow('TrackBar', 640, 320)

cv2.createTrackbar('Hue Min','TrackBar', 0, 179,empty)
cv2.createTrackbar('Hue Max','TrackBar', 179, 179,empty)
cv2.createTrackbar('Sat Min','TrackBar', 0, 255,empty)
cv2.createTrackbar('Sat Max','TrackBar', 255, 255,empty)
cv2.createTrackbar('Val Min','TrackBar', 0, 255,empty)
cv2.createTrackbar('Val Max','TrackBar', 255, 255,empty)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

def hand_angle(hand_):
    angle_list = []
    # thumb
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def hand_pos(finger_angle):
    f1 = finger_angle[0]   # thumb angle
    f2 = finger_angle[1]   # index finger angle
    f3 = finger_angle[2]   # middle finger angle
    f4 = finger_angle[3]   # ring finger angle
    f5 = finger_angle[4]   # little finger angle

    if f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '1'
    elif f1>=50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '2'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5>50:
        return '3'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '4'
    else:
        return ''

with mp_hands.Hands() as hands:
    while True:
        ret, img = cap.read()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgCountour_hand_test = img.copy()
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
        #imgCountour_hand_test = result.copy()
        img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(result, 200, 250)
        dilate = cv2.dilate(canny, kernel, iterations=2)
        erode = cv2.erode(dilate, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        imgRGB = cv2.cvtColor(imgCountour_hand_test, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = []
                for landmark in hand_landmarks.landmark:
                    xPos = int(landmark.x * imgWidth)
                    yPos = int(landmark.y * imgHeight)
                    finger_points.append((xPos, yPos))
                if finger_points:
                    finger_angle = hand_angle(finger_points)# print(finger_angle)
                    text = hand_pos(finger_angle)

                    cv2.putText(img, text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10, cv2.LINE_AA)
                    if(text == '1'):

                        for cnt in contours:
                            # cv2.drawContours(imgCountour, cnt, -1, (0, 0, 255), 4)
                            print(cv2.contourArea(cnt))
                            print(cv2.arcLength(cnt, True))
                            area = cv2.contourArea(cnt)
                            if area > 50000 and area <100000:

                                pri = cv2.arcLength(cnt, True)
                                vertices = cv2.approxPolyDP(cnt, pri * 0.02, True)
                                print(len(vertices))
                                corners = len(vertices)
                                #        x, y, w, h= cv2.boundingRect(vertices)
                                #        cv2.rectangle(imgCountour, (x, y), (x+w, y+h), (0, 0, 0), 4)
                                if corners > 3 and corners < 10:
                                    pri = cv2.arcLength(cnt, True)
                                    vertices = cv2.approxPolyDP(cnt, pri * 0.02, True)
                                    print(len(vertices))
                                    corners = len(vertices)
                                    x, y, w, h = cv2.boundingRect(vertices)
                                    coin_roi = result[y:y + h, x:x + w]
                                    custom_image_resized = cv2.resize(color_spade, (w, h))
                                    result[y:y + h, x:x + w] = cv2.addWeighted(coin_roi, 0, custom_image_resized, 1, 0)
                    if (text == '2'):
                        for cnt in contours:
                            # cv2.drawContours(imgCountour, cnt, -1, (0, 0, 255), 4)
                            print(cv2.contourArea(cnt))
                            print(cv2.arcLength(cnt, True))
                            area = cv2.contourArea(cnt)
                            if area > 50000 and area <100000:

                                pri = cv2.arcLength(cnt, True)
                                vertices = cv2.approxPolyDP(cnt, pri * 0.02, True)
                                print(len(vertices))
                                corners = len(vertices)
                                #        x, y, w, h= cv2.boundingRect(vertices)
                                #        cv2.rectangle(imgCountour, (x, y), (x+w, y+h), (0, 0, 0), 4)
                                if corners > 3 and corners < 10:
                                    pri = cv2.arcLength(cnt, True)
                                    vertices = cv2.approxPolyDP(cnt, pri * 0.02, True)
                                    print(len(vertices))
                                    corners = len(vertices)
                                    x, y, w, h = cv2.boundingRect(vertices)
                                    coin_roi = result[y:y + h, x:x + w]
                                    custom_image_resized = cv2.resize(color_clubs, (w, h))
                                    result[y:y + h, x:x + w] = cv2.addWeighted(coin_roi, 0, custom_image_resized, 1, 0)
                    if (text == '3'):
                        for cnt in contours:
                            # cv2.drawContours(imgCountour, cnt, -1, (0, 0, 255), 4)
                            print(cv2.contourArea(cnt))
                            print(cv2.arcLength(cnt, True))
                            area = cv2.contourArea(cnt)
                            if area > 50000 and area <100000:

                                pri = cv2.arcLength(cnt, True)
                                vertices = cv2.approxPolyDP(cnt, pri * 0.02, True)
                                print(len(vertices))
                                corners = len(vertices)
                                #        x, y, w, h= cv2.boundingRect(vertices)
                                #        cv2.rectangle(imgCountour, (x, y), (x+w, y+h), (0, 0, 0), 4)
                                if corners > 3 and corners < 10:
                                    pri = cv2.arcLength(cnt, True)
                                    vertices = cv2.approxPolyDP(cnt, pri * 0.02, True)
                                    print(len(vertices))
                                    corners = len(vertices)
                                    x, y, w, h = cv2.boundingRect(vertices)
                                    coin_roi = result[y:y + h, x:x + w]
                                    custom_image_resized = cv2.resize(color_diamonds, (w, h))
                                    result[y:y + h, x:x + w] = cv2.addWeighted(coin_roi, 0, custom_image_resized, 1, 0)
                    if (text == '4'):
                        for cnt in contours:
                            # cv2.drawContours(imgCountour, cnt, -1, (0, 0, 255), 4)
                            print(cv2.contourArea(cnt))
                            print(cv2.arcLength(cnt, True))
                            area = cv2.contourArea(cnt)
                            if area > 50000 and area < 100000:

                                pri = cv2.arcLength(cnt, True)
                                vertices = cv2.approxPolyDP(cnt, pri * 0.02, True)
                                print(len(vertices))
                                corners = len(vertices)
                                #        x, y, w, h= cv2.boundingRect(vertices)
                                #        cv2.rectangle(imgCountour, (x, y), (x+w, y+h), (0, 0, 0), 4)
                                if corners > 3 and corners < 10:
                                    pri = cv2.arcLength(cnt, True)
                                    vertices = cv2.approxPolyDP(cnt, pri * 0.02, True)
                                    print(len(vertices))
                                    corners = len(vertices)
                                    x, y, w, h = cv2.boundingRect(vertices)
                                    coin_roi = result[y:y + h, x:x + w]
                                    custom_image_resized = cv2.resize(color_diamonds, (w, h))
                                    result[y:y + h, x:x + w] = cv2.addWeighted(coin_roi, 0, custom_image_resized, 1,0)

        #cv2.imshow('img', img)
        #cv2.imshow('hsv', hsv)
        #cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        cv2.imshow('imgCountour_hand_test', imgCountour_hand_test)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
