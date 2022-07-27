import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0) # can change 0 to other media sources like a video or a streaming url as necessary
mpHands = mp.solutions.hands # choosing hands as it's the solution we want to use
hands = mpHands.Hands() # getting hands object
mpDraw = mp.solutions.drawing_utils # for visualisation

# points to be checked to see if one is below the other
# eg: index finger, 8 should be below the 6 as shown in hand_landmarks.png and so on
fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinate = (4,2)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # required for mediapipe
    
    results = hands.process(imgRGB) # processing with mediapipe, locates the points as shown in hand_landmarks.png
    multiLandMarks = results.multi_hand_landmarks # extracting the multi hand landmarks

    if multiLandMarks: # checking if there is a hand detected
        handPoints = []
        for handLms in multiLandMarks: # choosing one hand at a time
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # drawing the hand landmarks

            # idx = points as shown in hand_landmarks.png
            # lm = landmark coordinates
            for idx, lm in enumerate(handLms.landmark): # iterating over landmarks
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) # converting to pixel coordinates
                handPoints.append((cx, cy)) # appending to list of hand points

        # drawing the fingertips
        for point in handPoints:
            cv2.circle(img, point, 10, (255, 0, 0), cv2.FILLED)

        upCount = 0
        for coordinate in fingerCoordinates:
            # checking if y-coordinates of the points is below the other point as described above
            if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]: 
                upCount += 1
        
        # checking for thumb coordinates, the x-coordinate values
        if handPoints[thumbCoordinate[0]][0] > handPoints[thumbCoordinate[1]][0]:
            upCount += 1

        cv2.putText(img, str(upCount), (80,170), cv2.FONT_HERSHEY_PLAIN, 12, (0,0,255), 12)
        print(upCount)

    # Can comment out the following line to disable displaying the image
    cv2.imshow("Finger Counter", img)   
    cv2.waitKey(1)