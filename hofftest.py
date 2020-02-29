import numpy as np
import cv2
from modules import blur

hsvarray = None
upper = np.array([0, 0, 0])
lower = np.array([255, 255, 255])

variance = 10

# cappin
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -3.0)


def print_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsvarray[y, x]
        # return upper, lower
        print([pixel[0], pixel[1], pixel[2]])

def add_and_subtract(event, x, y, flags, param):
    global upper
    global lower
    pixel = hsvarray[y, x]
    if event == cv2.EVENT_LBUTTONDOWN: #add this pixel to the range
        if upper[0] < pixel[0]: upper[0] = pixel[0]
        if upper[1] < pixel[1]: upper[1] = pixel[1]
        if upper[2] < pixel[2]: upper[2] = pixel[2]

        if lower[0] > pixel[0]: lower[0] = pixel[0]
        if lower[1] > pixel[1]: lower[1] = pixel[1]
        if lower[2] > pixel[2]: lower[2] = pixel[2]


    if event == cv2.EVENT_RBUTTONDOWN:
        upper = np.array([pixel[0], pixel[1], pixel[2]])
        lower = np.array([pixel[0], pixel[1], pixel[2]])


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.setMouseCallback("Step 2: inRange", print_color)
    
    cv2.setMouseCallback("Step 1: Blur", add_and_subtract)
    hsvarray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blur to filter noise
    stepOne = blur.getSquareBlur(hsvarray, 5)

    # Filter pixels out of HSVrange
    stepTwo = cv2.inRange(stepOne, lower, upper)

    # Mask
    stepThree = cv2.bitwise_and(frame,frame, mask= stepTwo)

    # detect circles
    circles = cv2.HoughCircles(
        cv2.cvtColor(stepThree, cv2.COLOR_BGR2GRAY),
        cv2.HOUGH_GRADIENT,
        1,
        200,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=100,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    #     # Display the resulting frame
    cv2.imshow("Step 1: Blur", stepOne)
    cv2.imshow("Step 2: inRange", stepTwo)
    cv2.imshow("detected circles", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
# no cap
cap.release()
cv2.destroyAllWindows()

print("upper", upper)
print("lower", lower)