import cv2

def getSquareBlur(frame, radius):
    size = int(2 * round(radius) + 1)
    return cv2.blur(frame, (size, size))

def getGaussianBlur(frame, radius):
    size = int(6 * round(radius) + 1)
    return cv2.GaussianBlur(src, (size, size), round(radius))    