import cv2

img1 = cv2.imread('calibresult.png', 0)
cv2.namedWindow("test")


def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        pt1_x, pt1_y = x, y
        print(x, y)


cv2.setMouseCallback('test', line_drawing)

while(1):
    cv2.imshow('test', img1)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
