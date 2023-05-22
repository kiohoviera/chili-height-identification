import cv2

cam = cv2.VideoCapture(0)

result, image = cam.read()

if result:

    imshow("SoilMonitoringCapture", image)

    imwrite("input_image.jpg", image)
else:
    print("Unable to capture image, please check your camera")
