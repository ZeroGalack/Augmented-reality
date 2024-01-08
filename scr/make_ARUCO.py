import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
img_size = 700 # Define the size of the final image

for i in range(5):
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, img_size)
    cv2.imwrite(f"images/code_{i}.png", marker_img)
    