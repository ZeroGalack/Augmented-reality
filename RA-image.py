import cv2
import numpy as np
import os


def image_augmentation(frame, src_image, dst_points):
    src_h, src_w = src_image.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    
    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_w]])
    H, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)
    warp_image = cv2.warpPerspective(src_image, H, (frame_w, frame_h))
    cv2.imshow("warp image", warp_image)
    cv2.fillConvexPoly(mask, dst_points, 255)
    cv2.bitwise_and(warp_image, warp_image, frame, mask=mask)


marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

param_markers = cv2.aruco.DetectorParameters()


cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    marker_corners, marker_IDs, reject = cv2.aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)
    
    if len(marker_corners) > 0:
        ids = marker_IDs.flatten()
        
        for marker_IDs, corners in zip(ids, marker_corners):

            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            
            top_left, top_right, bottom_right, bottom_left = corners
            
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))

            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            

            cv2.putText(frame, str(marker_IDs),
                        (top_left[0], top_left[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            
            if int(marker_IDs) == 2:
                overlay_img = cv2.imread(r'media\aguaviva.png')
                image_augmentation(frame, overlay_img, corners)
                
            if int(marker_IDs) == 4:
                overlay_img = cv2.imread(r'media\aguaviva2.png')
                image_augmentation(frame, overlay_img, corners)
            
    frame = cv2.flip(frame, 1)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()