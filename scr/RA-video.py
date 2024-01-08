import cv2
import numpy as np


def video_augmentation(frame, src_video, dst_points):
    src_h, src_w = src_video.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    
    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_w]])
    H, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)
    warp_video = cv2.warpPerspective(src_video, H, (frame_w, frame_h))
    cv2.imshow("warp video", warp_video)
    cv2.fillConvexPoly(mask, dst_points, 255)
    cv2.bitwise_and(warp_video, warp_video, frame, mask=mask)
    

marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
param_markers = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(1)

# Caminho do vídeo
video_cap = cv2.VideoCapture(r'media\aguavivas.mp4')

while True:
    ret, frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    marker_corners, marker_IDs, reject = cv2.aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)
    
    if len(marker_corners) > 0:
        ids = marker_IDs.flatten()
        
        for marker_ID, corners in zip(ids, marker_corners):
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            
            if int(marker_ID) == 2 or int(marker_ID) == 4:
                ret_video, frame_video = video_cap.read()
                if not ret_video:
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                video_augmentation(frame, frame_video, corners)

    frame = cv2.flip(frame, 1)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    
cap.release()
video_cap.release()
cv2.destroyAllWindows()
