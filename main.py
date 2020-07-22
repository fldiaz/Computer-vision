import numpy as np
import numpy as np
import cv2
import imutils
import os
import time
from scipy.spatial.distance import pdist, squareform
from aux_func import *

#helo
###Recordar agregar los archivos de YOLO


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)

filename = 'videos/video_restricciones.mp4'
fourcc = cv2.VideoWriter_fourcc(*"XVID")


mouse_pts = []
font=cv2.FONT_HERSHEY_SIMPLEX

create = None
cap = cv2.VideoCapture(filename)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))

starting_time = time.time()


SOLID_BACK_COLOR = (41, 41, 41)


dis_h = height
dis_w = width


scale_h = dis_h/height
scale_w=dis_w/width


# Initialize necessary variables
frame_id=0
total_pedestrians_detected = 0
total_six_feet_violations = 0
total_pairs = 0
abs_six_feet_violations = 0
pedestrian_per_sec = 0
sh_index = 1


cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
num_mouse_points = 0
first_frame_display = True



while cap.isOpened():
    frame_id += 1 #extract frames
    ret, frame = cap.read()
    #frame = imutils.resize(frame, width=480)
    if not ret:
        print("end of the video file...")
        break
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    if frame_id == 1:
        # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
        while True:
            image = frame
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if len(mouse_pts) == 7:
                cv2.destroyWindow("image")
                break
            first_frame_display = False
        four_points = mouse_pts

        # Get perspective
        M, Minv = get_camera_perspective(frame, four_points[0:4])
        pts = src = np.float32(np.array([four_points[4:]]))
        warped_pt = cv2.perspectiveTransform(pts, M)[0]
        d_thresh = np.sqrt(
            (warped_pt[0][0] - warped_pt[1][0]) ** 2
            + (warped_pt[0][1] - warped_pt[1][1]) ** 2)

        bird_image = np.zeros(
            (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
        )

        bird_image[:] = SOLID_BACK_COLOR
        pedestrian_detect = frame

    print("Processing frame: ", frame_id)

    # draw polygon of ROI
    pts = np.array(
        [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
        )
    cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=2)

    # Detect person and bounding boxes using DNN
    boxes, num_pedestrians, pedestrian_detect=person_detect(frame)

    if len(boxes) > 0:

        warped_pts, bird_image  = plot_points_on_bird_eye_view(frame, boxes, M, scale_w, scale_h)#plot_points_on_bird_eye_view
        six_feet_violations, pairs = plot_lines_between_nodes ( warped_pts, bird_image, d_thresh)

        total_six_feet_violations += six_feet_violations / fps

        total_pedestrians_detected += num_pedestrians
        total_pairs += pairs
        abs_six_feet_violations += six_feet_violations




    print(total_pedestrians_detected,total_pairs,total_six_feet_violations)
    last_h = 75
    print('# Distancia menor a 2 metros:  .{}'.format(str(int(total_six_feet_violations))))

    if total_pairs != 0:
        sc_index = abs_six_feet_violations / total_pairs

    text = "No respetan la distancia: " + str(np.round(100 * sc_index, 1)) + "%"
    cv2.putText(pedestrian_detect, text , (10, 50), font, 0.7, (255, 255, 255), 2)


    if create is None:
        create = cv2.VideoWriter("./videos/"+ filename.split('/')[1][:-4]+ ".avi", fourcc, 30, (frame.shape[1], frame.shape[0]), True)
        bird_eye = cv2.VideoWriter("./videos/output_of_bird"+ filename.split('/')[1][:-4]+ ".avi", fourcc, fps, (int(width * scale_w), int(height * scale_h))
        )
    create.write(frame)
    bird_eye.write(bird_image)
    cv2.imshow('output', pedestrian_detect)
    if cv2.waitKey(27) & 0xFF == ord('s'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
