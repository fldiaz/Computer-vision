
import numpy as np

import cv2
import imutils
import os
import time
from scipy.spatial.distance import pdist, squareform



def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points)) #order src_points
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])#destinations points

    M = cv2.getPerspectiveTransform(src, dst)#puntos q forman el rect, y trasnf points
    M_inv = cv2.getPerspectiveTransform(dst, src)
    #warped = cv2.warpPerspective(img, M,IMAGE_W, IMAGE_H )[0][0] #topdown view
    return M, M_inv




def person_detect(frame):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 0.0039, (416, 416), (0,0,0), True)
    net.setInput(blob)
    outs=net.forward(output_layers)
    boxes=[]
    confidences=[]
    classes =[]
    total_pedestrians = 0
      #showing
    for out in outs:
        for detection in out:
            scores = detection [5:]
            class_id= np.argmax(scores)
            confidence=scores[class_id]
            if LABELS[class_id] == "person":
                if confidence > 0.7:#object detected, necesitamos las coordenadas
                    center_x=int(detection[0]* frame_w)
                    center_y= int (detection[1] * frame_h)
                    w= int(detection[2]* frame_w)
                    h= int(detection[3]* frame_h)
                    x = int(center_x - (w / 2))
                    y = int(center_y- (h / 2))
                    boxes.append([x,y,w,h]) #agrego todas las coordenadas de los q encuentra
                    confidences.append(float(confidence))
                    classes.append(class_id)


    indexes=cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3) #para q tome uno de los objetos y no repita

    if len(boxes) > 0:
        for i in range(len(boxes)):
            if i in indexes:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                frame_w_person=cv2.rectangle(frame, (x,y), (x+w, y+h), (160, 48, 112),2)
                total_pedestrians += 1

    return boxes, total_pedestrians, frame_w_person




def plot_pedestrian_boxes_on_image(frame, distance_mat, boxes):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    thickness = 2
    color_6 = (255,255,255)
    green = (0, 255, 0)

    for i in range(len(boxes)):

        x,y,w,h = boxes[i][:]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),green,2)

    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]

        if closeness == 1:
            x,y,w,h = per1[:]
            pedestrian_detect = cv2.rectangle(frame,(x,y),(x+w,y+h),red,2)

            x1,y1,w1,h1 = per2[:]
            pedestrian_detect = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),red,2)

            pedestrian_detect = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 2)

    return pedestrian_detect

def plot_points_on_bird_eye_view(frame, pedestrian_boxes, M, scale_w, scale_h):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)

    node_radius = 6
    color_node = (192, 133, 156)
    thickness_node = 20
    solid_back_color = (41, 41, 41)

    blank_image = np.zeros(
        (int(frame_h*scale_h ), int(frame_w*scale_w ), 3), np.uint8
        )
    blank_image[:] = white
    warped_pts = []

    for i in range(len(pedestrian_boxes)):
        mid_point_x = int(
            (pedestrian_boxes[i][0] + pedestrian_boxes[i][2]  ) / 2
        )
        mid_point_y = int(
            (pedestrian_boxes[i][1]  + pedestrian_boxes[i][3] ) / 2
        )


        pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
        warped_pt_scaled = [int(warped_pt[0] *scale_w), int(warped_pt[1]*scale_h  )]

        warped_pts.append(warped_pt_scaled)
        bird_image = cv2.circle(
            blank_image,
            (warped_pt_scaled[0], warped_pt_scaled[1]),
            node_radius,
            color_node,
            thickness_node)

    #cv2.imshow("Bird Eye", bird_image)
    #cv2.waitKey(1)

    return warped_pts, bird_image

def get_distances(boxes, warped_pts ,d_thresh ):

    distance_mat = []
    bxs = []

    for i in range(len(warped_pts)):
        for j in range(len(warped_pts)):
            if i != j:
                p = np.array(warped_pts)
                dist_condensed = pdist(p)#toma la distancia de los puntos warped
                dist = squareform(dist_condensed)
                dd = np.where(dist < d_thresh)
                no_dd = np.where(dist > d_thresh)
                for i in range(int(np.ceil(len(dd[0]) / 2))):
                    closeness = 0
                    if dd[0][i] != dd[1][i]:
                        distance_mat.append([warped_pts[i], warped_pts[j], closeness])
                        bxs.append([boxes[i], boxes[j], closeness])
                for i in range(int(np.ceil(len(no_dd[0]) / 2))):
                    closeness = 1
                    if no_dd[0][i] != no_dd[1][i]:
                        distance_mat.append([warped_pts[i], warped_pts[j], closeness])
                        bxs.append([boxes[i], boxes[j], closeness])

    return distance_mat, bxs


def plot_lines_between_nodes(warped_points, bird_image, d_thresh):
    p = np.array(warped_points)
    dist_condensed = pdist(p)#toma la distancia de los puntos warped
    dist = squareform(dist_condensed)


    # Really close: 1.8 metros feet mark
    no_dd = np.where(dist > d_thresh)
    dd = np.where(dist < d_thresh)
    six_feet_violations = len(np.where(dist_condensed < d_thresh)[0])
    total_pairs = len(dist_condensed)
    danger_p = []
    no_danger_p=[]
    color_6 = (0, 0, 255)
    green = (0, 255, 0)
    lineThickness=4
    for i in range(int(np.ceil(len(dd[0]) / 2))):
        if dd[0][i] != dd[1][i]:
            point1 = dd[0][i]
            point2 = dd[1][i]

            danger_p.append([point1, point2])
            cv2.line(
                    bird_image,
                    (p[point1][0], p[point1][1]),
                    (p[point2][0], p[point2][1]),
                    color_6,
                    lineThickness,
                )


    # Display Birdeye view
    cv2.imshow("Bird Eye", bird_image)
    cv2.waitKey(1)

    return six_feet_violations, total_pairs


#TYOLO MODEL
yolo = "yolo-coco/"
#weights = os.path.sep.join([yolo, "yolov3.weights"])
config = os.path.sep.join([yolo, "yolov3.cfg"])
labelsPath = os.path.sep.join([yolo, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(config, weights) #ln
layer_names = net.getLayerNames() #ln
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] #detecta los objetos
