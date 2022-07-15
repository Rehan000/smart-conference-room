# import required modules and packages
import cv2
import sys
import torch
import time
import numpy as np
from motpy import Detection, MultiObjectTracker

# RTSP camera stream client
RTSP_STREAM_1 = "rtsp://servizio:K2KAccesso2021!@188.10.33.54:7554/cam/realmonitor?channel=1&subtype=0"
RTSP_STREAM_2 = "rtsp://servizio:K2KAccesso2021!@188.10.33.54:9554/cam/realmonitor?channel=1&subtype=0"
RTSP_STREAM_3 = "rtsp://servizio:K2KAccesso2021!@188.10.33.54:8554/cam/realmonitor?channel=1&subtype=0"

TRACKING_ID_LIST_GLOBAL = []


# Function to resize image while preserving aspect ratio
def image_resize_aspect(img, newWidth):
    old_size = img.shape
    desired_size = newWidth
    if old_size[0] < desired_size and old_size[1] < desired_size:
        new_im = np.zeros([desired_size, desired_size, 3])
        y = (desired_size - old_size[0]) // 2
        x = (desired_size - old_size[1]) // 2
        new_im[y:y + old_size[0], x:x + old_size[1]] = img
    else:
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        new_size = (new_size[1], new_size[0])
        img = cv2.resize(img, new_size)
        new_im = np.zeros([desired_size, desired_size, 3])
        y = (desired_size - new_size[1]) // 2
        x = (desired_size - new_size[0]) // 2
        new_im[y:y + new_size[1], x:x + new_size[0]] = img
    return np.uint8(new_im)


# Function to make prediction on a given frame
def score_frame(frame, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels = results.xyxyn[0][:, -1]
    cord = results.xyxyn[0][:, :-1]
    return labels, cord


# Function to draw bounding boxes on a given frame
def plot_boxes(results, frame, model):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.20:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            bgr = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, model.names[int(labels[i])].capitalize(), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


# Function to draw custom bounding boxes on a given frame
def plot_boxes_custom(frame, rtsp_stream_num):
    if rtsp_stream_num == 1:
        # Meeting table
        cv2.rectangle(frame, (450, 480), (850, 900), (0, 255, 0), 2)
        cv2.putText(frame, "Meeting Table", (450, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Video-walls
        cv2.rectangle(frame, (480, 280), (800, 380), (255, 0, 0), 2)
        cv2.putText(frame, "Video Wall", (480, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Windows
        cv2.rectangle(frame, (1100, 280), (1250, 650), (3, 198, 252), 2)
        cv2.putText(frame, "Window", (1100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (3, 198, 252), 2)

    elif rtsp_stream_num == 2:
        # Meeting table
        cv2.rectangle(frame, (290, 790), (1020, 1000), (0, 255, 0), 2)
        cv2.putText(frame, "Meeting Table", (290, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Video-walls
        cv2.rectangle(frame, (350, 400), (1000, 650), (255, 0, 0), 2)
        cv2.putText(frame, "Video Wall", (350, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    elif rtsp_stream_num == 3:
        # Meeting table
        cv2.rectangle(frame, (290, 750), (1020, 1000), (0, 255, 0), 2)
        cv2.putText(frame, "Meeting Table", (290, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Windows
        cv2.rectangle(frame, (200, 300), (350, 550), (3, 198, 252), 2)
        cv2.putText(frame, "Window", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (3, 198, 252), 2)
        cv2.rectangle(frame, (550, 280), (700, 550), (3, 198, 252), 2)
        cv2.putText(frame, "Window", (550, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (3, 198, 252), 2)
        cv2.rectangle(frame, (900, 330), (1050, 580), (3, 198, 252), 2)
        cv2.putText(frame, "Window", (900, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (3, 198, 252), 2)

    return frame


# Function to get tracking ids and draw tracking bounding boxes for persons
def plot_boxes_tracks(results, frame, tracker):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    detections_list = []
    tracking_boxes_list = []
    tracking_id_list = []
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.20:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            detections_list.append(Detection(box=np.array([x1, y1, x2, y2])))
    tracker.step(detections=detections_list)
    tracks = tracker.active_tracks()
    for track in tracks:
        tracking_boxes_list.append(track.box)
        tracking_id_list.append(track.id)
        if track.id not in TRACKING_ID_LIST_GLOBAL:
            TRACKING_ID_LIST_GLOBAL.append(track.id)
            print("New tracking ID created!")

    for index in range(len(tracking_boxes_list)):
        cv2.rectangle(frame,
                      (int(tracking_boxes_list[index][0]), int(tracking_boxes_list[index][1])),
                      (int(tracking_boxes_list[index][2]), int(tracking_boxes_list[index][3])),
                      (255, 0, 0), 2)
        cv2.putText(frame, tracking_id_list[index],
                    (int(tracking_boxes_list[index][0]), int(tracking_boxes_list[index][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame


def main():
    # Load YOLO model
    print("Loading model.")
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='best.pt',
                           force_reload=True,
                           device="cuda:0"
                           )
    print("Model loaded.")

    # Initialize camera stream
    capture = cv2.VideoCapture(RTSP_STREAM_1)

    # Initialize MotPy tracker
    tracker = MultiObjectTracker(dt=0.5)

    while True:
        try:
            start_time = time.time()
            if capture.isOpened():
                (status, frame) = capture.read()
                if status:
                    frame = image_resize_aspect(frame, 1280)
                    results = score_frame(frame=frame, model=model)
                    # frame = plot_boxes(results=results, frame=frame, model=model)
                    frame = plot_boxes_custom(frame, rtsp_stream_num=1)
                    frame = plot_boxes_tracks(results=results, frame=frame, tracker=tracker)
                    frame = frame[250:1030, 0:1280]
                    # cv2.imshow("Camera Stream", frame)
                    # cv2.waitKey(1)
            else:
                print("Camera Stream Issue.")
            end_time = time.time()
            # print("Frames-per-second (FPS):", 1 / (end_time - start_time))
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Error:", error)
            print("Error Line:", exc_tb.tb_lineno)


if __name__ == '__main__':
    main()
