# import required modules and packages
import cv2
import sys
import torch
import time
import redis
import threading
import numpy as np
from motpy import Detection, MultiObjectTracker

# RTSP camera streams
RTSP_STREAM_1 = "rtsp://servizio:K2KAccesso2021!@188.10.33.54:7554/cam/realmonitor?channel=1&subtype=0"
RTSP_STREAM_2 = "rtsp://servizio:K2KAccesso2021!@188.10.33.54:9554/cam/realmonitor?channel=1&subtype=0"
RTSP_STREAM_3 = "rtsp://servizio:K2KAccesso2021!@188.10.33.54:8554/cam/realmonitor?channel=1&subtype=0"

# Dictionary to store tracking id, person name, bounding box and reconition attempts
TRACKING_DICT_GLOBAL = {}

# Variable to keep track of delay between person recognition requests
recognition_time = 0

# Variable to keep track of current camera stream and change according to input
stream_change = 1


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


# Function to get tracks for person object using tracker
def get_tracks(results, frame, tracker):
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
        if track.id not in TRACKING_DICT_GLOBAL.keys():
            TRACKING_DICT_GLOBAL[track.id] = ["Person", track.box, 0]
            print("New tracking ID created!")
        else:
            TRACKING_DICT_GLOBAL[track.id][1] = track.box.tolist()

    for tracking_id in list(TRACKING_DICT_GLOBAL):
        if tracking_id not in tracking_id_list:
            del TRACKING_DICT_GLOBAL[tracking_id]


# Function to draw tracking bounding boxes for persons
def plot_boxes_tracks(frame):
    for tracking_id in TRACKING_DICT_GLOBAL:
        cv2.rectangle(frame,
                      (int(TRACKING_DICT_GLOBAL[tracking_id][1][0]), int(TRACKING_DICT_GLOBAL[tracking_id][1][1])),
                      (int(TRACKING_DICT_GLOBAL[tracking_id][1][2]), int(TRACKING_DICT_GLOBAL[tracking_id][1][3])),
                      (255, 0, 0), 2)
        cv2.putText(frame, "Person",
                    (int(TRACKING_DICT_GLOBAL[tracking_id][1][0]), int(TRACKING_DICT_GLOBAL[tracking_id][1][1])-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, tracking_id,
                    (int(TRACKING_DICT_GLOBAL[tracking_id][1][0]), int(TRACKING_DICT_GLOBAL[tracking_id][1][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame


# Function to send cropped person image to process via Redis stream for recognition
def recognize_person(frame_fullsize, frame_resized, redis_client):
    global recognition_time
    for person in TRACKING_DICT_GLOBAL:
        if TRACKING_DICT_GLOBAL[person][2] < 5 and (time.time() - recognition_time) > 10:
            frame_resized = frame_resized[279:1000, :]
            x1, y1, x2, y2 = int(TRACKING_DICT_GLOBAL[person][1][0]), int(TRACKING_DICT_GLOBAL[person][1][1]) - 279, \
                             int(TRACKING_DICT_GLOBAL[person][1][2]), int(TRACKING_DICT_GLOBAL[person][1][3]) - 279
            new_x1, new_y1, new_x2, new_y2 = int(x1 * (frame_fullsize.shape[1]/frame_resized.shape[1])), \
                                             int(y1 * (frame_fullsize.shape[0]/frame_resized.shape[0])), \
                                             int(x2 * (frame_fullsize.shape[1]/frame_resized.shape[1])), \
                                             int(y2 * (frame_fullsize.shape[0]/frame_resized.shape[0]))
            person_image = frame_fullsize[new_y1: new_y2, new_x1:new_x2]

            redis_client.xadd(name="Recognition_Request",
                              fields={
                                      "Person_Image": cv2.imencode('.jpg', person_image)[1].tobytes(),
                                      "Tracking_ID": person
                                     },
                              maxlen=10,
                              approximate=False)
            redis_client.execute_command(f'XTRIM Recognition_Request MAXLEN 10')
            TRACKING_DICT_GLOBAL[person][2] += 1
            recognition_time = time.time()
            print("Recognition Request Sent!")


def main_process(rtsp_stream, rtsp_stream_num, model, redis_client, tracker):
    # Initialize camera stream
    capture = cv2.VideoCapture(rtsp_stream)

    while True:
        try:
            start_time_fps = time.time()
            if capture.isOpened():
                (status, frame_fullsize) = capture.read()
                if status:
                    frame_resized = image_resize_aspect(frame_fullsize, 1280)
                    results = score_frame(frame=frame_resized, model=model)
                    get_tracks(results=results, frame=frame_resized, tracker=tracker)
                    recognize_person(frame_fullsize=frame_fullsize,
                                     frame_resized=frame_resized,
                                     redis_client=redis_client)
                    # frame_resized = plot_boxes(results=results, frame=frame_resized, model=model)
                    frame_resized = plot_boxes_custom(frame=frame_resized, rtsp_stream_num=rtsp_stream_num)
                    frame_resized = plot_boxes_tracks(frame=frame_resized)
                    frame_show = frame_resized[250:1030, 0:1280]
                    redis_client.xadd(name="Frame",
                                      fields={
                                          "Final_Frame": cv2.imencode('.jpg', frame_show)[1].tobytes()
                                      },
                                      maxlen=10,
                                      approximate=False)
                    redis_client.execute_command(f'XTRIM Frame MAXLEN 10')
                    print(TRACKING_DICT_GLOBAL)
                    # cv2.imshow("Camera Stream", frame_show)
                    # cv2.waitKey(1)

                    if stream_change == rtsp_stream_num:
                        pass
                    else:
                        if stream_change == 1:
                            rtsp_stream = RTSP_STREAM_1
                            capture = cv2.VideoCapture(rtsp_stream)
                            print("Camera Stream Changed!")
                            rtsp_stream_num = stream_change
                        elif stream_change == 2:
                            rtsp_stream = RTSP_STREAM_2
                            capture = cv2.VideoCapture(rtsp_stream)
                            print("Camera Stream Changed!")
                            rtsp_stream_num = stream_change
                        elif stream_change == 3:
                            rtsp_stream = RTSP_STREAM_3
                            capture = cv2.VideoCapture(rtsp_stream)
                            print("Camera Stream Changed!")
                            rtsp_stream_num = stream_change

            else:
                print("Camera Stream Issue.")
            end_time_fps = time.time()
            print("Frames-per-second (FPS):", 1 / (end_time_fps - start_time_fps))
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Error:", error)
            print("Error Line:", exc_tb.tb_lineno)


# Function to read redis stream for camera stream change
def thread_function(redis_client):
    global stream_change
    while True:
        message = redis_client.xread({'Stream_Change': '$'}, None, 0)
        stream_change = int(message[0][1][0][1][b'Stream_Num'].decode("utf-8"))


def main():
    # Load YOLO model
    print("Loading model.")
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='best.pt',
                           force_reload=True,
                           device="cuda:0"
                           )
    model.cuda()
    print("Model loaded.")

    # Initialize redis client
    redis_client = redis.Redis(host='127.0.0.1')

    # Initialize MotPy tracker
    tracker = MultiObjectTracker(dt=3.0)

    # Thread to monitor stream change
    stream_change_thread = threading.Thread(target=thread_function, args=(redis_client, ))
    stream_change_thread.start()

    # Run the main process function
    main_process(rtsp_stream=RTSP_STREAM_1, rtsp_stream_num=1,
                 model=model, redis_client=redis_client, tracker=tracker)


if __name__ == '__main__':
    main()
