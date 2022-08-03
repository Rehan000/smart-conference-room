# Import required modules and packages
import sys
import time
import cv2
import redis
import threading
import mediapipe as mp
import numpy as np

# RTSP camera streams list
RTSP_STREAM_LIST = ["rtsp://servizio:K2KAccesso2021!@188.10.33.54:7554/cam/realmonitor?channel=1&subtype=0",
                    "rtsp://servizio:K2KAccesso2021!@188.10.33.54:9554/cam/realmonitor?channel=1&subtype=0",
                    "rtsp://servizio:K2KAccesso2021!@188.10.33.54:8554/cam/realmonitor?channel=1&subtype=0"]

# Variable to keep track of current camera stream and change according to input
stream_change = 1

# Variable to keep track of current process running (Detection/Tracking or Pose Estimation)
process_change = "Detection/Tracking"


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


# Function to read redis stream for camera stream change
def thread_function_stream_change(redis_client):
    global stream_change
    while True:
        message = redis_client.xread({'Stream_Change': '$'}, None, 0)
        stream_change = int(message[0][1][0][1][b'Stream_Num'].decode("utf-8"))


# Function to read redis stream for process change
def thread_function_process_change(redis_client):
    global process_change
    while True:
        message = redis_client.xread({'Process_Change': '$'}, None, 0)
        process_change = message[0][1][0][1][b'Process'].decode("utf-8")


def main_process(rtsp_stream, rtsp_stream_num, redis_client,
                 mpPose, mpDraw, pose, drawing_specs_points,
                 drawing_specs_line):
    # Initialize camera stream
    capture = cv2.VideoCapture(rtsp_stream)

    frame_show = None

    while True:
        try:
            if capture.isOpened():
                (status, frame_fullsize) = capture.read()
                if status:
                    if process_change == "Pose_Estimation":
                        start_time_fps = time.time()
                        frame_fullsize_RGB = cv2.cvtColor(frame_fullsize, cv2.COLOR_BGR2RGB)
                        pose_results = pose.process(frame_fullsize_RGB)
                        if pose_results.pose_landmarks:
                            mpDraw.draw_landmarks(frame_fullsize_RGB, pose_results.pose_landmarks,
                                                  mpPose.POSE_CONNECTIONS,
                                                  drawing_specs_points,
                                                  drawing_specs_line)
                        frame_fullsize_BGR = cv2.cvtColor(frame_fullsize_RGB, cv2.COLOR_RGB2BGR)
                        frame_resized = image_resize_aspect(frame_fullsize_BGR, 1280)
                        frame_show = frame_resized[250:1030, 0:1280]
                        redis_client.xadd(name="Frame",
                            fields={
                                    "Final_Frame": cv2.imencode('.jpg', frame_show)[1].tobytes()
                            },
                            maxlen=10,
                            approximate=False)
                        redis_client.execute_command(f'XTRIM Frame MAXLEN 10')
                        end_time_fps = time.time()
                        print("Frames-per-second (FPS):", 1 / (end_time_fps - start_time_fps))
                        # cv2.imshow("Camera Stream", frame_show)
                        # cv2.waitKey(1)

                    if stream_change == rtsp_stream_num:
                        pass
                    elif stream_change > len(RTSP_STREAM_LIST) or stream_change < 1:
                        pass
                    else:
                        rtsp_stream = RTSP_STREAM_LIST[stream_change - 1]
                        capture = cv2.VideoCapture(rtsp_stream)
                        print("Camera Stream Changed!")
                        rtsp_stream_num = stream_change
            else:
                print("Camera Stream Issue.")
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Error:", error)
            print("Error Line:", exc_tb.tb_lineno)

# Main function
def main():
    print("Loading pose estimation model.")
    # Initialize mediapipe pose object
    mpPose = mp.solutions.pose
    mpDraw = mp.solutions.drawing_utils
    pose = mpPose.Pose()
    drawing_specs_points = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), circle_radius=7, thickness=-1)
    drawing_specs_line = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=5)

    # Initialize redis client
    redis_client = redis.Redis(host='127.0.0.1')

    # Thread to monitor stream change
    stream_change_thread = threading.Thread(target=thread_function_stream_change, args=(redis_client,))
    stream_change_thread.start()

    # Thread to monitor process change
    process_change_thread = threading.Thread(target=thread_function_process_change, args=(redis_client,))
    process_change_thread.start()

    # Run the main process function
    main_process(rtsp_stream=RTSP_STREAM_LIST[0], rtsp_stream_num=1,
                 redis_client=redis_client,
                 mpPose=mpPose, mpDraw=mpDraw, pose=pose,
                 drawing_specs_points=drawing_specs_points,
                 drawing_specs_line=drawing_specs_line)


if __name__ == '__main__':
    main()