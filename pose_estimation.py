# Import required modules and packages
import cv2
import sys
import time
import redis
import mediapipe as mp
import numpy as np


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


# Main function
def main():
    # Initialize redis client
    redis_client = redis.Redis(host='127.0.0.1')

    # Initialize mediapipe pose object
    mpPose = mp.solutions.pose
    mpDraw = mp.solutions.drawing_utils
    pose = mpPose.Pose()
    drawing_specs_points = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), circle_radius=7, thickness=-1)
    drawing_specs_line = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=5)

    # Camera stream
    capture = cv2.VideoCapture("rtsp://servizio:K2KAccesso2021!@188.10.33.54:7554/cam/realmonitor?channel=1&subtype=0")

    while True:
        try:
            start_time_fps = time.time()
            if capture.isOpened():
                (status, frame_fullsize) = capture.read()
                if status:
                    frame_fullsize_RGB = cv2.cvtColor(frame_fullsize, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(frame_fullsize_RGB)
                    if pose_results.pose_landmarks:
                        mpDraw.draw_landmarks(frame_fullsize_RGB, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                              drawing_specs_points, drawing_specs_line)
                    frame_fullsize_BGR = cv2.cvtColor(frame_fullsize_RGB, cv2.COLOR_RGB2BGR)
                    frame_resized = image_resize_aspect(frame_fullsize_BGR, 1280)
                    frame_show = frame_resized[250:1030, 0:1280]
                    redis_client.xadd(name="Pose_Frame",
                                      fields={
                                          "Final_Frame": cv2.imencode('.jpg', frame_show)[1].tobytes()
                                      },
                                      maxlen=10,
                                      approximate=False)
                    redis_client.execute_command(f'XTRIM Pose_Frame MAXLEN 10')
                    # cv2.imshow("Camera Stream", frame_show)
                    # cv2.waitKey(1)

            else:
                print("Camera Stream Issue!")
            end_time_fps = time.time()
            print("Frames-per-second (FPS):", 1 / (end_time_fps - start_time_fps))
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Error:", error)
            print("Error Line:", exc_tb.tb_lineno)


if __name__ == '__main__':
    main()