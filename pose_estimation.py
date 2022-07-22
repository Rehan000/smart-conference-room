# Import required modules and packages
import cv2
import sys
import time
import mediapipe as mp


# Main function
def main():
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    capture = cv2.VideoCapture("rtsp://servizio:K2KAccesso2021!@188.10.33.54:7554/cam/realmonitor?channel=1&subtype=0")
    while True:
        try:
            start_time_fps = time.time()
            if capture.isOpened():
                (status, frame_fullsize) = capture.read()
                if status:
                    frame_fullsize_RGB = cv2.cvtColor(frame_fullsize, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(frame_fullsize_RGB)
                    cv2.imshow("Camera Stream", frame_fullsize_RGB)
                    cv2.waitKey(1)
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