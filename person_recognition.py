# Import required modules and packages
import redis
import cv2
import numpy as np
import rekognition


# Main function
def main():
    # Initialize redis stream
    redis_client = redis.Redis(host='127.0.0.1')

    # Initialize Rekognition client
    rekognition_client = rekognition.Rekognition()

    # Indicate process start
    print("Process Started!\n")

    # Keep checking for person recognition requests
    while True:
        message = redis_client.xread({'Recognition_Request': '$'}, None, 0)
        print("Recognition Request Received!")
        person_image = cv2.imdecode(np.frombuffer(message[0][1][0][1][b'Person_Image'], np.uint8), cv2.IMREAD_COLOR)
        person_tracking_id = message[0][1][0][1][b'Tracking_ID'].decode("utf-8")

        cv2.imwrite("person_image.jpg", person_image)

        try:
            rekognition_response = rekognition_client.search_face(collection_id="FacesCollection",
                                                                  source_image="person_image.jpg")
            print("AWS Rekognition Request Sent!")
            if len(rekognition_response['FaceMatches']) != 0:
                print("Face Matched!")
                person_name = rekognition_response['FaceMatches'][0]['Face']['ExternalImageId']
            else:
                print("Face Not Matched!")
                person_name = "Unknown"
        except rekognition_client.client.exceptions.InvalidParameterException:
            print("No Face Detected!")
            person_name = "Unknown"

        redis_client.xadd(name="Recognition_Response",
                          fields={
                              "Person_Name": person_name,
                              "Tracking_ID": person_tracking_id
                          },
                          maxlen=10,
                          approximate=False)
        redis_client.execute_command(f'XTRIM Recognition_Response MAXLEN 10')
        print("Recognition Response Sent!\n")


if __name__ == '__main__':
    main()