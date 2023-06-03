import time
import cv2 as cv
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv.VideoCapture("https://player.vimeo.com/external/376858866.sd.mp4?s=d963a730f18ac7e9e93a0664dec43eab3ae41617&amp;profile_id=164&amp;oauth2_token_id=57447761")

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:

    while cap.isOpened():
        # Capture frame-by-frame
        success, image = cap.read()
        # if frame is read correctly ret is True
        if not success:
            print("Frame not found!!")
            break
        
        # To improve performance, optionally mark the image as immutable to
        #   pass by reference.
        image.flags.writeable = False
        # Starting time for the process
        t1 = time.time()
        # Send this image to the model
        results = pose.process(image)
        # Ending time for the process
        t2 = time.time()
        # Number of frames that appears within a second
        fps = 1/(t2 - t1)
        # Draw the pose annotations on the image.
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
        )

        dim = (image.shape[1], image.shape[0])
        resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        # display the FPS
        cv.putText(resized, 'FPS : {:.2f}'.format(fps), (int((image.shape[1] * 75) /100), 40), cv.FONT_HERSHEY_SIMPLEX, 1, (188, 205, 54), 2, cv.LINE_AA)
        # Display the resulting frame
        cv.imshow("Media pipe Pose with yolov7", resized)
        if cv.waitKey(10) & 0XFF == ord('q'):
            break
# Release everything if job is finished
cap.release()
cv.destroyAllWindows()