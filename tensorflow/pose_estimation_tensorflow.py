import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path='models/movenet/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite')
interpreter.allocate_tensors()
input_size = 192

def draw_keypoints(frame, keypoints_with_scores, confidence_threshold):
    """Draws the keypoint predictions on frames
    
    Args:
    -----
    image: numpy array
            A numpy array with shape [height, width, channel] 
            representing the pixel values of the input image.
    keypoints_with_scores: numpy array
            A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.
    keypoint_threshold: float
                minimum confidence score for a keypoint to be visualized.
    """
    height, width, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores, [height,width,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 



def draw_connections(frame, keypoints_with_scores, confidence_threshold):
    """Draws the keypoint connection on frames

    Args:
    -----
    image: numpy array
            A numpy array with shape [height, width, channel] 
            representing the pixel values of the input image.
    keypoints_with_scores: numpy array
            A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.
    keypoint_threshold: float
                minimum confidence score for a keypoint to be visualized.
    """
    height, width, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores, [height,width,1]))
    
    for edge, _ in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

cap = cv2.VideoCapture("https://player.vimeo.com/external/376858866.sd.mp4?s=d963a730f18ac7e9e93a0664dec43eab3ae41617&amp;profile_id=164&amp;oauth2_token_id=57447761")

while cap.isOpened():
    ret, frame = cap.read()
    
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), input_size,input_size)
    input_image = tf.cast(img, dtype=tf.uint8)
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Rendering 
    draw_connections(frame, keypoints_with_scores, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    
    cv2.imshow('MoveNet Lightning', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


    

