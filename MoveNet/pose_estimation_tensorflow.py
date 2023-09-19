import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import time
import argparse



class PoseEstimation:

    def __init__(self, args )-> None:
        self.args = args

        
    def get_model(self):
        # Initialize the TFLite interpreter
        interpreter = None
        model = None
        if self.args.model == "movenet_lightning":
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            input_size = 192
        elif self.args.model == "movenet_thunder":
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            input_size = 256
        elif self.args.model == "movenet_lightning_f16":
            # Initialize the TFLite interpreter
            interpreter = tf.lite.Interpreter(model_path='models/movenet/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite')
            interpreter.allocate_tensors()
            input_size = 192
        elif self.args.model == "movenet_thunder_f16":
            # Initialize the TFLite interpreter
            interpreter = tf.lite.Interpreter(model_path='models/movenet/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite')
            interpreter.allocate_tensors()
            input_size = 192
        elif self.args.model == "movenet_thunder_f16":
            # Initialize the TFLite interpreter
            interpreter = tf.lite.Interpreter(model_path='models/movenet/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite')
            interpreter.allocate_tensors()
            input_size = 192
        elif self.args.model == "movenet_lightning_int8":
            # Initialize the TFLite interpreter
            interpreter = tf.lite.Interpreter(model_path='models/movenet/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite')
            interpreter.allocate_tensors()
            input_size = 192
        elif self.args.model == "movenet_thunder_int8":
            # Initialize the TFLite interpreter
            interpreter = tf.lite.Interpreter(model_path='models/movenet/lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite')
            interpreter.allocate_tensors()
            input_size = 192
        else:
            print(f"Sorry, we don't currently supporte the model {self.args.model}")

        model = module.signatures['serving_default']
        
        return model, input_size, interpreter

    

    def draw_keypoints(self, frame, keypoints_with_scores, confidence_threshold):
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
                cv2.circle(frame, (int(kx), int(ky)), 4, (100,205,120), -1) 



    def draw_connections(self, frame, keypoints_with_scores, confidence_threshold, KEYPOINT_EDGE_INDS_TO_COLOR):
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
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)

    def main(self, KEYPOINT_EDGE_INDS_TO_COLOR):
        model, input_size, interpreter = self.get_model()
        cap = cv2.VideoCapture("https://player.vimeo.com/external/376858866.sd.mp4?s=d963a730f18ac7e9e93a0664dec43eab3ae41617&amp;profile_id=164&amp;oauth2_token_id=57447761")

        while cap.isOpened():
            ret, frame = cap.read()
            
            # if frame is read correctly success is True
            if not ret:
                print("Frame not found!!")
                break
            # Reshape image
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), input_size,input_size)

            if interpreter:
                
                input_image = tf.cast(img, dtype=tf.uint8)

                # Setup input and output 
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Make predictions 
                # Starting time for the process
                t1 = time.time()
                interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
                interpreter.invoke()
                keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
                # Ending time for the process
                t2 = time.time()
                # Number of frames that appears within a second
                fps = 1/(t2 - t1)
            else:
                # SavedModel format expects tensor type of int32.
                img = tf.cast(img, dtype=tf.int32)
                # Run model inference.
                # Starting time for the process
                t1 = time.time()
                outputs = model(img)
                # Output is a [1, 1, 17, 3] tensor.
                keypoints_with_scores = outputs['output_0'].numpy()
                t2 = time.time()

            # Number of frames that appears within a second
            fps = 1/(t2 - t1)
            
            # Rendering 
            self.draw_connections(frame, keypoints_with_scores, self.args.confidence_threshold, KEYPOINT_EDGE_INDS_TO_COLOR)
            self.draw_keypoints(frame, keypoints_with_scores, self.args.confidence_threshold)
            
            dim = (frame.shape[1], frame.shape[0])
            resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            # display the FPS
            cv2.putText(resized, 'FPS : {:.2f}'.format(fps), (int((frame.shape[1] * 75) /100), 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (188, 205, 54), 2, cv2.LINE_AA)
            cv2.imshow(f'Single Pose Estimation with {self.args.model}', resized)
            
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Single Pose Estimation', description='Single pose estimation with movenet and tensorflow')
    parser.add_argument("-m", "--model", type=str, default='movenet_lightning',
                        help="Model to use : 'movenet_lightning' or 'movenet_thunder'")
    parser.add_argument('-i', '--input', type=str, default='camera',
                        help="path to video/image file to use as input")
    parser.add_argument("-s", "--confidence_threshold", default=0.2, type=float,
                        help="Minimum confidence score for a keypoint to be visualized")

    args = parser.parse_args()
    
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

    pose_estimation = PoseEstimation(args)
    pose_estimation.main(KEYPOINT_EDGE_INDS_TO_COLOR)