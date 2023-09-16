import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

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

class PoseEstimation:

    def __init__(self, model_name, input_image, keypoint_threshold, 
                 crop_region, close_figure, output_image_height) -> None:
        self.model_name          = model_name
        self.input_image         = input_image
        self.keypoint_threshold  = keypoint_threshold
        self.crop_region         = crop_region
        self.close_figure        = close_figure
        self.output_image_height = output_image_height

    def _keypoints_and_edges(self, keypoints_with_scores, height, width):

        keypoints              = []
        keypoint_edges         = []
        edge_colors            = []
        num_instances, _, _, _ = keypoints_with_scores.shape

        for index in range(num_instances):
            kpts_x                    = keypoints_with_scores[0, index, :, 1]
            kpts_y                    = keypoints_with_scores[0, index, :, 0]
            kpts_scores               = keypoints_with_scores[0, index, :, 2]
            kpts_absolute_xy          = np.stack(
                                            [width * np.array(kpts_x), 
                                             height * np.array(kpts_y)]
                                              , axis=-1)
            kpts_above_tresh_absolute = kpts_absolute_xy[
                                        kpts_scores > self.keypoint_threshold, :]
            keypoints.append(kpts_above_tresh_absolute)

            for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
                if (kpts_scores[edge_pair[0]] > self.keypoint_threshold and
                    kpts_scores[edge_pair[1]] > self.keypoint_threshold):

                    start_x  = kpts_absolute_xy[edge_pair[0], 0]
                    start_y  = kpts_absolute_xy[edge_pair[0], 1]
                    end_x    = kpts_absolute_xy[edge_pair[1], 0]
                    end_y    = kpts_absolute_xy[edge_pair[1], 1]
                    line_seg = np.array([[start_x, start_y], [end_x, end_y]])

                    keypoint_edges.append(line_seg)
                    edge_colors.append(color)

        if keypoints:
            keypoints_xy = np.concatenate(keypoints, axis=0)
        else:
            keypoints_xy = np.zeros((0, 17, 2))

        if keypoint_edges:
            edges_xy = np.stack(keypoint_edges, axis=0)
        else:
            edges_xy = np.zeros((0, 2, 2))

        return keypoints_xy, edges_xy, edge_colors
            

    def _draw_prediction(self, image, keypoints_xy):

        height, width, _ = image.shape
        shaped = np.squeeze(np.multiply(keypoints_xy, [height,width,1]))

        for edge, _ in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            if (c1 > self.keypoint_threshold) & (c2 > self.keypoint_threshold):      
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

    

