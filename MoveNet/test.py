import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import argparse



class PoseEstimation:

    def __init__(self, args )-> None:
        self.input                = args.input
        self.model                = args.model
        self.confidence_threshold = args.confidence_threshold
        self.output               = args.output

        print("Hello", self.model)

    def main(self):
        print("Argument ", self.confidence_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Single Pose Estimation', description='Single pose estimation with movenet and tensorflow')
    parser.add_argument("-m", "--model", type=str, default='movenet_lightning',
                        help="Model to use : 'movenet_lightning' or 'movenet_thunder'")
    parser.add_argument('-i', '--input', type=str, default='camera',
                        help="path to video/image file to use as input")
    parser.add_argument("-o","--output",
                        help="Path to output video file")
    parser.add_argument("-s", "--confidence_threshold", default=0.2, type=float,
                        help="Minimum confidence score for a keypoint to be visualized")

    args = parser.parse_args()
    print("Start here")
    pose_estimation = PoseEstimation(args)
    pose_estimation.main()