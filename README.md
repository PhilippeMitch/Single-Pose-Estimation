# Single Pose Estimation
**In progress...**<br>
Pose estimation is a task in computer vision and artificial intelligence (AI) where the goal is to detect and track the position and orientation of human body parts in images or videos. Usually, this is done by predicting the location of specific keypoints like hands, head, elbows, etc. in case of Human Pose Estimation.

### Human Pose Estimation with Mediapipe
The [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) task lets you detect landmarks of human bodies in an image or video. You can use this task to identify key body locations, analyze posture, and categorize movements. This task uses machine learning (ML) models that work with single images or video. The task outputs body pose landmarks in image coordinates and in 3-dimensional world coordinates.

### Human Pose Estimation with MoveNet
MoveNet (detects 17 key points & runs at 50+ fps) is a Google-based inference model developed by IncludeHealth, a digital health company. IncludeHealth unveiled the model in 2021, and solicited help from Google to support remote treatment of patients. Similar to PoseNet, the web version of MoveNet uses TensorFlow.js, and the mobile version uses TensorFlow Lite. The MoveNet models outperform Posenet (paper, blog post, model), the previous TensorFlow Lite pose estimation model, on a variety of benchmark datasets (see the evaluation/benchmark result in the table below).

There are two versions of MoveNet: 
1. The performance-oriented Lightning: is smaller, faster but less accurate than the Thunder version. It can run in realtime on modern smartphones. 
2. The accuracy-oriented Thunder:  is the more accurate version but also larger and slower than Lightning. 

The two models differ in input size and depth multiplier. In terms of input, Lightning receives a video or an image of a fixed size
(192×192) and three channels, and employs 1.0 depth multiplier. In contrast, Thunder receives an input of the size 256×256 and three channels, and employs 1.75 depth multiplier. 
