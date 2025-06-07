# GolfAI_Assignment
This project processes golf swing videos to automatically classify swing phases, annotate videos with predicted phases, and extract individual swing clips. It leverages MediaPipe Pose for landmark extraction and a RandomForestClassifier for phase classification.

## How to run
### To make a prediction / run swing clips extractor
Clone the repo and install Python version 3.12.10 and requirements.txt. Then , 
        1. To generate a labelled output with each frame labeled with its predicted class, execute the third last and second last cell in poseimg.ipynb. You also need to enter input video path and where you want to save the output video.
        2. To extract swing clips from a video, specify the input and output path as before in last cell and execute.

### To train the model again from scratch 
        1. Download the dataset from   https://drive.google.com/drive/folders/1zT9TWfjzsQVCo1euMMjUMXAddsAAiKQB?usp=sharing
        2. Place the downloaded folder in same folder as other items and execute poseimg.ipynb.

## Python version
Make sure that Python version 3.12.10 is install. Mediapipe  is not compatible with newer versions of Python.

## What this code does
Extracts pose landmarks from images using MediaPipe Pose.
Builds a labeled dataset from images organized in class-based subfolders.
Trains a RandomForestClassifier to recognize golf swing phases.
Evaluates model performance (precision, recall, F1-score).
Classifies each frame in a video and saves frame-wise predictions.
Annotates videos with predicted swing phases overlayed on each frame.
Automatically segments long videos into individual swing clips (from "Address" to "Finish") and saves them.

## Project Structure
1. Swing_events : Directory containing subfolders for each swing phase class. Each subfolder holds images for that class. It has 7 folders corresponding to 7 swing phases - Address, Toe-up, Mid-Backswing, Top, Mid-Downswing, Impact, Mid-Follow-Through, Finish (images extracted from GolfDB dataset, link can be found here - https://www.kaggle.com/datasets/marcmarais/golfdb-entire-image) . For idle class images have extracted from LSP dataset. 

2. pose_classifier.pkl : Trained RandomForest model for swing phase classification.

3. classified_frames.txt : Text file with per-frame predictions for a given video.

4. swing_clips : Output directory for extracted swing clips.


The code assumes a maximum of 1350 images per class for training.

### The swing phase classes are:
"Address", "Takeaway", "Mid-Backswing", "Top", "Mid-Downswing", "Impact", "Follow-through", "Finish"

### Smoothening of result 
The function smooth_predictions() applies temporal smoothing to the raw frame-wise classification predictions from the model. This helps reduce noise caused by frame-to-frame fluctuations — for example, when a single frame is misclassified in the middle of an otherwise correctly predicted phase like "Mid-Backswing".

For each frame, the most frequent label within a small surrounding window (e.g., 4 frames) is selected to replace the original prediction. This majority-voting approach helps eliminate brief misclassifications, resulting in more stable and accurate detection of complete swing segments.

For example 
Original predictions (noisy): [Address, Address, Top, Address, Mid-Backswing]

After smoothening : [Address, Address, Address, Address, Address]

### To qualify as a valid swing, the frame sequence must:

1. Start with: The first "Address" in a sequence
2. End with: The last "Finish" in a consecutive run
3. Minimum Length: Must span at least min_swing_length frames
4. Inclusion Rule:
- Must contain at least one "Mid-Backswing"
- Must contain at least one "Mid-Downswing"
5. Noise Tolerance: Labels like "Idle", "NoPose", or other noise do not break the swing
6. Restart Logic: If a second "Address" appears before a "Finish", restart the swing from the new "Address"

