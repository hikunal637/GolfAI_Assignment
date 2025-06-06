# GolfAI_Assignment
This project processes golf swing videos to automatically classify swing phases, annotate videos with predicted phases, and extract individual swing clips. It leverages MediaPipe Pose for landmark extraction and a RandomForestClassifier for phase classification.

## How to run
Clone the repo, install requirements.txt and execute poseimg.ipynb.
## Python version
Make sure that Python version 3.12.10 is install. Mediapipe  is not compatible with newer versions of Python.

Features
Extracts pose landmarks from images using MediaPipe Pose.
Builds a labeled dataset from images organized in class-based subfolders.
Trains a RandomForestClassifier to recognize golf swing phases.
Evaluates model performance (precision, recall, F1-score).
Classifies each frame in a video and saves frame-wise predictions.
Annotates videos with predicted swing phases overlayed on each frame.
Automatically segments long videos into individual swing clips (from "Address" to "Finish") and saves them.

Project Structure
Swing_events/
Directory containing subfolders for each swing phase class. Each subfolder holds images for that class. 

pose_classifier.pkl
Trained RandomForest model for swing phase classification.

classified_frames.txt
Text file with per-frame predictions for a given video.

swing_clips/
Output directory for extracted swing clips.

Dependencies
Python 3.x

OpenCV (cv2)

MediaPipe (mediapipe)

NumPy (numpy)

scikit-learn (sklearn)

joblib

Install dependencies using:

bash
pip install opencv-python mediapipe numpy scikit-learn joblib
Usage
1. Dataset Preparation
Organize your dataset in the Swing_events folder.

Each subfolder should be named after a swing phase (e.g., Address, Takeaway, etc.) and contain images for that phase.

2. Feature Extraction and Model Training
The script reads images, extracts pose landmarks, and builds feature vectors.

It trains a RandomForestClassifier and saves the model as pose_classifier.pkl.

3. Classifying Video Frames
Use the trained model to predict the swing phase for each frame in a video.

Results are saved in classified_frames.txt.

4. Annotating Videos
Annotate each frame of a video with the predicted swing phase.

The annotated video is saved to disk (e.g., latest7.mp4).

5. Swing Clip Extraction
The script detects swing segments (from "Address" to "Finish") in a video.

It saves each detected swing as a separate video clip in the output directory.

Key Functions
extract_pose_from_image(image_path)

Extracts pose landmarks from an image and returns a flattened feature vector.

process_dataset(base_folder, max_per_class=1350)

Processes the dataset to build feature and label arrays.

classify_video(video_path, model, pose, output_txt)

Classifies each frame in a video and saves predictions.

label_and_save_video(input_video_path, output_video_path, model, pose_model)

Annotates and saves a video with predicted swing phases overlayed.

extract_address_to_finish_segments(preds, min_swing_length)

Identifies swing segments in the video based on phase predictions.

save_swing_clips(video_path, segments, output_dir)

Saves each detected swing segment as a separate video clip.

Example Workflow
python
# Step 1: Process dataset and train the model
X, y = process_dataset('Swing_events', max_per_class=1350)
# Train/test split and model training as shown in the script

# Step 2: Classify frames in a video
classify_video('sample_vid1.mp4', model, pose, 'classified_frames.txt')

# Step 3: Annotate and save labeled video
label_and_save_video('sample_vid1.mp4', 'latest7.mp4', model, pose)

# Step 4: Extract swing clips from a video
frame_results = classify_video(video_path, model, pose)
smoothed = smooth_predictions(frame_results, window_size=4)
segments = extract_address_to_finish_segments(smoothed, min_swing_length=20)
save_swing_clips(video_path, segments, output_dir='swing_clips4')
Notes
The code assumes a maximum of 1350 images per class for training.

The swing phase classes are:
"Address", "Takeaway", "Mid-Backswing", "Top", "Mid-Downswing", "Impact", "Follow-through", "Finish"

Video segmentation requires both "Mid-Backswing" and "Mid-Downswing" phases to be present in a swing for extraction.
