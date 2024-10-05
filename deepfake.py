# import tensorflow as tf
# import cv2
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from joblib import dump  # Import joblib to save the model
# import os

# # 1. Load Pre-trained EfficientNet Model
# def load_efficientnet_model():
#     print("Loading EfficientNet model using local weights...")
#     efficientnet_weights_path = './models/efficientnetb0_notop.h5'  # Path to your local weights file
#     efficientnet_model = tf.keras.applications.EfficientNetB0(weights=efficientnet_weights_path, include_top=False, input_shape=(224, 224, 3))
#     efficientnet_model.trainable = False
#     print("EfficientNet model loaded.")
#     return efficientnet_model

# # 2. Video Frame Extraction Function
# def extract_frames_from_video(video_path, frame_rate=1):
#     """
#     Extracts frames from a video at a specified frame rate.
#     - video_path: Path to the video file
#     - frame_rate: Extract 1 frame every 'frame_rate' frames
#     """
#     video = cv2.VideoCapture(video_path)
#     frames = []
#     success, frame = video.read()
#     count = 0

#     while success:
#         if count % frame_rate == 0:  # Extract a frame every 'frame_rate' frames
#             resized_frame = cv2.resize(frame, (224, 224))  # Resize to match EfficientNet input
#             frames.append(resized_frame)
#         success, frame = video.read()
#         count += 1

#     video.release()
#     return np.array(frames)

# # 3. Feature Extraction Function
# def extract_features(efficientnet_model, frames):
#     """
#     Extract features from video frames using the pre-trained EfficientNet model.
#     - frames: A numpy array of video frames
#     """
#     print("Preprocessing frames...")
#     frames = tf.keras.applications.efficientnet.preprocess_input(frames)  # Normalize the frames
#     print("Extracting features using EfficientNet...")
#     features = efficientnet_model.predict(frames)
#     print("Feature extraction complete.")
#     features = features.reshape(features.shape[0], -1)  # Flatten the features
#     return features

# # 4. Train Logistic Regression Classifier
# def train_classifier(real_features, fake_features):
#     """
#     Train a logistic regression classifier using the extracted features.
#     - real_features: Features extracted from real videos
#     - fake_features: Features extracted from fake videos
#     """
#     print("Training logistic regression classifier...")
#     X_train = np.concatenate([real_features, fake_features])
#     y_train = np.concatenate([np.zeros(len(real_features)), np.ones(len(fake_features))])
    
#     classifier = LogisticRegression(max_iter=1000)
#     classifier.fit(X_train, y_train)
#     print("Training complete.")
    
#     # Save the trained classifier
#     model_path = './models/logistic_regression_deepfake.joblib'
#     dump(classifier, model_path)
#     print(f"Classifier saved to {model_path}.")
    
#     return classifier

# # 5. Evaluate Classifier
# def evaluate_classifier(classifier, real_features, fake_features):
#     """
#     Evaluate the logistic regression classifier on the test data.
#     - classifier: The trained logistic regression classifier
#     - real_features: Features extracted from real test videos
#     - fake_features: Features extracted from fake test videos
#     """
#     print("Evaluating classifier...")
#     X_test = np.concatenate([real_features, fake_features])
#     y_test = np.concatenate([np.zeros(len(real_features)), np.ones(len(fake_features))])
    
#     y_pred = classifier.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")
    
#     return accuracy

# # Main function to train and test classifier
# def train_and_evaluate():
#     # Load real and deepfake videos for training
#     real_videos_dir = '/Users/aether/Desktop/AI Content Detector/data/train/real/'  # Replace with the directory containing real videos
#     fake_videos_dir = '/Users/aether/Desktop/AI Content Detector/data/train/fake/'  # Replace with the directory containing fake videos

#     # Load EfficientNet model
#     efficientnet_model = load_efficientnet_model()

#     # Extract features for real videos
#     print("Processing real videos...")
#     real_features_list = []
#     for video_file in os.listdir(real_videos_dir):
#         video_path = os.path.join(real_videos_dir, video_file)
#         real_frames = extract_frames_from_video(video_path)
#         real_features = extract_features(efficientnet_model, real_frames)
#         real_features_list.append(real_features.mean(axis=0))  # Average the features across frames

#     # Extract features for fake videos
#     print("Processing fake videos...")
#     fake_features_list = []
#     for video_file in os.listdir(fake_videos_dir):
#         video_path = os.path.join(fake_videos_dir, video_file)
#         fake_frames = extract_frames_from_video(video_path)
#         fake_features = extract_features(efficientnet_model, fake_frames)
#         fake_features_list.append(fake_features.mean(axis=0))  # Average the features across frames

#     # Convert lists to NumPy arrays
#     real_features = np.array(real_features_list)
#     fake_features = np.array(fake_features_list)

#     # Split the dataset into training and testing sets
#     real_train, real_test = train_test_split(real_features, test_size=0.2, random_state=42)
#     fake_train, fake_test = train_test_split(fake_features, test_size=0.2, random_state=42)

#     # Train the classifier
#     classifier = train_classifier(real_train, fake_train)

#     # Evaluate the classifier
#     evaluate_classifier(classifier, real_test, fake_test)

# if __name__ == "__main__":
#     train_and_evaluate()




import tensorflow as tf
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import dump, load  # Import dump and load from joblib
import os

# 1. Load Pre-trained EfficientNet Model
def load_efficientnet_model():
    print("Loading EfficientNet model using local weights...")
    efficientnet_weights_path = './models/efficientnetb0_notop.h5'
    efficientnet_model = tf.keras.applications.EfficientNetB0(weights=efficientnet_weights_path, include_top=False, input_shape=(224, 224, 3))
    efficientnet_model.trainable = False
    print("EfficientNet model loaded.")
    return efficientnet_model

# 2. Video Frame Extraction Function
def extract_frames_from_video(video_path, frame_rate=1):
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    count = 0

    while success:
        if count % frame_rate == 0:  # Extract a frame every 'frame_rate' frames
            resized_frame = cv2.resize(frame, (224, 224))  # Resize to match EfficientNet input
            frames.append(resized_frame)
        success, frame = video.read()
        count += 1

    video.release()
    return np.array(frames)

# 3. Feature Extraction Function
def extract_features(efficientnet_model, frames):
    print("Preprocessing frames...")
    frames = tf.keras.applications.efficientnet.preprocess_input(frames)  # Normalize the frames
    print("Extracting features using EfficientNet...")
    features = efficientnet_model.predict(frames)
    print("Feature extraction complete.")
    features = features.reshape(features.shape[0], -1)  # Flatten the features
    return features

# 4. Train Logistic Regression Classifier with Cross-Validation and Regularization
def train_classifier(real_features, fake_features):
    print("Training logistic regression classifier...")

    # Combine features and labels
    X_train = np.concatenate([real_features, fake_features])
    y_train = np.concatenate([np.zeros(len(real_features)), np.ones(len(fake_features))])

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Save the scaler
    dump(scaler, './models/scaler_deepfake.joblib')

    # Dimensionality reduction with PCA (optional but can improve accuracy)
    pca = PCA(n_components=50)  # Adjust n_components to control dimensionality
    X_train_pca = pca.fit_transform(X_train_scaled)
    dump(pca, './models/pca_deepfake.joblib')  # Save PCA model

    # Logistic Regression with Cross-Validation and Regularization
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l2'],  # Use L2 regularization
        'max_iter': [1000],  # Max iterations to ensure convergence
        'solver': ['lbfgs']  # Solver for logistic regression
    }
    
    logistic_regression = LogisticRegression()
    classifier = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy')  # Cross-validation
    classifier.fit(X_train_pca, y_train)

    print(f"Training complete. Best parameters: {classifier.best_params_}")

    # Save the trained classifier
    model_path = './models/logistic_regression_deepfake.joblib'
    dump(classifier, model_path)
    print(f"Classifier saved to {model_path}.")

    return classifier

# 5. Evaluate Classifier
def evaluate_classifier(classifier, real_features, fake_features):
    print("Evaluating classifier...")

    # Combine test features and labels
    X_test = np.concatenate([real_features, fake_features])
    y_test = np.concatenate([np.zeros(len(real_features)), np.ones(len(fake_features))])

    # Load the scaler and PCA model
    scaler = load('./models/scaler_deepfake.joblib')
    pca = load('./models/pca_deepfake.joblib')

    # Scale and transform the features
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    # Predict and evaluate accuracy
    y_pred = classifier.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    return accuracy

# Main function to train and test classifier
def train_and_evaluate():
    real_videos_dir = '/Users/aether/Desktop/AI Content Detector/data/train/real/'
    fake_videos_dir = '/Users/aether/Desktop/AI Content Detector/data/train/fake/'

    # Load EfficientNet model
    efficientnet_model = load_efficientnet_model()

    # Extract features for real videos
    print("Processing real videos...")
    real_features_list = []
    for video_file in os.listdir(real_videos_dir):
        video_path = os.path.join(real_videos_dir, video_file)
        real_frames = extract_frames_from_video(video_path)
        real_features = extract_features(efficientnet_model, real_frames)
        real_features_list.append(real_features.mean(axis=0))

    # Extract features for fake videos
    print("Processing fake videos...")
    fake_features_list = []
    for video_file in os.listdir(fake_videos_dir):
        video_path = os.path.join(fake_videos_dir, video_file)
        fake_frames = extract_frames_from_video(video_path)
        fake_features = extract_features(efficientnet_model, fake_frames)
        fake_features_list.append(fake_features.mean(axis=0))

    # Convert lists to NumPy arrays
    real_features = np.array(real_features_list)
    fake_features = np.array(fake_features_list)

    # Split the dataset into training and testing sets
    real_train, real_test = train_test_split(real_features, test_size=0.2, random_state=42)
    fake_train, fake_test = train_test_split(fake_features, test_size=0.2, random_state=42)

    # Train the classifier
    classifier = train_classifier(real_train, fake_train)

    # Evaluate the classifier
    evaluate_classifier(classifier, real_test, fake_test)

if __name__ == "__main__":
    train_and_evaluate()
