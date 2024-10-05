import re
from datetime import datetime, timedelta
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import cv2
import os
from joblib import load  # For loading the logistic regression model

class UnifiedDetector:
    def __init__(self):
        # Load fine-tuned BERT model for text classification
        finetuned_bert_model_path = './models/bert_finetuned/'  # Path to fine-tuned BERT files
        self.bert_tokenizer = BertTokenizer.from_pretrained(finetuned_bert_model_path)
        self.bert_model = TFBertForSequenceClassification.from_pretrained(finetuned_bert_model_path, num_labels=2)

        # Load EfficientNetB0 for Image/Video Detection (using local weights)
        efficientnet_weights_path = './models/efficientnetb0_notop.h5'
        self.efficientnet_model = tf.keras.applications.EfficientNetB0(
            weights=efficientnet_weights_path, 
            include_top=False, 
            input_shape=(224, 224, 3)  # Using 224x224 for EfficientNet
        )
        self.efficientnet_model.trainable = False  # Use EfficientNet as a feature extractor

        # Load the pre-trained logistic regression classifier for deepfake detection
        self.logistic_regression_model = load('./models/logistic_regression_deepfake.joblib')  # Path to logistic regression model
        self.scaler = load('./models/scaler_deepfake.joblib')  # Load scaler
        self.pca = load('./models/pca_deepfake.joblib')  # Load PCA

    # -------- Text Detection (Fine-tuned BERT-based) --------
    def preprocess_text(self, text):
        # Preprocess text using the fine-tuned BERT tokenizer
        inputs = self.bert_tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=128)
        return inputs

    def predict_text(self, text):
        # Predict AI-generated text using the fine-tuned BERT model
        inputs = self.preprocess_text(text)
        outputs = self.bert_model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask'])
        predictions = tf.nn.softmax(outputs.logits, axis=1)  # Softmax to get confidence scores
        predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
        confidence = predictions.numpy()[0][predicted_label]
        return predicted_label, confidence

    # -------- Image Detection (Deepfake with EfficientNet + Logistic Regression) --------
    def preprocess_image(self, img_path):
        # Preprocess images for EfficientNet
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)  # Expand dimensions for batch processing
        return img_array / 255.0  # Normalize image to [0, 1] range

    def predict_image(self, img_path):
        # Predict deepfake images using EfficientNet and logistic regression classifier
        img = self.preprocess_image(img_path)
        features = self.efficientnet_model(img)  # Extract features using EfficientNet
        features_flattened = features.numpy().reshape(1, -1)  # Flatten the features
        
        # Apply scaling and PCA
        features_scaled = self.scaler.transform(features_flattened)
        features_pca = self.pca.transform(features_scaled)

        # Use logistic regression classifier to predict if the image is real or fake
        prediction = self.logistic_regression_model.predict(features_pca)
        confidence = self.logistic_regression_model.predict_proba(features_pca).max()
        return prediction[0], confidence

    # -------- Video Detection (Deepfake with EfficientNet + Logistic Regression) --------
    def extract_frames_from_video(self, video_path, frame_rate=10):
        """
        Extract frames from video and process each frame for deepfake detection.
        - video_path: Path to the video file
        - frame_rate: Extract 1 frame every 'frame_rate' frames
        """
        video = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        success, frame = video.read()

        while success:
            if count % frame_rate == 0:  # Extract a frame every 'frame_rate' frames
                resized_frame = cv2.resize(frame, (224, 224))  # Resize to match EfficientNet input
                frames.append(resized_frame)
            success, frame = video.read()
            count += 1

        video.release()
        return np.array(frames)

    def predict_video(self, video_path):
        # Extract frames from the video
        frames = self.extract_frames_from_video(video_path)
        features_list = []

        for frame in frames:
            # Preprocess and extract features for each frame
            frame = np.expand_dims(frame, axis=0) / 255.0  # Normalize the frame
            features = self.efficientnet_model(frame)  # Extract features
            features_flattened = features.numpy().reshape(1, -1)  # Flatten the features
            
            # Apply scaling and PCA
            features_scaled = self.scaler.transform(features_flattened)
            features_pca = self.pca.transform(features_scaled)

            features_list.append(features_pca)

        # Average the features across all frames
        avg_features = np.mean(features_list, axis=0)

        # Use logistic regression to predict if the video is real or fake
        prediction = self.logistic_regression_model.predict(avg_features)
        confidence = self.logistic_regression_model.predict_proba(avg_features).max()
        return prediction[0], confidence

    # -------- Fake Account Detection (Rule-based) --------
    def detect_fake_account(self, account_data):
        """
        Detect if an account is likely fake based on its features.
        """
        reasons = []

        # 1. Check if the account is newly created
        creation_date = account_data.get('creation_date')
        if creation_date:
            if self.is_new_account(creation_date):
                reasons.append("Account is newly created.")

        # 2. Check if the account has a profile picture
        profile_picture = account_data.get('profile_picture', None)
        if not profile_picture or profile_picture == "default":
            reasons.append("No profile picture or using default profile picture.")

        # 3. Check the username pattern
        username = account_data.get('username', '')
        if self.is_suspicious_username(username):
            reasons.append("Username appears suspicious (random characters or pattern).")

        # 4. Check follower/following count
        followers = account_data.get('followers', 0)
        following = account_data.get('following', 0)
        if followers < 10 and following > 1000:
            reasons.append("Low follower count but follows many people.")
        elif followers == 0 and following == 0:
            reasons.append("No followers and no following.")

        # 5. Check if the account has posted any content
        posts = account_data.get('posts', 0)
        if posts == 0:
            reasons.append("No posts available.")

        # 6. Check bio content
        bio = account_data.get('bio', '')
        if self.is_generic_or_empty_bio(bio):
            reasons.append("Bio is empty or generic.")

        # 7. Check if the account has a URL in bio
        url_in_bio = account_data.get('url_in_bio', '')
        if url_in_bio:
            reasons.append("URL in bio, which is a common trait of promotional/spam accounts.")

        # Return a verdict
        if reasons:
            return f"Fake Account Detected: {'; '.join(reasons)}"
        else:
            return "Account appears real."

    def is_new_account(self, creation_date_str):
        """
        Check if the account creation date is too recent (within the last 30 days).
        """
        try:
            creation_date = datetime.strptime(creation_date_str, '%Y-%m-%d')
            if datetime.now() - creation_date < timedelta(days=30):
                return True
        except ValueError:
            pass
        return False

    def is_suspicious_username(self, username):
        """
        Detect if a username looks suspicious (random characters, too many special characters).
        """
        if re.match(r'^[a-zA-Z0-9_.]{3,}$', username):
            # Check for excessive numbers, repeating characters, or multiple underscores
            if len(re.findall(r'[0-9]', username)) > 4 or '__' in username or username.count('_') > 2:
                return True
        else:
            return True
        return False

    def is_generic_or_empty_bio(self, bio):
        """
        Check if the bio is generic or empty.
        """
        generic_phrases = ['just living life', 'love to travel', 'happy person', 'living my best life']
        if not bio or any(phrase in bio.lower() for phrase in generic_phrases):
            return True
        return False
