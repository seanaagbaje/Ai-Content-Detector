# AI Content Detection and Fake Account Identification

## Overview

This project aims to create a robust pipeline for detecting AI-generated content—including text, images, and videos—and identifying fake social media accounts. By integrating machine learning, deep learning models, and rule-based techniques, this unified detection framework addresses the growing challenges posed by synthetic content and fraudulent online identities.

## Objectives

1. **AI-Generated Content Detection**:
   - **Text**: Develop a model to distinguish between human-written and AI-generated text, identifying subtle linguistic indicators typical of AI-generated content.
   - **Images**: Create a system to differentiate AI-generated images from real photographs, focusing on visual artifacts and inconsistencies.
   - **Videos**: Identify AI-generated videos, such as deepfakes, by analyzing frames for artifacts and temporal inconsistencies.

2. **Fake Account Detection**:
   - **Profile Analysis**: Identify fake social media accounts by analyzing metadata such as account creation date, post frequency, follower count, and engagement rates.
   - **Behavioral Patterns**: Score accounts based on their engagement behavior to assess authenticity.

## Data Collection

- **Text Data**: Utilizes the 20 Newsgroups Dataset for human-written content alongside AI-generated text samples from models like GPT-3 and GPT-4.
- **Image Data**: Combines AI-generated images from sources such as GANs with real photographs from public datasets (ImageNet, COCO).
- **Video Data**: Analyzes real and deepfake video samples from the UADFV dataset for identifying spatio-temporal artifacts.
- **Fake Accounts**: Scrapes social media data for real-time analysis of user profiles and interaction patterns.

## Methodology

### Preprocessing
- **Text**: Tokenization using BERT, normalization, and feature extraction.
- **Images**: Resizing and normalization of pixel values, along with data augmentation techniques.
- **Videos**: Frame extraction using OpenCV and feature extraction from each frame.
- **Account Data**: Feature engineering to extract relevant characteristics and normalization for model input.

### Model Selection
- **Text**: BERT for human vs. AI-generated text classification.
- **Images**: EfficientNet combined with logistic regression for image classification.
- **Videos**: EfficientNet for frame-level feature extraction and classification.
- **Fake Accounts**: Heuristic rules for metadata analysis.

### Training and Evaluation
- Text, image, and video detection models are trained and evaluated based on various performance metrics, including accuracy, precision, recall, and F1 score.

## Deployment

The unified detection system will be encapsulated in a `unified_detector.py` script that interfaces with a graphical user interface (GUI) for user inputs. The application will be containerized using Docker and deployed on AWS for scalability.

## Ethical Considerations

- **Data Privacy**: Ensures data anonymity in compliance with data protection regulations.
- **Bias and Fairness**: Continuous monitoring for biases to maintain objectivity across diverse datasets.

## Requirements

Make sure to install the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt

Usage

To run the detection system, execute the following command:

bash
Copy code
python unified_detector.py
Follow the on-screen instructions to input your data for analysis.

Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or features you'd like to see.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

sql
Copy code

Feel free to add any specific installation instructions, usage examples, or additional sections as needed!












download the fine-tuned bert model from here 
https://drive.google.com/drive/folders/1L1t8JPoG2wuwA2TPNaGaNkVwTg3H2-pr?usp=share_link
