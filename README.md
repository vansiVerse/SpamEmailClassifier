# Spam Email Detection

This project is a Python script that detects spam emails using machine learning. It uses the UCI SMS Spam Collection dataset, processes the data, extracts features, trains a Naive Bayes classifier, and allows you to classify new emails as spam or not spam.

## Features
- Data loading and preprocessing
- Feature extraction with TF-IDF
- Model training (Naive Bayes)
- Model evaluation
- Classify new emails

## How to Run
1. Ensure you have Python 3.7+ installed.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script:
   ```sh
   python spam_detector.py
   ```

## Dataset
The script will automatically download the UCI SMS Spam Collection dataset if not present.

## Project Structure
- `spam_detector.py`: Main script for training and classifying emails
- `requirements.txt`: Python dependencies

---
