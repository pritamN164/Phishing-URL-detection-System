# Phishing URL Detection using Machine Learning

## Project Overview
This project detects whether a URL is **phishing or legitimate** using Machine Learning and Natural Language Processing techniques.  
The model analyzes URL text patterns and predicts if the link is malicious.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Pickle

## Project Files
app.py – Application used to test URL predictions  
train_model.py – Script used to train the machine learning model  
phishing_model.pkl – Saved trained ML model  
tfidf_vectorizer.pkl – Saved TF-IDF vectorizer  
model_metrics.pkl – Model performance metrics  
requirements.txt – Required Python libraries  

## Dataset
The dataset used in this project contains phishing and legitimate URLs.
Due to file size limitations, the dataset is not uploaded to this repository.
Download it from:https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

Download the dataset and place it in the project folder as:
malicious_phish.csv

## Installation

Clone the repository

git clone https://github.com/pritamN164/Phishing-URL-detection-System.git

Move into the project folder

cd Phishing-URL-detection-System

Install required libraries


## Run the Project

Run the application

python app.py

To retrain the model

python train_model.py

## Author
Pritam Nayak
