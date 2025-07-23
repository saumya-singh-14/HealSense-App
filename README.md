# HealSense – Smart symptom-based disease prediction

HealSense is an intelligent system that predicts possible diseases based on user-selected symptoms, offering clear explanations and preventive measures to support early detection and informed health decisions.

# Key Features

Predicts likely disease from 100+ symptoms

Simple, interactive web interface

Shows disease description and preventive advice

Uses decision tree for interpretability

# Technologies Used

Programming Language: Python

Machine Learning: scikit-learn (Decision Tree Classifier, SVM)

Web Framework: Streamlit

Data Processing: pandas, numpy

Text & Data Handling: csv, LabelEncoder

# Libraries Required

pip install pandas scikit-learn streamlit numpy

# Workflow Overview

Data loading: Load training and testing data with symptoms and associated diseases.

Preprocessing: Encode disease labels numerically using LabelEncoder, split data into training/testing sets.

Model training: Train Decision Tree and SVM models; validate performance using cross-validation.

Prediction logic: Build a binary symptom vector based on user input; predict disease; map numeric label back to disease name.

Frontend: Streamlit web app collects user inputs and displays the prediction, disease description and precautions.

# How to Clone and Run HealSense Locally

1. Clone the repository:
git clone https://github.com/saumya-singh-14/healsense.git

2. Navigate into the project folder:
cd healsense

3. Install required dependencies:
pip install pandas scikit-learn streamlit numpy pyttsx3

4. Run the application:

streamlit run app.py

You’ll see a local URL in the terminal, e.g.:

Local URL: http://localhost:8501

Open it in your browser to use the app.

# Disclaimer

HealSense is for educational and research purposes only. It is not a substitute for professional medical consultation.
