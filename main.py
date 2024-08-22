import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
print (scores.mean())

model=SVC()
model.fit(x_train,y_train)
print("for svm: ")
print(model.score(x_test,y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Define global variables and functions for the chatbot logic
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {}

# Load symptom description data
def getDescription():
    global description_list
    with open('Indicator_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

# Load symptom severity data
def getSeverityDict():
    global severityDictionary
    with open('Indicator_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

# Load symptom precaution data
def getprecautionDict():
    global precautionDictionary
    with open('Indicator_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# Function to predict disease based on symptoms
def predict_disease(symptoms_exp, num_days):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X, y)
    symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
    input_vector = pd.Series(0, index=X.columns)
    for symptom in symptoms_exp:
        input_vector[symptom] = 1
    prediction = rf_clf.predict([input_vector])[0]
    return prediction

# Main Streamlit app
def main():
    st.title('Pain Management App')

    # Load data
    getDescription()
    getSeverityDict()
    getprecautionDict()

    st.header('User Input')

    # Name input
    name = st.text_input('Your Name')

    # Symptoms experiencing input
    symptoms_exp = st.text_input('Symptoms experiencing')

    # How many days experiencing input
    num_days = st.number_input('How many days you are experiencing', min_value=1)

    # Symptom selection
    st.header('Are you experiencing any')
    symptoms_list = training.columns[:-1]  # Get list of symptoms from training data
    selected_symptoms = st.multiselect('Select symptoms', symptoms_list)

    # Predict disease button
    if st.button('Predict'):
        if symptoms_exp and num_days and selected_symptoms:
            predicted_disease = predict_disease(selected_symptoms, num_days)
            st.write(f'Hello, {name}!')
            st.header('Predicted Disease')
            st.write('Based on your symptoms, you may have:')
            st.write(predicted_disease)
            
            st.header('Description')
            description = description_list.get(predicted_disease, 'No description available')
            st.write(description)

            precaution_list = precautionDictionary.get(predicted_disease, [])
            st.header('Precaution Measures')
            st.write('Take the following measures:')
            for i, measure in enumerate(precaution_list):
                st.write(f"{i+1}. {measure}")
        else:
            st.write('Please fill in all the required inputs.')

if __name__ == '__main__':
    main()


