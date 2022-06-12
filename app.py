import streamlit as st
import pandas as pd
import joblib

# Title
st.header('Dementia Classification App')

# Dropdown input
gender = st.selectbox('Select Gender of The Subject', ('Male', 'Female'))

# Number input 1
age = st.number_input('Enter Age of The Subject', 10, 100)

# Slider input 1
educ = st.number_input('Enter Educational Level of The Subject', 1, 5)

# Slider input 2
ses = st.number_input('Enter Socio-Economic Status Level of The Subject', 1, 5)

# Slider input 3
mmse = st.number_input('Enter Mini-Mental State Examination Score of The Subject', 0, 30)

# Input bar 5
etiv = st.number_input('Enter Estimated Total Intercranial Volume of The Brain in Cubic Milimeters of The Subject', 1000.0 , 2000.0)

# Input bar 6
nwbv = st.number_input('Enter normalized Whole Brain Volume in Miligrams of The Subject', 0.5, 1.0)

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[gender, age, educ, ses, mmse, etiv, nwbv]], 
                     columns = ['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV'])
    X = X.replace(['Male', 'Female'], ['M', 'F'])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    if prediction == '0.0':
        st.text('The Subject Is Cognitively Normal')
    elif prediction == '0.5':
        st.text('The Subject Has Very Mild Dementia')
    elif prediction == '1.0':
        st.text('The Subject Has Mild Dementia')
    elif prediction == '2.0':
        st.text('The subject has moderate dementia')