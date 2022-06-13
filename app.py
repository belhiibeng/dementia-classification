import streamlit as st
import pandas as pd
import joblib

# Title
st.header('Dementia Classification App')

# App Description
with st.container():
    st.write('This app classify the subject using machine learning model into 4 classes.')
    st.write('The 4 classes are :')
    st.write('1. Cognitively normal.')
    st.write('2. Has very mild dementia.')
    st.write('3. Has mild dementia.')
    st.write('4. Has moderate dementia.')
    st.write('To classify the subject you need to input 7 data of the subject.')  
    st.write('You can see the data description on the sidebar.')
    st.write('You can input the needed data below.')

# Data Description
with st.sidebar:
    st.subheader('Data Description')
    st.write('Educational level of the subject is a 5-level categorization.')
    st.write('Level 1 – subject received lower education than high school.')
    st.write('Level 2 – subject graduated high school.')
    st.write('Level 3 – subject received college level education.')
    st.write('Level 4 – subject graduated college.')
    st.write('Level 5 – subject received education beyond college.')
    st.write('Socio-Economic Status (SES) of the subject is a 5-level categorization.')
    st.write('Level 1 – subject belongs to the lower class of society.')
    st.write('Level 2 – subject belongs to the lower-middle class of society.')
    st.write('Level 3 – subject belongs to the middle class of society.')
    st.write('Level 4 – subject belongs to the middle-upper class of society.')
    st.write('Level 5 – subject belongs to the upper class of society.')
    st.write('Mini-Mental State Examination (MMSE) score. Range = 0 – 30, where 0 – more likely to be demented and 30 – least likely to be demented.')
    st.write('eTIV is estimated Total Intercranial Volume of the brain in cubic milimeters.')
    st.write('nWBV is normalized Whole Brain Volume in miligrams.')

# Dropdown input
gender = st.selectbox('Select Gender of The Subject', ('Male', 'Female'))

# Input bar 1
age = st.number_input('Enter Age of The Subject', 0, 150)

# Input bar 2
educ = st.number_input('Enter Educational Level of The Subject', 1, 5)

# Input bar 3
ses = st.number_input('Enter SES Level of The Subject', 1, 5)

# Input bar 4
mmse = st.number_input('Enter MMSE Score of the subject', 0, 30)

# Input bar 5
etiv = st.number_input('Enter eTIV of The Subject', 0.0 , 2000.0)

# Input bar 6
nwbv = st.number_input('Enter nWBV of The Subject', 0.0, 1.0)

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
        st.write('The Subject Is Cognitively Normal')
    elif prediction == '0.5':
        st.write('The Subject Has Very Mild Dementia')
    elif prediction == '1.0':
        st.write('The Subject Has Mild Dementia')
    elif prediction == '2.0':
        st.write('The Subject Has Moderate Dementia')