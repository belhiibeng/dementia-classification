import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Dementia Classification App")

# Dropdown input
gender = st.selectbox("Select Gender", ("Male", "Female"))

# Input bar 1
age = st.number_input("Enter Age")

# Input bar 2
ses = st.number_input("Enter SES")

# Input bar 3
mmse = st.number_input("Enter MMSE")

# Input bar 4
etiv = st.number_input("Enter eTIV")

# Input bar 5
nwbv = st.number_input("Enter nWBV")

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[gender, age, ses, mmse, etiv, nwbv]], 
                     columns = ['M/F', 'Age', 'SES', 'MMSE', 'eTIV', 'nWBV'])
    X = X.replace(["Male", "Female"], [1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"Your CDR is  {prediction}")
