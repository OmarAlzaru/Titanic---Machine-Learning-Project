import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 

def load_model():
    with open('C:/Users/New/GitProjects/MyProjects/Predicting-Titanic-Survivors/appmodel.pkl', 'rb') as file:
        mod = pickle.load(file)
    return mod
pickled_model = load_model()

def view_home_page():
    st.title("Welcome to the Titanic Survival Prediction Web App")
    st.write("""## Input some data to determine if you would have survived the titanic shipwreck of 1912""")
    
    # Create Categories for User Input Fields
    sex = ("Female" , "Male")
    pclass = ("1st Class (Top Level)", "2nd Class (Middle Level)" , "3rd Class (Bottom Level)")
    embarked = ("Cherbourg, France" , "Queenstown, Ireland", "Southampton, England")
    ticket_type = ("Regular" , "Lettered")

    # Creating User Input Fields
    Age = st.slider("Age in years", 0, 100, 1)
    Sex	= st.selectbox("Sex", sex)
    Pclass = st.selectbox("Passenger Class", pclass)
    SibSp = st.slider("How Many Siblings/Spouses Do You Intend to Take on the Trip", 0, 25, 1) 
    Parch = st.slider("How Many Parents/Children Do You Intend to Take on the Trip", 0, 25, 1) 
    Fare = st.slider("How Much Would You Pay to Go on the Titanic Trip?", 0, 550, 1) 
    Embarked = st.selectbox("Boarding Point", embarked)
    Ticket_Type = st.selectbox("Type of Boarding Ticket", ticket_type)


    # Create DataFrame from input data
    data_dict = {'Pclass': [Pclass], 'Sex': [Sex], 'Age': [Age],  'SibSp': [SibSp], 'Parch': [Parch], 'Fare': [Fare], 'Embarked': [Embarked], 'Ticket_Type':[Ticket_Type]}
    
    data = pd.DataFrame(data_dict)
    data["Sex"].replace(['Male', 'Female'], [1, 0], inplace=True)
    data["Pclass"].replace(["1st Class (Top Level)", "2nd Class (Middle Level)" , "3rd Class (Bottom Level)"], [1, 2, 3], inplace=True)
    data["Embarked"].replace(["Cherbourg, France" , "Queenstown, Ireland", "Southampton, England"], [0,1,2], inplace=True)
    data["Ticket_Type"].replace(["Regular" , "Lettered"], [0, 1], inplace=True)

    # Generate Prediction
    ok = st.button("Did I Survive the Shipwreck???")
    
    if ok:
        X = data
        prediction = pickled_model.predict(X)
        if prediction == 1:
            st.subheader(f"Result: You Survived")
        else:
            st.subheader(f"Result: You Didn't Make It")
