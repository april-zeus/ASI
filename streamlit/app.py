import streamlit as st
import os
import urllib.request
import pickle
import pandas as pd

fast_api_model_url = "http://127.0.0.1:8000/model_download"
model = pickle.load(urllib.request.urlopen(fast_api_model_url))
print(model)

if st.button("load new model"):
    model = pickle.load(urllib.request.urlopen(fast_api_model_url))

CreditScore = st.slider('CreditScore', 350, 850, 350, 1)
Geography = st.selectbox(["France", "Germany", "Spain"])
Gender = st.selectbox(["Female", "Male"])
Age = st.slider('Age', 18, 92, 18, 1)
Tenure = st.slider('Tenure', 0, 10, 0, 1)
Balance = st.slider('Balance', 0, 250000, 0, 1)
NumOfProducts = st.slider('NumOfProducts', 1, 4, 1, 1)
HasCrCard = st.selectbox(["No", "Yes"])
IsActiveMember = st.selectbox(["No", "Yes"])
EstimatedSalary = st.slider('EstimatedSalary', 0, 200000, 0, 1)

NewTenure = Tenure / Age

pred = None
result = "Nie wiadomo czy opuści bank"

#TODO: pass proper data to model below, right now only some boilerplate (don't know if working)

if st.button("predict"):
    d = {
        "CreditScore": [CreditScore],
        "Geography": [Geography],
        "Gender": [Gender],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCrCard],
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary": [EstimatedSalary],
        "NewTenure": [NewTenure]
    }

    df = pd.DataFrame(data=d)
    pred = model.predict(df)[0]
    if pred == 1:
        result = "Opuści bank"
    elif pred == 0:
        result = "Nie opuści banku"

st.header(result)
