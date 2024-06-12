import streamlit as st
import urllib.request
import pickle
import pandas as pd

fast_api_model_url = "http://127.0.0.1:8000/model_download"
predictor = pickle.load(urllib.request.urlopen(fast_api_model_url))

st.set_page_config(page_title="Churn Prediction App")

overview = st.container()
left, right = st.columns(2)
prediction = st.container()

geography_d = {0: "France", 1: "Germany", 2: "Spain"}
gender_d = {0: "Female", 1: "Male"}
has_credit_card_d = {0: "No", 1: "Yes"}
is_active_member_d = {0: "No", 1: "Yes"}

with overview:
    st.title("Churn Prediction App")
    st.image(
        "https://images.unsplash.com/photo-1501167786227-4cba60f6d58f?q=80&w=2970&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    )

with left:
    geography = st.selectbox("Geography", list(geography_d.keys()), format_func=lambda x: geography_d[x])
    gender = st.selectbox("Gender", list(gender_d.keys()), format_func=lambda x: gender_d[x])
    has_credit_card = st.selectbox("Has credit card", list(has_credit_card_d.keys()), format_func=lambda x: has_credit_card_d[x])
    is_active_member = st.selectbox("Is active member", list(is_active_member_d.keys()), format_func=lambda x: is_active_member_d[x])

with right:
    credit_score = st.slider("CreditScore", value=600, min_value=350, max_value=850, step=1)
    age = st.slider("Age", value=40, min_value=18, max_value=92, step=1)
    tenure = st.slider("Tenure", value=5, min_value=0, max_value=10, step=1)
    balance = st.slider("Balance", value=100000, min_value=0, max_value=251000, step=1)
    num_of_products = st.slider("Number of products", value=2, min_value=1, max_value=4, step=1)
    estimated_salary = st.slider("Estimated salary", value=100000, min_value=10, max_value=200000, step=1)

    new_tenure = tenure / age

    data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_credit_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
        "NewTenure": [new_tenure],
    })

    pred = predictor.predict(data)
    s_confidence = predictor.predict_proba(data)

    with prediction:
        st.header("Will the person resign? {0}".format("Yes" if pred[0] == 1 else "No"))
        st.subheader("Prediction confidence {0:.2f} %".format(s_confidence[pred[0]][0] * 100))