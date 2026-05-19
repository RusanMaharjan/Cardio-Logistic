import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import predict_cardio

features, target, X, Y, scaler, model, Y_pred, cr, cm = predict_cardio()

st.header('Cardiovascular Disease Prediction System')
st.subheader('Using Logistic Regression')



# age = st.text_input(
#     'Age',
#     placeholder='Enter your age (25 - 66)'
# )

# Code for creating sidebar
st.sidebar.header(
    'Features'
)

age = st.sidebar.slider(
    'Age',
    min_value=24,
    max_value=66,
    value=30,
    step=1
)

height = st.sidebar.slider(
    'Height',
    min_value=54,
    max_value=250,
    value=90,
    step=1
)

weight = st.sidebar.slider(
    'Weight',
    min_value=10,
    max_value=200,
    value=60,
    step=1
)

ap_hi = st.sidebar.slider(
    'Systolic Pressure',
    min_value=90,
    max_value=200,
    value=120,
    step=1
)

ap_lo = st.sidebar.slider(
    'DySystolic Pressure',
    min_value=40,
    max_value=90,
    value=80,
    step=1
)

cholestrol_options = {1: 'Healthy', 2: 'Mild', 3: 'High Cholestrol'}
cholestrol = st.sidebar.radio(
    'Cholestrol',
    options=list(cholestrol_options.keys()),
    format_func = lambda x : cholestrol_options.get(x)
)

glucose_options = {1: 'Healthy', 2: 'Mild', 3: 'High Glucose'}
glucose = st.sidebar.radio(
    'Glucose',
    options = list(glucose_options.keys()),
    format_func = lambda x : glucose_options.get(x)
)

smoke_options = {0: 'Doesnot Smoke', 1: 'Does Smoke'}
smoke = st.sidebar.radio(
    'Smoke',
    options = list(smoke_options.keys()),
    format_func = lambda x : smoke_options.get(x)
)

alcohol_options = {0: 'Doesnot Drink Alcohol', 1: 'Does Drink Alcohol'}
alco = st.sidebar.radio(
    'Alcohol',
    options = list(alcohol_options.keys()),
    format_func = lambda x : alcohol_options.get(x)
)

active_options = {0: 'Doesnot do PA', 1: 'Does PA'}
active = st.sidebar.radio(
    'Physical Activities',
    options = list(active_options.keys()),
    format_func = lambda x : active_options.get(x)
)

## Predict Button
if st.button('Predict Cardio'):
    input_data = pd.DataFrame([[
        age, height, weight, ap_hi, ap_lo, cholestrol, glucose, smoke, alco, active
    ]], columns=features)
    # Feature Scaling
    input_scaler = scaler.transform(input_data)
    prediction = model.predict(input_scaler)
    
    if prediction[0] == 0:
        st.write('No Cardio Disease Found!')
        st.success('Patient is Likely to be Healthy😌.') # win + >
    else:
        st.write('Cardio Disease Found!')
        st.warning('Patient is Likely to be unhealthy😭🥲.')

st.title('Visualization')
st.subheader('Confusion Matrix')

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=['Predicted Healthy [0]', 'Predicted Unhealthy [1]'],
            yticklabels=['Actual Healthy [0]', 'Actual Unhealthy [1]'])
st.pyplot(fig)

st.subheader('Classification Report')
# st.text(cr)
cr_df = pd.DataFrame(cr).transpose()
st.dataframe(cr_df.style.format(precision=2))
# st.table(cr_df)