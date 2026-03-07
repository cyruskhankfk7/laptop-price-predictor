import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

# Load model
pipe = pickle.load(open('pipe.pkl','rb'))

# Load dataset
data = pd.read_csv("traineddata.csv")

st.title("💻 Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', data['Company'].unique())

# Laptop type
type_name = st.selectbox('Type', data['TypeName'].unique())

# RAM
ram = st.selectbox('Ram (GB)', [2,4,6,8,12,16,24,32,64])

# OS
os = st.selectbox('Operating System', data['OpSys'].unique())

# Weight
weight = st.number_input('Weight of laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS Display',['No','Yes'])

# Screen Size
screen_size = st.number_input('Screen Size (inches)')

# Resolution
resolution = st.selectbox(
'Screen Resolution',
['1920x1080','1366x768','1600x900','3840x2160',
 '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
)

# CPU
cpu = st.selectbox('CPU', data['CPU_name'].unique())

# HDD
hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])

# SSD
ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])

# GPU
gpu = st.selectbox('GPU Brand', data['Gpu brand'].unique())


if st.button('Predict Price'):

    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])

    ppi = ((X_resolution**2)+(Y_resolution**2))**0.5 / screen_size

    query = pd.DataFrame({
        'Company':[company],
        'TypeName':[type_name],
        'Ram':[ram],
        'Weight':[weight],
        'TouchScreen':[touchscreen],
        'IPS':[ips],
        'PPI':[ppi],
        'CPU_name':[cpu],
        'HDD':[hdd],
        'SSD':[ssd],
        'Gpu brand':[gpu],
        'OpSys':[os]
    })

    prediction = int(np.exp(pipe.predict(query)[0]))

    st.success(f"Predicted Price: ₹{prediction-1000} to ₹{prediction+1000}")