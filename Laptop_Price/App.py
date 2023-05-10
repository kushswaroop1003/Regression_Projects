import streamlit as st
import pickle
import numpy as np

#import model

pipe = pickle.load(open("pipe.pkl", "rb"))
dataset = pickle.load(open("dataset.pkl", 'rb'))
st.title("Laptop Predictor")

#brand
company = st.selectbox("Brand", dataset["Company"].unique())

#type of laptop
type = st.selectbox("Type", dataset["TypeName"].unique())

# Ram
RAM = st.selectbox("Ram (in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])

os = st.selectbox('OS',dataset['OpSys'].unique())

#Weight

weight = st.number_input("Weight of the Laptop")

#Touchscreen

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

#IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

#screen_size
screen_size = st.number_input("Screen Size")


#screen Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu

cpu = st.selectbox("CPU", dataset["CPU Brand"].unique())

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',dataset['GPU'].unique())



if st.button("Predict Price"):
    ppi=None
    if touchscreen =="Yes":
        touchscreen = 1
    else:
        touchscreen = 0

    if ips =="Yes":
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    A_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (A_res**2))**.5/screen_size


    query = np.array([company,type,RAM,os,weight,touchscreen,cpu,ips,ppi,ssd,gpu])
    query =query.reshape(1,11)
    st.title(pipe.predict(query))

