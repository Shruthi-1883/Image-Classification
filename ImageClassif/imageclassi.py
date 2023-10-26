import streamlit as st
import os
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
#%matplotlib inline
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray
import pickle
model = pickle.load(open("C:\\Users\\Shruthi\\OneDrive\\Desktop\\PA\\model.pkl",'rb'))
n=st.file_uploader("Upload the image")
#n = "/content/images.jpg"
button=st.button("Click to see the result")
if button:
    image = imread(n)
    gray_image = rgb2gray(image)
    resized = resize(gray_image,(240,240))
    flattened = np.ndarray.flatten(resized).reshape(57600,1)
    sem_fin=np.dstack(flattened)
    fin_img = sem_fin.reshape(1,57600)
    img_data=pd.DataFrame(fin_img)
    output = model.predict(img_data)
    #imshow(n)
    if output[0][0] == 1:
     st.write("gryf")
    else:
     st.write("slyt")
