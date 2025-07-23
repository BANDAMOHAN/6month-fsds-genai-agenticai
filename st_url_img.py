import streamlit as st 
import numpy as np
from PIL import Image 
import requests
from io import BytesIO
import matplotlib.pyplot as plt


st.set_page_config(page_title="Elephant Image Processor", layout="wide")
st.title("Elephant Image - Multi-Color Channel Visualizer")


@st.cache_data
def load_image():
    url="https://cdn.britannica.com/02/152302-050-1A984FCB/African-savanna-elephant.jpg"
    response=requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


elephant = load_image()
st.image(elephant,caption="Original Elephant Image",use_container_width=True)


elephant_np=np.array(elephant)
R,G,B=elephant_np[:,:,0],elephant_np[:,:,1],elephant_np[:,:,2]



red_img=np.zeros_like(elephant_np)
green_img=np.zeros_like(elephant_np)
blue_img=np.zeros_like(elephant_np)


red_img[:,:,0]=R
green_img[:,:,1]=G
blue_img[:,:,2]=B

st.subheader("RGB Channel Visualization")
col1,col2,col3=st.columns(3)

with col1:
    st.image(red_img,caption="Red Channel",use_container_width=True)
    
with col2:
    st.image(green_img,caption="Green Channel",use_container_width=True)
    
with col3:
    st.image(blue_img,caption="Blue Channel",use_container_width=True)
    
st.subheader("Colormapped GrayScale Image")

colormap=st.selectbox(
    "Choose a Matplotlib colormap",
    ["viridis","plasma","inferno","magma","berlin","cividis","hot","cool","gray","Spectral","coolwarm"]
)


elephant_gray=elephant.convert("L")
elephant_gray_np=np.array(elephant_gray)

fig,ax=plt.subplots(figsize=(6,4))
im=ax.imshow(elephant_gray_np, cmap=colormap)
plt.axis("off")
st.pyplot(fig)