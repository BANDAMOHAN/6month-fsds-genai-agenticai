import streamlit as st 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


st.set_page_config(page_title="Prabhas Image Processor",layout="wide")

st.title("Prabhas Image - Multi-color Channel Visualizer")

@st.cache_data
def load_image():
    path=r"D:\DEVI (MS-OFFICE -WORD- PAD)\OneDrive\Pictures\darling.jpg"
    return Image.open(path).convert("RGB")
 
 
darling = load_image()
st.image(darling,caption="Original Prabhas Image",use_container_width=True)
 
 
darling_np=np.array(darling)
R,G,B=darling_np[:,:,0],darling_np[:,:,1],darling_np[:,:,2]
 
 
red_img=np.zeros_like(darling_np)
green_img=np.zeros_like(darling_np)
blue_img=np.zeros_like(darling_np)
 
 
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
    



st.subheader("Color Mapped Grayscale Image")

colormap=st.selectbox(
    "Choose a Matplotlib colormap",
    ["viridis","plasma","inferno","magma","cividis","hot","cool","gray","Spectral","berlin","coolwarm"]
)



darling_gray=darling.convert("L")
darling_gray_np=np.array(darling_gray)



fig,ax=plt.subplots(figsize=(6,4))
ax.imshow(darling_gray_np,cmap=colormap)
ax.axis("off")
st.pyplot(fig)