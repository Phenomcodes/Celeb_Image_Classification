import streamlit as st
from PIL import Image
import cv2
import numpy as np
from process import get_faces,w2d,name_from_number
import joblib

st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #404040;  /* light blue */
    }

    /* Optional: Center and style title text */
    h1 {
        color: #3cb371;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("CELEBRITY IMAGE CLASSIFIER")

#Load Images
img1 = Image.open("./images/Cristiano.jpg")
img2 = Image.open("./images/10026.jpg")
img3 = Image.open("./images/10036.jpg")
img4 = Image.open("./images/10391.jpg")

#Resize all images to the same size
size=(200, 200)
img1 = img1.resize(size)
img2 = img2.resize(size)
img3 = img3.resize(size)
img4 = img4.resize(size)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image(img1, use_container_width=False)
    st.markdown(
        "<p style='text-align: center; color: red; font-size: 16px;'>Cristiano Ronaldo</p>",
        unsafe_allow_html=True
    )
with col2:
    st.image(img2,use_container_width=False)
    st.markdown(
        "<p style='text-align: center; color: red; font-size: 16px;'>Lebron James</p>",
        unsafe_allow_html=True
    )
with col3:
    st.image(img3, use_container_width=False)
    st.markdown(
        "<p style='text-align: center; color: red; font-size: 16px;'>Lewis Hamilton</p>",
        unsafe_allow_html=True
    )
with col4:
    st.image(img4, use_container_width=False)
    st.markdown(
        "<p style='text-align: center; color: red; font-size: 16px;'>Simone Biles</p>",
        unsafe_allow_html=True
    )

# Create the upload box
uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png", "gif", "bmp"])

#write saved model back
model = joblib.load("./artifacts/saved_model.pkl")



# If an image is uploaded
if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    #get face from uploaded image
    face = get_faces(image)
    try:
        raw_img = cv2.resize(face, (32, 32))
        trans_img = w2d(raw_img, 'db1', 5)
    #scaled_trans_img = cv2.resize(trans_img, (32, 32))
        combined_image = np.vstack((raw_img.reshape(32 * 32 * 3, 1), trans_img.reshape(32 * 32, 1)))
        input_image = np.array(combined_image).T
        prediction = model.predict(input_image)[0]
        prediction_name = name_from_number(prediction)
        st.markdown(f"<h3 style='text-align: center; color: blue; font-weight: bold;'>Predicted Celebrity: {prediction_name}</h3>",unsafe_allow_html=True)
    except:
        st.markdown(f"<h3 style='text-align: center; color: blue; font-weight: bold;'>Image is not clear </h3>",unsafe_allow_html=True)
