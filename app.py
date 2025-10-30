import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Facial Mask Detection", page_icon="😷")

st.title("😷 Facial Mask Detection Demo")
st.write("Upload an image and the app will tell if the person is wearing a mask or not!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing...")

    # Just a fake prediction for demo (randomly says Mask / No Mask)
    result = np.random.choice(["✅ Wearing Mask", "❌ No Mask"])

    if "✅" in result:
        st.success(result)
    else:
        st.error(result)
