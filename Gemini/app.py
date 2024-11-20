# Invoice Extractor

from dotenv import load_dotenv

load_dotenv() # This will load all of the environments from the .env file.

import streamlit as st

import os

from PIL import Image

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, image, prompt):
    # loading Gemini model
    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_contetnt([input, image[0], prompt])

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type, # Get the mimie type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")

st.set_page_config(page_title = "Invoice Extractor")

st.header("Gemini Application")
input = st.text_input("Input Prompt: ", key = "input")
uploaded_file = st.file_uploader("Choose an image...", type = ["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Image.", use_column_width = True)

submit = st.button("Tell Me About The Invoice") # We created a submit button for our streamlit app

input_prompt = """
You are an expert in understanding invoices. You will
recieve images as invoices and you will be expected to
answer questions correctly and honestly based on the input images.
"""

if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)

    st.subheader("The Response is")
    st.write(response)

