#----------------------------
# Importing the Libraries
#----------------------------

import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
import io

# -------------------------------
# Underlying Functionality
# -------------------------------

class MultiLLM:
    # Making a class to perform Image analysis using Google's AI model.

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY") # Calling the Google's API from the .env file.
        if not self.api_key:
            raise ValueError("Missing Google API Key! Ensure it is set in the .env file.") # return's error if API key not present.
        genai.configure(api_key=self.api_key)

    def analyze_image(self, uploaded_file):
        # Main function that analyze the image for text extraction and sentiment analysis.

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_container_width=True)
            try:
                model = genai.GenerativeModel("gemini-2.0-flash") # Using Google's Gemini 2.0 flash model. 
                buffered = io.BytesIO() # Converting images to bytes for the model.
                image.save(buffered, format="JPEG") # Saving the image into buffer and converting the image into standard JPEG format.
                image_bytes = buffered.getvalue()
                response = model.generate_content([ # Making request to the model.
                    {
                        "parts": [
                            {"text": # Promt for the LLM model.
                             "You are an advanced AI assistant skilled in image analysis and natural language processing, with a strong focus on extracting text from images and providing insightful summaries. You also excel in performing sentiment analysis on the extracted content, determining the emotional tone and overall sentiment."
                            "Your task is to analyze an image provided to you for text extraction and sentiment analysis. Please follow these guidelines:"
                            "Extract any visible text from the image accurately."
                            "Summarize the extracted text in a concise manner, highlighting key points and themes."
                            "Conduct sentiment analysis on the extracted content, categorizing the sentiment as positive, negative, or neutral, and providing a brief explanation for your assessment."},
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}
                        ]
                    }
                ])
                return response.text
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            return "No file uploaded."

# -----------------------------------
# UI Configuration & Customization
# -----------------------------------

st.set_page_config(page_title="AI Image Analyzer", page_icon="🖼️", layout="centered")

st.markdown( # Custom Styling for Streamlit.
    """
    <style>
    [data-testid="stSidebar"] { display: none; }
    .css-1d391kg, .css-1v3fvcr, .css-12oz5g7 { text-align: center; }
    .stButton > button {
        border-radius: 20px !important;
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 10px 24px !important;
        font-size: 16px !important;
        border: none !important;
        cursor: pointer !important;
        transition: background-color 0.3s !important;
        margin: 10px auto !important;
        display: block !important;
    }
    .stButton > button:hover { background-color: #45a049 !important; }
    h1, h2, h3 { text-align: center !important; }
    .message { text-align: center; margin: 10px 0; padding: 10px; border-radius: 10px; }
    .assistant-message { background-color: #0f1116;  }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Image Analysis UI
# -------------------------------

st.markdown("<h2>Image Analysis</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="hidden")
    analyze = st.button("Analyze Image", use_container_width=True)

multi_llm = MultiLLM() # Object creation for the class MultiLLM.
if analyze and uploaded_file is not None:
    with st.spinner("Analyzing..."):
        result = multi_llm.analyze_image(uploaded_file)
        st.markdown(
            f'<div class="message assistant-message">Analysis Result: {result}</div>',
            unsafe_allow_html=True
        )
