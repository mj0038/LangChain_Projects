# Importing necessary libraries and modules
from dotenv import load_dotenv  # Used for loading environment variables from .env file
load_dotenv()  # Load environment variables (e.g., API keys)
import base64  # Used for encoding binary data to base64
import streamlit as st  # Main library for building Streamlit apps
import os  # Used for interacting with the operating system
import io  # Used for handling input/output streams (e.g., reading/writing bytes)
from PIL import Image  # Used for image processing
import pdf2image  # Used for converting PDF files to images
import google.generativeai as genai  # Used for interacting with Google's Generative AI

# Configure the Google Generative AI with the API key from the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get a response from the generative AI model
def get_gemini_response(input, pdf_content, prompt):
    # Initialize the generative AI model with the specified model name
    model = genai.GenerativeModel('gemini-pro-vision')
    # Generate content using the model based on the input, pdf_content, and additional prompt
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text  # Return the text part of the response

# Function to process the uploaded PDF and prepare it for the generative AI model
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Convert the uploaded PDF file to images (one image per page)
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        # Select the first page of the PDF
        first_page = images[0]
        # Convert the first page image to a byte array
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        # Prepare the image data in the required format for the generative AI model
        pdf_parts = [
            {
                "mime_type": "image/jpeg",  # Specify the mime type of the image
                "data": base64.b64encode(img_byte_arr).decode()  # Encode the image bytes to base64
            }
        ]
        return pdf_parts
    else:
        # If no file is uploaded, raise an error
        raise FileNotFoundError("No file uploaded")

# Initialize the Streamlit app interface
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
# Create a text area for users to input the job description
input_text = st.text_area("Job Description: ", key="input")
# Create a file uploader widget for users to upload their resume in PDF format
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

# Confirmation message after successful PDF upload
if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

# Define submit buttons for different functionalities
submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage match")

# Define input prompts to guide the generative AI model
input_prompt1 = """
 You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
 Please share your professional evaluation on whether the candidate's profile aligns with the role. 
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First, the output should come as a percentage and then keywords missing and last final thoughts.
"""

# Handle actions when the "Tell Me About the Resume" button is clicked
if submit1:
    if uploaded_file is not None:
        # Process the uploaded PDF file
        pdf_content = input_pdf_setup(uploaded_file)
        # Get the AI model's response based on the job description and the processed PDF content
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        # Display the model's response
        st.subheader("Response")
        st.write(response)
    else:
        # Prompt the user to upload a resume if not uploaded
        st.write("Please upload the resume")

# Handle actions when the "Percentage match" button is clicked
elif submit3:
    if uploaded_file is not None:
        # Process the uploaded PDF file
        pdf_content = input_pdf_setup(uploaded_file)
        # Get the AI model's response for the percentage match based on the job description and the processed PDF content
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        # Display the model's response
        st.subheader("Response")
        st.write(response)
    else:
        # Prompt the user to upload a resume if not uploaded
        st.write("Please upload the resume")
