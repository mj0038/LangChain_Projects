# Import necessary libraries and modules
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file for secure access to sensitive data like API keys.
import streamlit as st  # Import Streamlit for creating web app UI
import os  # Import os module for interacting with the operating system
import pathlib  # Import pathlib for handling filesystem paths
import textwrap  # Import textwrap for text processing
from PIL import Image  # Import Image from PIL for image processing
import google.generativeai as genai  # Import Google's Generative AI module

# Fetch the Google API key from environment variables and configure the Generative AI
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Generative Model with the specified model name
model = genai.GenerativeModel('gemini-pro-vision')
# Start a chat session with the model, initializing the history as empty
chat = model.start_chat(history=[])

def get_gemini_response(input, image, prompt):
    # Generate content using the model based on the input, image, and additional prompt
    response = model.generate_content([input, image[0], prompt])
    return response.text  # Return the text part of the response
    

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        # Prepare the image data in the required format
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data  # Include the byte data of the image
            }
        ]
        return image_parts
    else:
        # If no file is uploaded, raise a FileNotFoundError
        raise FileNotFoundError("No file uploaded")


# Initialize the Streamlit app with a specific page title
st.set_page_config(page_title="Gemini Image Demo")

# Display the main header of the app
st.header("Gemini Image Application")
# Create a text input box for user to enter input prompts
input = st.text_input("Input Prompt: ", key="input")
# Create a file uploader widget to allow users to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize image variable
image = ""   
if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)
    # Display the uploaded image in the Streamlit app with a caption
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Create a submit button in the app
submit = st.button("Give me more information about the image")

# Define the input prompt for the model
input_prompt = """
               You are an expert in understanding landscape images. 
               Describe in depth what is in the picture
               """

# When the submit button is clicked
if submit:
    # Process the uploaded image
    image_data = input_image_setup(uploaded_file)
    # Get the model's response by passing the input, processed image, and additional prompt
    response = get_gemini_response(input_prompt, image_data, input)
    # Display the response in the app under the "Response" subheader
    st.subheader("Response")
    st.write(response)
    # Append the response to the chat history
    st.session_state['chat_history'].append(("Bot", response))

# Display the chat history subheader
st.subheader("Chat History")
# Iterate through chat history and display each entry
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")