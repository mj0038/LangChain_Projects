from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get response from Google Gemini Pro Vision API
def get_gemini_response(input_prompt, image, custom_input):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input_prompt, image[0], custom_input])
    return response.text

# Function to process the uploaded image
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # MIME type of the uploaded file
                "data": bytes_data  # Byte data of the image
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to parse nutrition data from the AI model's response
# Function to parse nutrition data from the AI model's plain text response
def parse_nutrition_data(response_text):
    # Initialize total counts
    total_nutrition = {
        "Protein": 0,
        "Carbs": 0,
        "Fats": 0
    }

    # Split the response into lines
    lines = response_text.strip().split('\n')
    for line in lines:
        # Assume each line is in the format "Food Item - X calories"
        parts = line.split(' - ')
        if len(parts) == 2:
            item, calories_info = parts
            calories = int(calories_info.split(' ')[0])  # Extract the calorie part and convert to integer
            
            # For the sake of example, assume some distribution of macros in each food item
            # These values should be based on actual nutritional data
            total_nutrition["Protein"] += calories * 0.2  # 20% protein
            total_nutrition["Carbs"] += calories * 0.5   # 50% carbs
            total_nutrition["Fats"] += calories * 0.3    # 30% fats

    return total_nutrition


def set_bg_color():
    # Define the CSS
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #FFB6C1;
        }
        .css-1d391kg {
            background-color: #FFB6C1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to set the background color
set_bg_color()

# Initialize Streamlit app configuration
st.set_page_config(page_title="Gemini Health App")
st.header("Gemini Health App")

# User inputs
custom_input = st.text_input("Enter your query or request:", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Analyze Image")

# Default input prompt for the AI model
input_prompt = """
You are an expert nutritionist. Analyze the food items from the image
and calculate the total calories. Provide details of every food item with calorie intake
in the following format:

1. Item 1 - number of calories
2. Item 2 - number of calories
...
"""

# Handle submit action
if submit:
    if not uploaded_file:
        st.error("Please upload an image to proceed.")
    else:
        try:
            image_data = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt, image_data, custom_input)
            st.subheader("Analysis Results")
            st.write(response)

            # Parse nutrition data from the response
            nutrition_data = parse_nutrition_data(response)
            
            # Calculate total macronutrients for percentage calculation
            total_macros = sum(nutrition_data.values())
            
            # Display nutrition data in the sidebar
            st.sidebar.header("Nutrition Breakdown")
            for nutrient, value in nutrition_data.items():
                # Calculate percentage
                percent = (value / total_macros) * 100 if total_macros > 0 else 0
                st.sidebar.write(f"{nutrient}: {percent:.2f}%")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
