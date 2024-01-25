# Langchain_Projects

This repository contains three distinct Streamlit applications, each harnessing the power of AI to perform specialized tasks. The applications integrate various advanced libraries such as Google's Generative AI, Langchain, PyPDF2, ChromaDB, FAISS, and pdf2image to deliver unique and powerful functionalities.

## General Setup Instructions
Before diving into the specific projects, ensure you have set up your environment correctly:

1. **Create a Conda Environment:**
    ```bash
    conda create -p venv python==3.10 -y
    ```
    Activate the environment:
    ```bash
    conda activate ./venv
    ```

2. **Install Required Libraries:**
    Navigate to each project directory and install the required dependencies listed in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application:**
    Launch the Streamlit application by running:
    ```bash
    streamlit run app.py
    ```
    Replace `app.py` with the name of the Python file you wish to run.

## Project 1: Landscape Image Q/A
![Landscape Image Q/A App](path_to_image_1.jpg)
**Description:**
The Landscape Image Q/A application leverages Google's Generative AI to analyze and answer questions about landscape images. Users can upload an image, and the application will provide detailed, context-aware descriptions and answer specific questions about the image.

**Technical Details:**
- Uses `google-generativeai` for generating content based on landscape images.
- Implements `streamlit` for the user interface, allowing image uploads and displaying AI-generated content.
- `python-dotenv` manages sensitive API keys and configurations.

## Project 2: ATS_Tracker for Resume
![ATS_Tracker App](path_to_image_2.jpg)
**Description:**
The ATS_Tracker for Resume is an advanced implementation of the Retriever-Answer Generator (RAG) model. It provides detailed feedback and a compatibility score by comparing resumes against job descriptions. It's an invaluable tool for job seekers looking to optimize their resumes for Applicant Tracking Systems (ATS).

**Technical Details:**
- Utilizes `langchain` and `langchain_google_genai` for implementing RAG and processing resumes.
- `PyPDF2` and `pdf2image` are used for reading and converting PDF resumes to a format suitable for analysis.
- `faiss-cpu` manages efficient similarity search for matching resumes with job descriptions.
- `chromadb` is used for storing and retrieving color information, enhancing visual aspects of the app.

## Project 3: Multi-PDF Chat
![Multi-PDF Chat App](path_to_image_3.jpg)
**Description:**
The Multi-PDF Chat application offers a customized Q/A interface, allowing users to upload multiple PDF documents and engage in a conversational Q/A session. The AI model provides accurate answers by referring to the content within the uploaded PDFs, making it an excellent tool for research and study.

**Technical Details:**
- `PyPDF2` and `pdf2image` handle PDF processing, converting them into a suitable format for the AI model.
- `langchain_google_genai` and `google-generativeai` work together to provide a conversational AI interface that refers to the uploaded PDFs for generating responses.
- `FAISS` is used for indexing and retrieving text chunks efficiently, ensuring quick response times even with multiple documents.

---

Each project in this repository demonstrates the fusion of AI and user-friendly interfaces to solve complex problems and provide valuable insights. Whether you're analyzing images, optimizing your resume, or seeking knowledge from a multitude of documents, these applications pave the way for innovative interactions and solutions.

