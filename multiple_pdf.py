# Importing necessary libraries for the app
from PyPDF2 import PdfReader  # Library for reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
import os  # For interacting with the operating system
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For embedding text using Google's Generative AI
import google.generativeai as genai  # Google's Generative AI library
from langchain.vectorstores import FAISS  # For storing and retrieving vector embeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # Conversational AI interface
from langchain.chains.question_answering import load_qa_chain  # For loading a question-answering chain
from langchain.prompts import PromptTemplate  # For creating prompt templates
from dotenv import load_dotenv  # For loading environment variables from a .env file
import streamlit as st
load_dotenv()  # Load environment variables
os.getenv("GOOGLE_API_KEY")  # Get the Google API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Configure Google Generative AI with the API key



def get_pdf_text(pdf_docs):
    # Function to extract text from a list of PDF documents
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Initialize the PDF reader
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page and concatenate
    return text  # Return the concatenated text from all PDFs

def get_text_chunks(text):
    # Function to split text into chunks using a RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)  # Split text into chunks
    return chunks  # Return the list of text chunks

def get_vector_store(text_chunks):
    # Function to convert text chunks into vector embeddings and store them using FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Initialize the embedding model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Convert text chunks to vectors and store them
    vector_store.save_local("faiss_index")  # Save the vector store locally

def get_conversational_chain():
    # Function to initialize a conversational chain with a prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)  # Initialize the conversational AI model
    chat = model.start_chat(history=[])  # Start a chat session
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Create a prompt template
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)  # Load a question-answering chain
    return chain  # Return the conversational chain

def user_input(user_question):
    # Function to handle user input and generate a response using the conversational chain
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Initialize the embedding model
    new_db = FAISS.load_local("faiss_index", embeddings)  # Load the local vector store
    docs = new_db.similarity_search(user_question)  # Find similar documents based on the user question
    chain = get_conversational_chain()  # Get the conversational chain
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )  # Generate a response using the conversational chain
    print(response)  # Print the response (useful for debugging)
    st.session_state['chat_history'].append(("User", user_question))  # Append user question to chat history
    st.session_state['chat_history'].append(("Bot", response["output_text"]))  # Append bot response to chat history
    st.write("Reply: ", response["output_text"])  # Display the response in the Streamlit app


def main():
    # Function to initialize and run the main Streamlit application
    st.set_page_config("Chat PDF")  # Set the page configuration
    st.header("Chat with PDF using Gemini💁")  # Display the header

    user_question = st.text_input("Ask a Question from the PDF Files")  # Create a text input for user questions

    if user_question:
        user_input(user_question)  # Handle user input when a question is asked
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []  # Initialize chat history in session state if not already present

    with st.sidebar:  # Create a sidebar for additional options
        st.title("Menu:")  # Display a title in the sidebar
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)  # Create a file uploader in the sidebar
        if st.button("Submit & Process"):  # Create a button to process the uploaded PDFs
            with st.spinner("Processing..."):  # Display a spinner while processing
                raw_text = get_pdf_text(pdf_docs)  # Extract text from the uploaded PDFs
                text_chunks = get_text_chunks(raw_text)  # Split the extracted text into chunks
                get_vector_store(text_chunks)  # Convert text chunks to vector embeddings and store them
                st.success("Done")  # Display a success message after processing is complete
        if st.button('Reset Chat History'):  # Create a button to reset the chat history
            st.session_state['chat_history'] = []  # Reset the chat history

    # Display chat history
    st.subheader("Chat History")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")  # Display each entry in the chat history

if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly
