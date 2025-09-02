import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq  # <-- CHANGED: Import ChatGroq
import os
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables from .env file
load_dotenv()


# --- Core Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text


def get_text_chunks(text):
    """Splits a long text into smaller, manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from text chunks using Hugging Face embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain using Groq."""

    # --- CHANGED: Switched to Groq API ---
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Groq API key not found. Please add it to your .env file.")
        st.stop()

    # Initialize the LLM using Groq's service
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.1-8b-instant",  # Uses Llama 3, a great model
        groq_api_key=groq_api_key
    )

    # Set up memory to retain conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the final chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# --- Streamlit UI ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
    st.header("Chat with your PDFs 🚀")

    # Sidebar for PDF Upload and Processing
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file before processing.")
                return

            with st.spinner("Processing... This may take a moment."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("Failed to extract text from the PDFs. Please check if the files are valid.")
                    return

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete! You can now ask questions.")

    # Main Chat Interface
    if "conversation" not in st.session_state:
        st.info("Please upload and process your PDFs using the sidebar to begin.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask a question about your documents:"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation.invoke({"question": user_question})
                    answer = response.get('answer', 'Sorry, I could not find an answer.')
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred while communicating with the LLM: {e}")


if __name__ == "__main__":
    main()