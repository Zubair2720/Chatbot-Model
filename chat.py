
import streamlit as st
import os
import tempfile

from dotenv import load_dotenv, find_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ---------------------------------
# 1. Load API Key
# ---------------------------------

load_dotenv(find_dotenv(), override=True)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Gemini API Key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ---------------------------------
# 2. Streamlit Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Assistant")
st.write("Upload your PDF and ask questions about its content.")

# ---------------------------------
# 3. Initialize Session State
# ---------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None

# ---------------------------------
# 4. PDF Upload
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF Document",
    type=["pdf"]
)

# ---------------------------------
# 5. Process PDF
# ---------------------------------

if uploaded_file is not None:

    if st.button("Process PDF"):

        try:
            with st.spinner("Reading PDF..."):

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    pdf_path = temp_file.name

                # Load PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                # Text Splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                texts = text_splitter.split_documents(documents)

                if not texts:
                    st.error("No readable text found in this PDF.")
                    st.stop()

                # Create Embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=API_KEY
                )

                # Create Vector Database
                vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    collection_name="pdf_docs"
                )

                # Save in session state
                st.session_state.vector_store = vector_store
                st.session_state.chat_history = []
                st.session_state.file_name = uploaded_file.name

                st.success("PDF processed successfully!")

                st.write("Total Pages:", len(documents))
                st.write("Total Chunks:", len(texts))

        except Exception as e:
            st.error(f"Error processing PDF: {e}")

        finally:
            if "pdf_path" in locals() and os.path.exists(pdf_path):
                os.remove(pdf_path)

# ---------------------------------
# 6. Ask Questions
# ---------------------------------

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader("💬 Chat with your PDF")

    st.caption(f"Document: {st.session_state.file_name}")

    query = st.chat_input("Ask a question about your PDF...")

    if query:

        with st.spinner("Searching your PDF..."):

            # Similarity Search
            results = st.session_state.vector_store.similarity_search(
                query,
                k=3
            )

            # Retrieve Context
            context = "\n\n".join(
                doc.page_content for doc in results
            )

            # RAG Prompt
            prompt = f"""
            You are a helpful AI assistant.

            Answer the question using only the following context.

            If the answer is not available in the context, say:
            "I could not find this information in the document."

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

            # Generate Answer using Gemini
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            answer = response.text

            # Save Chat History
            st.session_state.chat_history.append(
                {
                    "question": query,
                    "answer": answer
                }
            )

    # ---------------------------------
    # 7. Display Chat History
    # ---------------------------------

    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

else:
    st.info("Please upload and process a PDF to start chatting.")
