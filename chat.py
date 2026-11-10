
import streamlit as st
import os
import tempfile

from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# 2. LOAD GEMINI API KEY
# ==========================================

load_dotenv()

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Gemini API Key not found! Please configure Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)

GEMINI_MODEL = "gemini-3.5-flash-lite"


# ==========================================
# 3. PAGE TITLE
# ==========================================

st.title("📚 AI PDF Assistant")
st.write(
    "Upload your PDF document and ask questions. "
    "Gemini AI will answer using your document."
)

st.divider()


# ==========================================
# 4. SESSION STATE
# ==========================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ==========================================
# 5. PDF UPLOAD
# ==========================================

st.subheader("📄 Upload PDF")

uploaded_file = st.file_uploader(
    "Choose your PDF document",
    type=["pdf"]
)


# ==========================================
# 6. PROCESS PDF
# ==========================================

if uploaded_file is not None:

    if st.button("🚀 Process PDF", type="primary"):

        pdf_path = None

        try:

            with st.spinner("Processing your PDF..."):

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

                if not documents:
                    st.error("No pages found in this PDF.")
                    st.stop()

                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                texts = text_splitter.split_documents(documents)

                if not texts:
                    st.error("No readable text found in this PDF.")
                    st.stop()

                # Create Gemini embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=API_KEY
                )

                # Create ChromaDB vector store
                vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    collection_name="pdf_docs"
                )

                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.chat_history = []
                st.session_state.file_name = uploaded_file.name

            st.success("✅ PDF processed successfully!")

            st.write("Total Pages:", len(documents))
            st.write("Total Chunks:", len(texts))

        except Exception as e:
            st.error(f"PDF Processing Error: {e}")

        finally:

            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)


# ==========================================
# 7. CHAT WITH PDF
# ==========================================

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader("💬 Chat with your PDF")

    st.caption(
        f"Currently loaded document: {st.session_state.file_name}"
    )

    # Display previous chat messages
    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # User question
    query = st.chat_input("Ask anything about your PDF...")

    if query:

        with st.chat_message("user"):
            st.write(query)

        try:

            with st.chat_message("assistant"):

                with st.spinner("Gemini is thinking..."):

                    # Similarity search
                    results = st.session_state.vector_store.similarity_search(
                        query,
                        k=3
                    )

                    # Retrieve context
                    context = "\n\n".join(
                        doc.page_content for doc in results
                    )

                    # RAG prompt
                    prompt = f"""
                    You are a helpful AI assistant.

                    Answer the question using only the
                    following PDF context.

                    If the answer is not available in the
                    document, say:
                    "I could not find this information
                    in the document."

                    Context:
                    {context}

                    Question:
                    {query}

                    Answer:
                    """

                    # Generate Gemini response
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt
                    )

                    answer = response.text

                    if not answer:
                        answer = "Gemini did not return an answer."

                    st.write(answer)

                    # Save chat history
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer
                    })

        except Exception as e:
            st.error(f"Chat Error: {e}")


# ==========================================
# 8. RESET CHAT
# ==========================================

if st.session_state.vector_store is not None:

    if st.button("🗑️ Clear Chat History"):

        st.session_state.chat_history = []
        st.rerun()


else:

    st.info("👆 Upload a PDF and click Process PDF to start.")
