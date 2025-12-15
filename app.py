import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.chains import create_retrieval_chain
from langchain_community.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    layout="wide"
)

st.title("📄 Multi-PDF RAG Chatbot (Groq API)")
st.caption("Retrieval-Augmented Generation using LangChain & Streamlit")

# --------------------------------------------------
# Load Groq API Key (Cloud-safe)
# --------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()

# --------------------------------------------------
# Sidebar: PDF Upload
# --------------------------------------------------
with st.sidebar:
    st.header("📂 Upload PDF Documents")
    pdf_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

# --------------------------------------------------
# Helper Functions (Paper Pipeline)
# --------------------------------------------------
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if pdf_files:
    with st.spinner("🔍 Processing documents..."):
        raw_text = extract_text_from_pdfs(pdf_files)
        text_chunks = chunk_text(raw_text)
        vectorstore = create_vector_store(text_chunks)

    st.success("✅ Documents processed successfully")

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        api_key=GROQ_API_KEY
    )

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the context below.
    Context:
    {context}

    Question:
    {input}
    """
)

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain
)
    )

    query = st.text_input("❓ Ask a question based on the uploaded PDFs")

    if query:
        with st.spinner("🤖 Generating response..."):
            result = qa_chain(query)

        st.subheader("🧠 Answer")
        st.write(result["result"])

        with st.expander("📌 Retrieved Source Chunks"):
            for doc in result["source_documents"]:
                st.write(doc.page_content)

else:
    st.info("⬅ Upload PDFs from the sidebar to begin")
