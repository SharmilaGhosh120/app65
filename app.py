import streamlit as st
import numpy as np
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
import torch
import re
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="RAG Agentic AI Assistant",
    page_icon=":robot_face:",
    layout="centered"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        font-family: 'Roboto', sans-serif;
    }
    .stTextInput, .stTextArea {
        border: 1px solid #ccc;
        border-radius: 5px;
        font-family: Arial, sans-serif;
    }
    .submit-button {
        display: flex;
        justify-content: center;
    }
    .submit-button .stButton {
        background-color: #4fb8ac;
        color: white;
        font-size: 18px;
        margin: 10px;
        padding: 10px;
        border-radius: 4px;
        width: 200px;
    }
    .history-entry {
        padding: 15px;
        border: 1px solid #ccc;
        border-radius: 12px;
        margin-bottom: 10px;
        background-color: #fff;
        box-shadow: 1px 1px 3px #ccc;
        font-family: Arial, sans-serif;
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 12px;
        background-color: #f9f9f9;
    }
    .footer {
        text-align: center;
        font-family: Arial, sans-serif;
        color: #4fb8ac;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sample knowledge base
documents = [
    "In 2025, AI advancements include multimodal models integrating text, images, and audio for enhanced reasoning.",
    "Agentic AI systems in 2025 can autonomously refine queries and evaluate responses for accuracy.",
    "Large language models like GPT-4 and BART have improved context understanding but still face hallucination issues.",
    "Retrieval-Augmented Generation (RAG) combines retrieval and generation to provide factually accurate responses.",
    "Recent AI research focuses on scalable vector search using FAISS for efficient document retrieval."
]

# Initialize models
@st.cache_resource
def load_models():
    try:
        retriever_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        generator_model_name = "facebook/bart-large"
        
        # Set Hugging Face token for authentication if available
        if HF_TOKEN:
            from huggingface_hub import login
            login(token=HF_TOKEN)
        
        retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
        retriever_model = AutoModel.from_pretrained(retriever_model_name)
        generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
        
        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        retriever_model.to(device)
        generator_model.to(device)
        
        # Initialize FAISS index
        dimension = 384  # Dimension of sentence transformer embeddings
        index = faiss.IndexFlatL2(dimension)
        
        # Encode and index documents
        embeddings = encode_documents(documents)
        index.add(embeddings)
        
        return retriever_tokenizer, retriever_model, generator_tokenizer, generator_model, index, device
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise Exception("Model initialization failed. Please check dependencies and try again.")

def encode_documents(docs):
    try:
        embeddings = []
        for doc in docs:
            inputs = retriever_tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                embedding = retriever_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(embedding)
        return np.vstack(embeddings)
    except Exception as e:
        logger.error(f"Error encoding documents: {str(e)}")
        raise

# Agentic AI: Query Refinement
def refine_query(query):
    try:
        if len(query.split()) < 3 or "?" not in query:
            query = query + " in 2025"
        if "AI" in query and "recent" not in query:
            query = "recent " + query.lower()
        return query
    except Exception as e:
        logger.error(f"Error refining query: {str(e)}")
        return query

# Agentic AI: Response Evaluation
def evaluate_response(response, query, retrieved_docs):
    try:
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        doc_words = set(" ".join(retrieved_docs).lower().split())
        overlap_score = len(query_words.intersection(response_words)) / len(query_words)
        doc_relevance = len(response_words.intersection(doc_words)) / len(response_words)
        return overlap_score > 0.5 and doc_relevance > 0.3
    except Exception as e:
        logger.error(f"Error evaluating response: {str(e)}")
        return False

# Retrieve top-k documents
def retrieve_documents(query, k=3):
    try:
        query = refine_query(query)
        inputs = retriever_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            query_embedding = retriever_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        distances, indices = index.search(query_embedding, k)
        retrieved_docs = [documents[i] for i in indices[0]]
        return retrieved_docs, query
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise

# Generate response using LLM
def generate_response(query, retrieved_docs):
    try:
        context = " ".join(retrieved_docs)
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        inputs = generator_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = generator_model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
        response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, retrieved_docs
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

# Main RAG pipeline with agentic AI
def rag_pipeline(query, max_iterations=2):
    conversation_history = []
    for _ in range(max_iterations):
        try:
            retrieved_docs, refined_query = retrieve_documents(query)
            response, sources = generate_response(refined_query, retrieved_docs)
            if evaluate_response(response, refined_query, retrieved_docs):
                conversation_history.append({"query": query, "response": response, "sources": sources})
                return response, sources, conversation_history
            query = refine_query(query + " provide more details")
        except Exception as e:
            logger.error(f"Error in RAG pipeline iteration: {str(e)}")
            continue
    return response, sources, conversation_history

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load models and FAISS index
try:
    retriever_tokenizer, retriever_model, generator_tokenizer, generator_model, index, device = load_models()
except Exception as e:
    st.error("Failed to initialize models. Please check your setup and try again.")
    logger.error(f"Model loading error: {str(e)}")
    st.stop()

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4fb8ac; font-family: \"Roboto\", sans-serif;'>RAG Agentic AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-family: \"Roboto\", sans-serif;'>Ask about recent AI advancements!</p>", unsafe_allow_html=True)
st.markdown("---")

# Query Input
st.subheader("Ask a Question")
sample_questions = [
    "What are recent advancements in AI?",
    "How does RAG improve AI responses?",
    "What are agentic AI systems?"
]
selected_question = st.selectbox("Choose a sample question or type your own:", sample_questions + ["Custom question"])
query_text = st.text_area("Your Question", value=selected_question if selected_question != "Custom question" else "", height=100, placeholder="E.g., What are recent advancements in AI?")

# Submit Query
st.markdown('<div class="submit-button">', unsafe_allow_html=True)
if st.button("Submit", type="primary"):
    if not query_text:
        st.error("Please enter a question.")
    else:
        with st.spinner("Processing your query..."):
            try:
                response, sources, history = rag_pipeline(query_text)
                st.session_state.chat_history.append({
                    "query": query_text,
                    "response": response,
                    "sources": sources,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Query processed successfully!")
                with st.expander("🧠 Assistant’s Response:", expanded=True):
                    response_text = response[:500] + "..." if len(response) > 500 else response
                    st.markdown(
                        f"""
                        <div class='history-entry'>
                            <strong>Response:</strong> <br>{response_text}</br><br>
                            <strong>Sources:</strong><br>
                            {"<br>".join([f"{i+1}. {source}" for i, source in enumerate(sources)])}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error("Failed to process query. Please try again.")
st.markdown('</div>', unsafe_allow_html=True)

# Query History
st.subheader("Query History")
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for entry in reversed(st.session_state.chat_history):
        response_text = entry['response'][:500] + "..." if len(entry['response']) > 500 else entry['response']
        with st.expander(f"Question at {entry['timestamp']}"):
            st.markdown(
                f"""
                <div class='history-entry'>
                    <strong>You asked:</strong> {entry['query']}<br>
                    <strong>Response:</strong> {response_text}<br>
                    <strong>Sources:</strong><br>
                    {"<br>".join([f"{i+1}. {source}" for i, source in enumerate(entry['sources'])])}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No query history yet.")

# Footer
st.markdown(
    "<p class='footer'>Powered by RAG and Agentic AI. Ask freely!</p>",
    unsafe_allow_html=True
)