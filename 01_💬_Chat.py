import ollama
import streamlit as st
from openai import OpenAI
from utilities.icon import page_icon
import os
import tempfile
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from typing import List, Tuple

st.set_page_config(
    page_title="Chat playground",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def extract_model_names(models_info: list) -> tuple:
    """
    Extracts the model names from the models information.

    :param models_info: A dictionary containing the models' information.

    Return:
        A tuple containing the model names.
    """

    return tuple(model["model"] for model in models_info["models"])


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from uploaded PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name

    text = ""
    pdf_reader = PdfReader(temp_path)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    # Clean up temp file
    os.unlink(temp_path)
    return text


def chunk_text(text: str) -> List[str]:
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    return text_splitter.split_text(text)


def get_embeddings(client, text_chunks: List[str], model: str) -> np.ndarray:
    """Get embeddings for text chunks using Ollama API"""
    embeddings = []
    for chunk in text_chunks:
        response = client.embeddings.create(model=model, input=chunk)
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings, dtype=np.float32)


def create_vector_store(embeddings: np.ndarray) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """Create a FAISS vector store from embeddings"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def get_relevant_context(
    query: str, client, model: str, index, chunks: List[str], top_k: int = 3
) -> str:
    """Retrieve relevant context based on query"""
    query_embedding = (
        client.embeddings.create(model=model, input=query).data[0].embedding
    )

    query_embedding_array = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_embedding_array, top_k)

    context = "\n\n".join([chunks[i] for i in indices[0]])
    return context


def main():
    """
    The main function that runs the application.
    """

    page_icon("💬")
    st.subheader("Ollama Playground", divider="red", anchor=False)

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )
    else:
        st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
        if st.button("Go to settings to download a model"):
            st.page_switch("pages/03_⚙️_Settings.py")
        return

    # Add PDF Upload Section
    with st.expander("📄 Upload PDF for RAG"):
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                # Extract text from PDF
                text = extract_text_from_pdf(uploaded_file)

                # Chunk the text
                text_chunks = chunk_text(text)

                if not text_chunks:
                    st.error("Could not extract text from the PDF")
                else:
                    # Get embeddings and create vector store
                    embeddings = get_embeddings(client, text_chunks, selected_model)
                    index = create_vector_store(embeddings)

                    # Store in session state
                    st.session_state.rag_enabled = True
                    st.session_state.vector_index = index
                    st.session_state.text_chunks = text_chunks

                    st.success(
                        f"PDF processed successfully! {len(text_chunks)} chunks extracted."
                    )

    message_container = st.container(height=500, border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False

    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a prompt here..."):
        print(f"user message: {prompt}")
        try:
            st.session_state.messages.append({"role": "user", "content": prompt})

            message_container.chat_message("user", avatar="😎").markdown(prompt)
            print(f"selected_model= {selected_model}")

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("model working..."):
                    # If RAG is enabled, get relevant context
                    messages_for_model = []

                    if st.session_state.rag_enabled:
                        context = get_relevant_context(
                            prompt,
                            client,
                            selected_model,
                            st.session_state.vector_index,
                            st.session_state.text_chunks,
                        )

                        # Add system message with context
                        messages_for_model.append(
                            {
                                "role": "system",
                                "content": f"Answer the question based on this context: {context}",
                            }
                        )

                    # Add conversation history
                    messages_for_model.extend(
                        [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                    )

                    print(messages_for_model)
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=messages_for_model,
                        stream=True,
                    )
                # stream response
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(e, icon="⛔️")


if __name__ == "__main__":
    main()
