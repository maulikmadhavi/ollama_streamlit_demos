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
import requests
from duckduckgo_search import DDGS

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


def perform_internet_search(query: str, num_results: int = 3) -> str:
    """Perform internet search using DuckDuckGo and return formatted results"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return "No search results found."

        formatted_results = "### Internet Search Results:\n\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No content")
            href = result.get("href", "No link")
            formatted_results += f"{i}. **{title}**\n{body}\n[Source]({href})\n\n"

        return formatted_results
    except Exception as e:
        return f"Error performing internet search: {str(e)}"


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

    # Main settings section with tabs for different features
    st.write("### Model Enhancement Features")
    tabs = st.tabs(["📄 PDF Context (RAG)", "🌐 Internet Search", "⚙️ Advanced"])

    # PDF Upload Section in first tab
    with tabs[0]:
        st.caption("Upload a PDF document to provide context for your questions.")
        uploaded_file = st.file_uploader(
            "Upload a PDF document", type="pdf", key="pdf_uploader"
        )
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

    # Internet Search Options in second tab
    with tabs[1]:
        st.caption("Enable internet search to get real-time information from the web.")

        col1, col2 = st.columns([1, 1])

        with col1:
            internet_search_enabled = st.toggle(
                "Enable Internet Search",
                value=st.session_state.get("internet_search_enabled", False),
                help="When enabled, the model will search the internet for relevant information before responding",
            )

        with col2:
            num_search_results = st.slider(
                "Number of search results",
                min_value=1,
                max_value=10,
                value=st.session_state.get("num_search_results", 3),
                disabled=not internet_search_enabled,
                help="How many search results to retrieve from the internet",
            )

        if internet_search_enabled:
            st.session_state.internet_search_enabled = True
            st.session_state.num_search_results = num_search_results
            st.info(
                "🌐 Internet search is now ENABLED. Your queries will be sent to DuckDuckGo to retrieve relevant information."
            )
        else:
            st.session_state.internet_search_enabled = False
            st.warning(
                "🔍 Internet search is DISABLED. The model will rely only on its training data."
            )

    # Advanced options in third tab
    with tabs[2]:
        st.caption("Additional configuration options.")
        st.text("No advanced options available yet.")

    # Display status of enabled features
    feature_status = []
    if st.session_state.get("rag_enabled", False):
        feature_status.append("📄 PDF Context")
    if st.session_state.get("internet_search_enabled", False):
        feature_status.append("🌐 Internet Search")

    if feature_status:
        st.success(f"Active features: {', '.join(feature_status)}")

    message_container = st.container(height=500, border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False

    if "internet_search_enabled" not in st.session_state:
        st.session_state.internet_search_enabled = False

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
                    context_parts = []

                    if st.session_state.rag_enabled:
                        rag_context = get_relevant_context(
                            prompt,
                            client,
                            selected_model,
                            st.session_state.vector_index,
                            st.session_state.text_chunks,
                        )
                        context_parts.append(f"PDF context: {rag_context}")

                    # If internet search is enabled, get search results
                    if st.session_state.internet_search_enabled:
                        with st.spinner("Searching the internet..."):
                            search_results = perform_internet_search(
                                prompt, st.session_state.num_search_results
                            )
                            context_parts.append(
                                f"Internet search results: {search_results}"
                            )

                    # Add system message with combined context if any
                    if context_parts:
                        combined_context = "\n\n".join(context_parts)
                        messages_for_model.append(
                            {
                                "role": "system",
                                "content": f"Answer the question based on this information: {combined_context}",
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
