# 🚀 Ollama x Streamlit Playground

This project demonstrates how to run and manage models locally using [Ollama](https://ollama.com/) by creating an interactive UI with [Streamlit](https://streamlit.io).

The app has a page for running chat-based models and also one for nultimodal models (_llava and bakllava_) for vision.

## App in Action

![GIF](assets/ollama_streamlit.gif)

![alt text](assets/new_RAG_pdf_support.png)
**Check out the video tutorial 👇**

<a href="https://youtu.be/bAI_jWsLhFM">
  <img src="https://img.youtube.com/vi/bAI_jWsLhFM/hqdefault.jpg" alt="Watch the video" width="100%">
</a>

## Features
- **RAG Support with custom PDFs**: Run your RAG models with custom PDFs.
- **Internet Search Integration**: Enhance responses with real-time information from DuckDuckGo (disabled by default).
- **Interactive UI**: Utilize Streamlit to create a user-friendly interface.
- **Local Model Execution**: Run your Ollama models locally without the need for external APIs.
- **Real-time Responses**: Get real-time responses from your models directly in the UI.

## Installation

Before running the app, ensure you have Python installed on your machine. Then, clone this repository and install the required packages using pip:

```bash
git clone https://github.com/tonykipkemboi/ollama_streamlit_demos.git
```

```bash
cd ollama_streamlit_demos
```

```bash
pip install -r requirements.txt
```

## Usage

To start the app, run the following command in your terminal:

```bash
streamlit run 01_💬_Chat_Demo.py
```

Navigate to the URL provided by Streamlit in your browser to interact with the app.

### Using Internet Search

1. After launching the app, navigate to the "🌐 Internet Search" tab
2. Toggle the "Enable Internet Search" switch to ON
3. Adjust the number of search results if needed
4. Now when you ask questions, the model will search the internet for the most up-to-date information

**Note:** Internet search is disabled by default to respect your privacy and to limit API calls.

**NB: Make sure you have downloaded [Ollama](https://ollama.com/) to your system.**

## Contributing

Interested in contributing to this app?

- Great!
- I welcome contributions from everyone.

Got questions or suggestions?

- Feel free to open an issue or submit a pull request.

## Acknowledgments

👏 Kudos to the [Ollama](https://ollama.com/) team for their efforts in making open-source models more accessible!
