# 🦜 Enhanced LangChain: Multi-Modal Summarizer

This project is a powerful, **multi-modal summarization tool** leveraging LangChain, Groq API, and Streamlit. It supports summarizing content from various sources, including **YouTube videos, websites, PDFs, and text files**, using advanced LLMs like Gemma, and Llama models.

## 🚀 Features

- **Multi-Modal Summarization**:
  - Summarize content from YouTube videos, websites, PDFs, and plain text files.
- **Customizable Summaries**:
  - Choose summary length: Short, Medium, Long.
  - Select focus area: Quick Overview, Critical Insights, Actionable Steps, Storytelling, Sentiment Analysis, In-Depth Analysis.
- **Model Flexibility**:
  - Integrate with various LLM models through the Groq API.
  - Supports models like Gemma-7b-It, Gemma2-9b-it, Llama3-8b-8192, and Llama-3.1-70b.
- **Analytics and Insights**:
  - Keyword extraction and content analytics (word count, summary length).

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Charan-BG/enhanced-langchain-summarizer.git
   cd enhanced-langchain-summarizer
   
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt 
   
3. Run the Streamlit app:
   ```bash
   streamlit run app.py

## 🔑 API Key Setup
You need a Groq API key to access the models. Enter your API key in the Streamlit app sidebar.

## 📦 Requirements
- Python 3.8+
- streamlit
- langchain
- langchain_groq
- langchain_community
- PyPDF2 (for PDF handling)
go through requriments for more..

## ⚙️ Usage
1. Set API Key: Enter your Groq API Key in the sidebar.
2. Select a Model: Choose from multiple LLM options provided.
3. Input Content:
   - Enter a URL (YouTube or website) or upload a PDF/text file.
4. Customize Summary:
   - Select summary length and focus area.
5.Click "Summarize Content": View the generated summary, keywords, and content analytics.


## 📚 Supported Models

| Model             | Description                                      |
|-------------------|--------------------------------------------------|
| **Gemma-7b-It**   | General-purpose model for fast summarization.    |
| **Gemma2-9b-it**  | Deeper summarization with higher accuracy.       |
| **Llama3-8b-8192**| Open-source Llama-based model.                   |
| **Llama-3.1-70b** | Advanced model for complex summarization tasks.  |


## 🛡️ Error Handling
YouTube Loading Error: If a YouTube video fails to load, ensure the URL is correct or try another video.
PDF Handling Error: If PDF loading fails, check the file format or try another document.

## 🤖 Future Enhancements
Add support for audio-based insights using ASR (Automatic Speech Recognition).
Extend to include more document formats like Word (.docx).
Provide detailed topic distribution analysis.

## 📝 License
This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for details.

💬 Feedback
If you encounter any issues or have suggestions, please open an issue on the GitHub repository or reach out via email.



