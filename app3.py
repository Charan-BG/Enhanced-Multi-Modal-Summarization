import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, PyPDFLoader, TextLoader
import re
import tempfile
import time
import os

# Streamlit App Configuration
st.set_page_config(page_title="Enhanced LangChain Summarizer", page_icon="🦜")
st.title("🦜 Enhanced LangChain: Multi-Modal Summarizer")
st.subheader("Summarize YouTube Videos, Websites, PDFs, or Text Files")

# Sidebar for API Key and Customization
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    # Model Selection
    model_options = {
        "Gemma-7b-It": "A general-purpose 7B model.",
        "gemma2-9b-it": "More in-depth summarization.",
        "llama3-8b-8192": "Llama-based open-source model.",
        "llama-3.1-70b-versatile": "A larger model optimized for complex tasks."
    }
    selected_model = st.selectbox("Select Groq Model", list(model_options.keys()))
    st.write(f"**Model Info:** {model_options[selected_model]}")
    summary_length = st.selectbox("Select Summary Length", ["Short", "Medium", "Long"])
    focus_area = st.selectbox("Select Focus Area", ["Quick Overview", "Critical Insights", "Actionable Steps", "Storytelling", "Sentiment Analysis", "In-Depth Analysis"])

# Input URL or File
generic_url = st.text_input("URL (YouTube or Website)", label_visibility="visible")
uploaded_files = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])
###"""""""

# uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
# if uploaded_files:
#     documents=[]
#     for uploaded_file in uploaded_files:
#         temppdf=f"./temp.pdf"
#         with open(temppdf,"wb") as file:
#             file.write(uploaded_file.getvalue())
#             file_name=uploaded_file.name

#         loader=PyPDFLoader(temppdf)
#         docs=loader.load()
#         documents.extend(docs)
###"""""""
def get_summary_length_choice(length):
    """Map user choice to word count for summary."""
    if length == "Short":
        return "100 words"
    elif length == "Medium":
        return "400 words"
    else:
        return "700 words"

def extract_keywords(text):
    """Extract keywords using a simple regex-based method."""
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in set(words) if len(word) > 4]
    return ', '.join(sorted(keywords[:10]))



##########################
if uploaded_files:
    st.write(f"File '{uploaded_files.name}' successfully uploaded.")
else:
    st.error("No file uploaded.")
###########################

# Initialize LLM
llm = ChatGroq(model=selected_model, groq_api_key=groq_api_key)

# Prompt Template for Summarization
final_prompt_template = """
You are an expert summarizer. Read the provided content and generate a concise summary based on user preferences:
- **Summary Length**: {length}
- **Focus Area**: {focus}

Please ensure the summary:
1. Captures the most important information.
2. Highlights key points relevant to the focus area.
3. Provides clear and concise insights.

Content:
{text}
"""

chunks_prompt_template = """
Extract the most relevant and meaningful content from the following text. Focus on providing:
1. Key insights and core information.
2. Details that align with user preferences and context.
3. A coherent and concise response that can be used in the final summary.

Text:
{text}
"""

map_prompt = PromptTemplate(template=chunks_prompt_template, input_variables=["text"])
final_prompt = PromptTemplate(template=final_prompt_template, input_variables=["text", "length", "focus"])


def load_and_process_pdf(uploaded_files):
    """Efficiently load and process PDF documents."""
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
    #     temp_pdf.write(uploaded_files.read())
    #     temp_pdf_path = temp_pdf.name

    # loader = PyPDFLoader(temp_pdf_path)


    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    return documents


    # st.write(f"Attempting to load PDF from: {temp_pdf_path}")
    
    # try:
    #     docs = loader.load()
    #     if not docs:
    #         raise ValueError("No content extracted from the PDF.")
        
    #     for i, doc in enumerate(docs):
    #         st.write(f"Document {i+1} content preview: {doc.page_content[:500]}")  # Show a preview of document text

        # return docs
    # except Exception as e:
    #     raise Exception(f"Error loading PDF: {e}")
    # finally:
    #     # Clean up the temporary file
    #     if os.path.exists(temp_pdf_path):
    #         os.remove(temp_pdf_path)


def summarize_document(docs, llm, summary_length_choice, focus_area):
    """Summarize document with LLM."""
    final_documents = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300).split_documents(docs)

    if not final_documents:
        st.error("No chunks were created after splitting the documents.")
    else:
        st.write(f"Created {len(final_documents)} chunks.")
        st.write(f"First chunk preview: {final_documents[0].page_content[:500]}")

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=final_prompt,
        verbose=True
    )

    output_summary = chain.run(input_documents=final_documents, length=summary_length_choice, focus=focus_area)
    return output_summary

# Button to Summarize Content
if st.button("Summarize Content"):
    if not groq_api_key.strip() or (not generic_url.strip() and not uploaded_files):
        st.error("Please provide the necessary information to get started.")
    else:
        try:
            with st.spinner("Processing..."):
                start_time = time.time()

                docs = []

                # Load content based on input type
                if generic_url:
                    st.subheader("URL Content Summary")

                    if "youtube.com" in generic_url:
                        try:
                            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                            docs = loader.load()
                        except Exception as e:
                            st.error(f"Failed to load YouTube video. Error: {e}")
                    else:
                        loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                       headers={"User-Agent": "Mozilla/5.0"})
                        docs = loader.load()

                    # Prepare summary customization options
                    summary_length_choice = get_summary_length_choice(summary_length)
                    output_summary = ""

                    final_documents = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500).split_documents(docs)
                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type="map_reduce",
                        map_prompt=map_prompt,
                        combine_prompt=final_prompt,
                        verbose=True
                    )
                    output_summary = chain.run(input_documents=final_documents, length=summary_length_choice, focus=focus_area)

                    # Extract keywords
                    keywords = extract_keywords(output_summary)

                    # Display Summary and Insights
                    st.success(output_summary)
                    st.markdown("### 📋 Keywords")
                    st.write(keywords)

                    # Analytics Dashboard
                    st.markdown("### 📊 Content Analytics")
                    word_count = len(output_summary.split())
                    st.write(f"**Word Count:** {word_count}")
                    st.write(f"**Summary Length:** {summary_length_choice}")

                # If document is uploaded (PDF/Text)
                if uploaded_files:
                    st.subheader("Document Content Summary")

                    docs = load_and_process_pdf(uploaded_files)
                    summary_length_choice = get_summary_length_choice(summary_length)
                    output_summary = summarize_document(docs, llm, summary_length_choice, focus_area)

                    # Extract keywords and display results
                    keywords = extract_keywords(output_summary)
                    st.success(output_summary)
                    st.markdown("### 📋 Keywords")
                    st.write(keywords)

                    # Analytics Dashboard
                    st.markdown("### 📊 Content Analytics")
                    word_count = len(output_summary.split())
                    st.write(f"**Word Count:** {word_count}")
                    st.write(f"**Summary Length:** {summary_length_choice}")

                processing_time = time.time() - start_time
                st.write(f"**Processing Time:** {processing_time:.2f} seconds")

        except Exception as e:
            st.exception(f"Exception: {e}")