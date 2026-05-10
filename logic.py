import os
# Silences the transformers library warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain.schema import Document
from langchain_openai import  ChatOpenAI

load_dotenv()

def extract_video_id(url_or_id):
    pattern = r"(?:v=|\/|be\/)([0-9A-Za-z_-]{11})(?:[&?]|$)"
    match = re.search(pattern, url_or_id)
    return match.group(1) if match else url_or_id

def process_video_to_vectorstore(video_url_or_id):
    '''Fetches the transcript, chunks it, and stores it in a vector store.'''
    video_id = extract_video_id(video_url_or_id)

    try:
       
    #    raw_transcript=YouTubeTranscriptApi.get_transcript(video_id,languages=['en','hi'])
       ytt_api = YouTubeTranscriptApi()
       raw_transcript = ytt_api.fetch(video_id, languages=["en","hi"])
       full_text = ""
       for item in raw_transcript:
            if isinstance(item, dict):
                full_text += item['text'] + " "
            else:
                # This handles the new 'FetchedTranscriptSnippet' object
                full_text += item.text + " "
        
       if not full_text.strip():
            raise Exception("Transcript content is empty.")
       docs = [Document(page_content=full_text, metadata={"source": video_id})]
    except Exception as e:
        raise Exception(f"Could not retrieve transcript: {str(e)}")
    
    # 2. Chunking 
    text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
    )

    chunks=text_splitter.split_documents(docs)

     # 3. create Vector stores
    embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore= FAISS.from_documents(chunks, embedding)

    return vectorstore
   


def query_vectorstores(vectorstore, query):
    '''Retrieves relevant chunks and generates answer'''

    #model=ChatGoogleGenerativeAI(model='gemini-2.5-pro')

    # We use the standard ChatOpenAI but redirect the base_url
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini", # Or "anthropic/claude-3.5-sonnet", etc.
        openai_api_base="https://openrouter.ai/api/v1", # The magic line
        openai_api_key=os.getenv("openrouter_key"), # Your OpenRouter API key from .env
        default_headers={
            "HTTP-Referer": "http://localhost:8501", # Optional: for OpenRouter rankings
            "X-Title": "YouTube Transcript Chatbot",
        },
        temperature=0
    )

    # Define how to handle the retrieved context
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context from a video transcript to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "\n\n"
        "{context}"
    )

    prompt=ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', "{input}"),
    ])

    # create a chain to combine the retrieved documents and generate an answer
    chain=create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    # Retrieve relevant chunks from the vector store
    retrieval_chain=create_retrieval_chain(
        retriever=vectorstore.as_retriever(search_type='similarity',search_kwargs={"k":4}),
        combine_docs_chain=chain
    )

    response=retrieval_chain.invoke({"input": query})
    return response['answer']