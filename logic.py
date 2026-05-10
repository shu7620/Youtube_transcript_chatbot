from langchain_community.loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain

load_dotenv()


def process_video_to_vectorstore(video_url):
    '''Fetches the transcript, chunks it, and stores it in a vector store.'''
    try:
        # 1. Load Transcript
        loader=YoutubeLoader.from_youtube_url(video_url,add_video_info=True)
        transcript=loader.load()
        
        # 2. Chunking 
        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks=text_splitter.split_documents(transcript)

        # 3. create Vector stores
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore= FAISS.from_documents(chunks, embedding)

        return vectorstore
    except Exception as e:
        raise Exception(f"Could not retrieve transcript: {str(e)}")


def query_vectorstores(vectorstore, query):
    '''Retrieves relevant chunks and generates answer'''

    model=ChatGoogleGenerativeAI(model='gemini-2.5-pro')

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
        ('human', {input}),
    ])

    # create a chain to combine the retrieved documents and generate an answer
    chain=create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    # Retrieve relevant chunks from the vector store
    retrieval_chain=create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_documents_chain=chain
    )

    response=retrieval_chain.invoke({"input": query})
    return response['answer']