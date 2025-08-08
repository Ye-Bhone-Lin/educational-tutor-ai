from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq 
from dotenv import load_dotenv 
from langchain.chains import RetrievalQA 
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool, AgentType
import os


def load_document(file_path, file_type):
    """
    Load document based on file type
    """
    if file_type.lower() == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type.lower() == "pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please use 'pdf' or 'pptx'")
    
    return loader.load()


def setup_vector_store(documents):
    """
    Setup vector store with documents
    """
    # Split documents
    chunk = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=False)
    split_doc = chunk.split_documents(documents)
    
    # Setup Qdrant client
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="zoomcamp-project1",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Setup embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Setup vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="zoomcamp-project1",
        embedding=embedding_model
    )
    
    # Add documents to vector store
    vector_store.add_documents(split_doc)
    
    return vector_store


def create_agent(vector_store):
    """
    Create the agent with tools
    """
    # Setup LLM
    model = ChatGroq(model_name="llama-3.3-70b-versatile")
    
    # Setup retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 1}
    )
    
    # Setup retrieval QA chain
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever, 
        return_source_documents=True
    )
    
    # Setup DuckDuckGo search tool
    def ddg_search(query: str):
        ddg = DuckDuckGoSearchResults()
        return ddg.invoke(query)
    
    website_tool = Tool(
        name="Website Tool",
        func=ddg_search,
        description="Help the STUDENTS to know better life time"
    )
    
    # Setup tools
    tools = [
        Tool(
            name="Document Retrieval",
            func=lambda q: retrieval_qa_chain.invoke({"query": q})["result"],
            description="Retrieve knowledge from the document"
        ),
        website_tool
    ]
    
    # Initialize agent
    agent = initialize_agent(
        tools=tools, 
        llm=model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
    
    return agent
