import streamlit as st
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
import tempfile
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def load_document(uploaded_file, file_type):
    """
    Load document based on file type
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load document based on type
        if file_type == "pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_type == "pptx":
            loader = UnstructuredPowerPointLoader(tmp_file_path)
        else:
            raise ValueError("Unsupported file type")
        
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def setup_vector_store(documents):
    """
    Setup vector store with documents
    """
    try:
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
    except Exception as e:
        st.error(f"Error setting up vector store: {str(e)}")
        return None

def create_agent(vector_store):
    """
    Create the agent with tools
    """
    try:
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
                description="Retrieve knowledge from the slides and pdf file"
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
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

def main():
    """
    Main Streamlit app
    """
    # Header
    st.markdown('<h1 class="main-header">📚 Document AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown('<h3 class="sub-header">📁 Upload Document</h3>', unsafe_allow_html=True)
        
        # File type selection
        file_type = st.selectbox(
            "Select document type:",
            ["pdf", "pptx"],
            help="Choose the type of document you want to upload"
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            f"Upload your {file_type.upper()} file",
            type=[file_type],
            help=f"Upload a {file_type.upper()} file to analyze"
        )
        
        # Info box
        st.markdown("""
        <div class="info-box">
            <h4>ℹ️ How it works:</h4>
            <ul>
                <li>Upload your PDF or PPTX file</li>
                <li>The system will process and index your document</li>
                <li>Ask questions about your document content</li>
                <li>Get AI-powered answers with web search capability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if uploaded_file is not None:
        # Display file info
        st.markdown(f"### 📄 File: {uploaded_file.name}")
        st.markdown(f"**Size:** {uploaded_file.size} bytes")
        
        # Process button
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing your document..."):
                # Step 1: Load document
                st.info("📖 Loading document...")
                documents = load_document(uploaded_file, file_type)
                
                if documents:
                    st.success(f"✅ Successfully loaded {len(documents)} document(s)")
                    
                    # Step 2: Setup vector store
                    st.info("🔍 Setting up vector store...")
                    vector_store = setup_vector_store(documents)
                    
                    if vector_store:
                        st.success("✅ Vector store setup complete!")
                        
                        # Step 3: Create agent
                        st.info("🤖 Creating AI agent...")
                        agent = create_agent(vector_store)
                        
                        if agent:
                            st.success("✅ Agent created successfully!")
                            
                            # Store agent in session state
                            st.session_state.agent = agent
                            st.session_state.document_processed = True
                            
                            st.markdown("""
                            <div class="success-box">
                                <h4>🎉 Document processed successfully!</h4>
                                <p>You can now ask questions about your document below.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Failed to create agent")
                    else:
                        st.error("Failed to setup vector store")
                else:
                    st.error("Failed to load document")
    
    # Chat interface
    if st.session_state.get('document_processed', False):
        st.markdown("---")
        st.markdown('<h3 class="sub-header">💬 Ask Questions</h3>', unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.agent.invoke(prompt)
                        answer = response['output']
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        st.markdown(answer)
                    except Exception as e:
                        error_msg = f"Error processing your question: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Author - Ye Bhone Lin
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
