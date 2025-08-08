from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq 
from dotenv import load_dotenv 
from langchain.chains import RetrievalQA 
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool, AgentType


load_dotenv()

pdf_loader = PyPDFLoader("Machine_Translation_Approaches_and_Design_Aspects.pdf")
pdf_loader = pdf_loader.load()
chunk = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=False)
split_doc = chunk.split_documents(pdf_loader)

client = QdrantClient(":memory:")
client.create_collection(
collection_name="zoomcamp-project1",
vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = QdrantVectorStore(
client=client,
collection_name="zoomcamp-project1",
embedding=embedding_model
)

model = ChatGroq(model_name="llama-3.3-70b-versatile")

vector_store.add_documents(split_doc)

retriever = vector_store.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.5, "k": 1})

system_prompt = (
"You are a smart assistant. "
"Provide concise and professional responses from the given context.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
("system", system_prompt),
("human", "{input}")
])

retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm = model,
    retriever = retriever, 
    return_source_documents=True
)


tools = [
    Tool(
        name = "Document Retrieval",
        func=lambda q: retrieval_qa_chain.invoke({"query": q})["result"],
        description = "Retrieve knowledge from the document"
    ),
]

agent = initialize_agent(
    tools = tools, 
    llm = model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

print(agent.invoke("what is this paper title?"))