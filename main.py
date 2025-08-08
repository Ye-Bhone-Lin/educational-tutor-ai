from typing import Literal, Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain_openai import OpenAI

class RetrieveApproach:
    def __init__(self, model, pptx: Optional[str] = None, pdf_file: Optional[str] = None, youtube_vd: Optional[str] = None):
        self.pptx = pptx
        self.pdf_file = pdf_file
        self.youtube_vd = youtube_vd
        self.model = model
        self.documents = None
        self.chain = None

    def pdf_load_documents(self):
        loader = PyPDFLoader(self.pdf_file)
        self.documents = loader.load()
    
    def pptx_load_documents(self):
        loader = UnstructuredPowerPointLoader(self.pptx, mode='paged')
        self.documents = loader.load()

    def youtube_load_documents(self):
        loader = YoutubeLoader.from_youtube_url(self.youtube_vd, add_video_info=False)
        self.documents = loader.load()

    def load_documents(self):
        if self.pdf_file:
            self.pdf_load_documents()
        elif self.pptx:
            self.pptx_load_documents()
        elif self.youtube_vd:
            self.youtube_load_documents()
        else:
            raise ValueError("No document source provided.")


    def vector_database(self):
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs = splitter.split_documents(self.documents)

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
        model = OpenAI(
            base_url="http://0.0.0.0:1233/v1",  # example endpoint
            api_key="lm-studio",
            model="llama-3.2-3b-instruct"
        )

        vector_store.add_documents(split_docs)

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": 1}
        )

        system_prompt = (
            "You are a smart assistant. "
            "Provide concise and professional responses from the given context.\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        stuff_chain = create_stuff_documents_chain(model, prompt)
        chain = create_retrieval_chain(retriever, stuff_chain)
        return chain 
    
    def query(question: str) -> str:
        if not self.chain:
            raise ValueError("Chain not initialized. Call vector_database() first.")
        result = self.chain.invoke({"input": question})
        return result["answer"]
