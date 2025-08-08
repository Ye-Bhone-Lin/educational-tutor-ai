# Document AI Assistant

A web-based AI assistant that can process PDF and PPTX documents and answer questions about their content using LangChain, Groq, and Streamlit.

## Features

- 📄 **Document Processing**: Upload and process PDF and PPTX files
- 🤖 **AI-Powered Q&A**: Ask questions about your documents using advanced AI
- 🔍 **Web Search Integration**: Combines document knowledge with web search capabilities
- 💬 **Interactive Chat Interface**: Modern chat interface for easy interaction

## Installation

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd educational-tutor-ai
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the project root with your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Running the Streamlit App

1. **Start the Streamlit app**:
```bash
streamlit run streamlit_app.py
```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Upload a document**:
   - Select the document type (PDF or PPTX)
   - Upload your file using the file uploader
   - Click "Process Document" to analyze your file

4. **Ask questions**:
   - Once processing is complete, you can ask questions about your document
   - The AI will provide answers based on the document content and web search

### Running the Command Line Version

If you prefer the command-line interface:

```bash
python main.py
```

## Supported File Types

- **PDF**: Uses PyPDFLoader for text extraction
- **PPTX**: Uses UnstructuredPowerPointLoader for presentation content

## Technical Stack

- **Frontend**: Streamlit
- **AI Framework**: LangChain
- **LLM**: Groq (Llama 3.3 70B)
- **Vector Database**: Qdrant
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Document Processing**: Unstructured, PyPDF, python-pptx

## Project Structure

```
educational-tutor-ai/
├── streamlit_app.py      # Main Streamlit application
├── main.py              # Command-line version
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .env                # Environment variables (create this)
├── pdf_ext.py          # Original PDF processing script
├── pptx_ext.py         # Original PPTX processing script
└── tools/
    └── tools.py        # Additional tools
```

## Environment Variables

Create a `.env` file with the following variables:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Add it to your `.env` file

## Features in Detail

### Document Processing
- Automatic text extraction from PDF and PPTX files
- Chunk-based processing for optimal AI understanding
- Vector embedding for semantic search

### AI Agent
- Zero-shot reasoning capabilities
- Document retrieval with similarity scoring
- Web search integration via DuckDuckGo
- Context-aware responses

### User Interface
- Modern, responsive design
- Real-time processing feedback
- Chat-like interaction
- File upload with drag-and-drop support

### Demonstration

<img width="1916" height="956" alt="Screenshot 2025-08-08 at 5 39 37 PM" src="https://github.com/user-attachments/assets/1d754b77-fec6-4a7d-af03-959c887df1b9" />

<img width="1917" height="959" alt="Screenshot 2025-08-08 at 5 41 42 PM" src="https://github.com/user-attachments/assets/387d01b9-4d2c-4eb6-a0b5-9eb10a135ba8" />

<img width="1916" height="956" alt="Screenshot 2025-08-08 at 5 57 00 PM" src="https://github.com/user-attachments/assets/b645b9cc-1b3b-437d-8ce7-fb422f8598b5" />

### Common Issues

1. **API Key Error**: Ensure your `.env` file contains the correct Groq API key
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **File Upload Issues**: Check file format and size
4. **Processing Errors**: Ensure the document contains extractable text

### Performance Tips

- For large documents, processing may take a few minutes
- The system uses in-memory vector storage for faster access
- Web search is available for additional context

## License

This project is for educational purposes.
