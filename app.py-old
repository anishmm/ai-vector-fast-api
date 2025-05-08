import os
import tempfile
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from dotenv import load_dotenv, find_dotenv

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

# Configuration class
class Settings():
    groq_api_key: str =GROQ_API_KEY
    embedding_model: str = "all-MiniLM-L6-v2"
    store_path: str = "./tmp_pdf/pdf_store.faiss"
    db_type: str = "support_collection"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Load settings
settings = Settings()
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Document Analyzer API")

# Global state (for simplicity; consider dependency injection for production)
embeddings = None
llm = None
databases: Dict[str, FAISS] = {}

# Initialize models
def initialize_models():
    global embeddings, llm, databases
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        print("Embeddings initialized successfully.")

        # Initialize LLM
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not found.")
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name="mistral-saba-24b",
            temperature=0
        )
        print("LLM initialized successfully.")

        # Load existing FAISS index if available
        if os.path.exists(settings.store_path):
            print(f"Loading vector store from {settings.store_path}")
            databases[settings.db_type] = FAISS.load_local(
                settings.store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        return True
    except Exception as e:
        print(f"Initialization error: {e}")
        embeddings = None
        llm = None
        return False

# Process uploaded PDFs
async def process_document(file: UploadFile) -> List[Document]:
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

# Query the database
def query_database(db: FAISS, question: str) -> tuple[str, list, dict]:
    try:
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        relevant_docs = retriever.get_relevant_documents(question)

        if relevant_docs:
            retrieval_qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer concisely based on the provided context. If the context lacks sufficient information, say so."),
                ("human", "Context: {context}\nQuestion: {input}\nAnswer:"),
            ])
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            response = retrieval_chain.invoke({"input": question})

            context_text = "\n".join([doc.page_content for doc in relevant_docs])
            prompt_text = retrieval_qa_prompt.format(context=context_text, input=question)
            answer_text = response['answer']

            # Rough token estimation
            prompt_tokens = len(prompt_text) // 4
            completion_tokens = len(answer_text) // 4
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }

            return response['answer'], relevant_docs, token_usage
        
        return "No relevant information found in the documents.", [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query error: {str(e)}"
        )

# Pydantic model for query request
class QueryRequest(BaseModel):
    question: str

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if embeddings and llm else "unhealthy",
        "embeddings_initialized": embeddings is not None,
        "llm_initialized": llm is not None,
        "database_initialized": settings.db_type in databases
    }

# Upload PDF endpoint
@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global databases
    if not embeddings:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embeddings not initialized."
        )

    try:
        all_texts = []
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only PDF files are supported."
                )
            texts = await process_document(file)
            all_texts.extend(texts)
        
        if all_texts:
            vector_store = FAISS.from_documents(all_texts, embeddings)
            vector_store.save_local(settings.store_path)
            databases[settings.db_type] = vector_store
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": "Documents processed and added to the database."}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid documents processed."
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing documents: {str(e)}"
        )

# Query endpoint
@app.post("/query")
async def query(request: QueryRequest):
    try:
        if not llm or settings.db_type not in databases:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM or database not initialized."
            )

        db = databases.get(settings.db_type)
        answer, relevant_docs, token_usage = query_database(db, request.question)
        
        return {
            "answer": answer,
            "relevant_documents": [doc.page_content for doc in relevant_docs],
            "token_usage": token_usage
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "answer": 'Error',
            "token_usage": ''
        }

# Startup event to initialize models
@app.on_event("startup")
async def startup_event():
    if not initialize_models():
        print("Failed to initialize models. Check configuration and dependencies.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)