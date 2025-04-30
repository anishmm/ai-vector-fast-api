from typing import Optional
from fastapi import FastAPI
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

load_dotenv()
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





# Global state (for simplicity; consider dependency injection for production)
embeddings = None
llm = None
databases: Dict[str, FAISS] = {}


def lifespan(app: FastAPI):
    print("Application is starting up...")
    global embeddings, llm, databases
    try:
        # Initialize embeddings
        # embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        print("Embeddings initialized successfully.")

        # # Initialize LLM
        # if not settings.groq_api_key:
        #     raise ValueError("GROQ_API_KEY not found.")
        # llm = ChatGroq(
        #     api_key=settings.groq_api_key,
        #     model_name="mistral-saba-24b",
        #     temperature=0
        # )
        # print("LLM initialized successfully.")

        # # Load existing FAISS index if available
        # if os.path.exists(settings.store_path):
        #     print(f"Loading vector store from {settings.store_path}")
        #     databases[settings.db_type] = FAISS.load_local(
        #         settings.store_path,
        #         embeddings,
        #         allow_dangerous_deserialization=True
        #     )
        # return True
    except Exception as e:
        print(f"Initialization error: {e}")
        embeddings = None
        llm = None
        return False

 
# Pass the lifespan handler to FastAPI
app = FastAPI()

app.add_event_handler('startup', lambda: print("API startup"))
app.add_event_handler('shutdown', lambda: print("API shutdown"))

# Startup event to initialize models
# @app.on_event("startup")
# async def startup_event():
#     if not initialize_models():
#         print("Failed to initialize models. Check configuration and dependencies.")


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}