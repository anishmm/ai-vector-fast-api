import os
import logging
from typing import Optional,List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager

load_dotenv()

logger = logging.getLogger('uvicorn.error')
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

@asynccontextmanager
async def initialize_models(app: FastAPI):
    global embeddings, llm, databases
    try:
        logger.info('START')
        yield
        logger.info('END')       
    except Exception as e:
        print(f"Initialization error: {e}")
        embeddings = None
        llm = None

app = FastAPI(lifespan=initialize_models)


@app.get("/")
async def root():
    logger.info('GET /')
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info('health /')
    return {
        "status": "healthy" if embeddings and llm else "unhealthy",
        "embeddings_initialized": embeddings is not None,
        "llm_initialized": llm is not None,
        "database_initialized": settings.db_type in databases
    }