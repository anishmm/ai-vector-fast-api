import os
import logging
from typing import Optional,List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from contextlib import asynccontextmanager
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

logger = logging.getLogger('uvicorn.error')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

# Configuration class
class Settings():
    groq_api_key: str =GROQ_API_KEY
    embedding_model: str = "all-MiniLM-L6-v2"
    store_path: str = "./tmp_pdf/index.faiss"
    db_type: str = "support_collection"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Pydantic model for query request
class QueryRequest(BaseModel):
    question: str

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
        print('START initialize')

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        

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
       
        print('END initialize')   
        print(f" databases {databases[settings.db_type]}")    
        yield
    except Exception as e:
        print(f"Initialization error: {e}")
        embeddings = None
        llm = None

app = FastAPI(lifespan=initialize_models)

# Query the database
def query_database(db: FAISS, question: str) -> tuple[str, list, dict]:
    try:
        
        print(f"db: {db}")

        print(f"question: {question}")

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )


        print(f"retriever: {retriever}")
 
        
        relevant_docs = retriever.get_relevant_documents(question)
        print("333")
        if relevant_docs:
            print("44")
            retrieval_qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer concisely based on the provided context. If the context lacks sufficient information, say so."),
                ("human", "Context: {context}\nQuestion: {input}\nAnswer:"),
            ])
            print("55")
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            print("66")
            response = retrieval_chain.invoke({"input": question})
            print(response)

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
        print(f"{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query error: {str(e)}"
        )


@app.get("/")
def root():
    logger.info('GET /')
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


# Health check endpoint
@app.get("/health")
def health_check():
    logger.info('health /')
    return {
        "status": "healthy" if embeddings and llm else "unhealthy",
        "embeddings_initialized": embeddings is not None,
        "llm_initialized": llm is not None,
        "database_initialized": settings.db_type in databases
    }


# Query endpoint
@app.post("/query")
def query(request: QueryRequest):
    try:
        if not llm or settings.db_type not in databases:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM or database not initialized."
            )
        
        db = databases.get(settings.db_type)
        answer, relevant_docs, token_usage = query_database(db, request.question)
        print("3")
        return {
            "answer": answer,
            "relevant_documents": [doc.page_content for doc in relevant_docs],
            "token_usage": token_usage
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "token_usage": ''
        }
    