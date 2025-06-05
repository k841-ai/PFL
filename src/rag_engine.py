import os
import json
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
from pathlib import Path
import logging
from datetime import datetime
import sys

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_engine.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", ollama_model: str = "llama3.2"):
        """Initialize RAG engine with Ollama and vector store."""
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'}
        )
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        try:
            persist_directory = "data/chroma_db"
            if os.path.exists(persist_directory):
                logger.info("Loading existing vector store...")
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("Vector store loaded successfully")
            else:
                logger.info("Creating new vector store...")
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("New vector store created")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _load_text_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if not content:
                    logger.warning(f"Empty file: {file_path}")
                    return ""
                return content
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return ""

    def _create_chunks(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        try:
            if not text.strip():
                return []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            return [Document(page_content=chunk) for chunk in chunks]
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            return []

    def process_text_file(self, file_path: str) -> bool:
        try:
            logger.info(f"Processing file: {file_path}")
            content = self._load_text_file(file_path)
            if not content:
                return False
            chunks = self._create_chunks(content)
            if not chunks:
                logger.warning(f"No valid chunks created from {file_path}")
                return False
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
            logger.info(f"Successfully processed {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return False

    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        try:
            results = {
                "processed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": []
            }
            directory = Path(directory_path)
            if not directory.exists():
                raise ValueError(f"Directory not found: {directory_path}")
            text_files = list(directory.glob("**/*.txt"))
            total_files = len(text_files)
            logger.info(f"Found {total_files} text files to process")
            for file_path in text_files:
                try:
                    if self.process_text_file(str(file_path)):
                        results["processed"] += 1
                    else:
                        results["skipped"] += 1
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"{file_path}: {str(e)}")
            logger.info(f"Directory processing complete. Results: {json.dumps(results, indent=2)}")
            return results
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise

    def _get_relevant_context(self, query: str, k: int = 3) -> List[Document]:
        """Get relevant context with improved retrieval."""
        try:
            # First try semantic search with more documents
            docs = self.vector_store.similarity_search(
                query,
                k=k*2  # Retrieve more documents for better context
            )
            
            if not docs:
                logger.warning("No relevant documents found")
                return []
            
            # Log the retrieved documents for debugging
            logger.info(f"Retrieved {len(docs)} documents")
            for i, doc in enumerate(docs):
                logger.info(f"Document {i+1} content: {doc.page_content[:200]}...")
                
            return docs
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def _call_ollama(self, prompt: str) -> str:
        """Send prompt to Ollama and return the response."""
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try again."

    def _generate_response(self, query: str, context: List[Document]) -> str:
        """Generate response with improved prompt engineering."""
        try:
            if not context:
                return "I apologize, but I don't have enough relevant information to answer your question accurately. Could you please rephrase your question or provide more context?"

            # Construct context from documents
            context_text = "\n\n".join([doc.page_content for doc in context])
            
            # Enhanced prompt template with better instructions
            prompt = f"""You are a helpful AI assistant. Use the following context to answer the question. 
            If the context contains specific numbers or data, include them in your response.
            If the context doesn't contain enough information to answer the question accurately, say so.
            If the question is unclear or ambiguous, ask for clarification.
            
            Context:
            {context_text}
            
            Question: {query}
            
            Answer:"""
            
            return self._call_ollama(prompt).strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try again."

    def query(self, query: str, k: int = 3) -> Dict[str, Any]:
        try:
            start_time = datetime.now()
            context = self._get_relevant_context(query, k)
            response = self._generate_response(query, context)
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                "response": response,
                "context_used": [doc.page_content for doc in context],
                "processing_time": processing_time,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error while processing your query. Please try again.",
                "context_used": [],
                "processing_time": 0,
                "status": "error",
                "error": str(e)
            } 