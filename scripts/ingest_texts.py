import sys
import shutil
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.rag_engine import RAGEngine
from app.utils.logger import get_logger
from app.utils.text_processor import TextProcessor

logger = get_logger(__name__)

def clean_existing_data():
    """Clean up existing chunked data and embeddings."""
    try:
        # Clean up chunks directory
        chunks_dir = Path("data/chunks")
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
            logger.info("Cleaned up chunks directory")
        
        # Clean up embeddings directory
        embeddings_dir = Path("data/embeddings")
        if embeddings_dir.exists():
            shutil.rmtree(embeddings_dir)
            logger.info("Cleaned up embeddings directory")
        
        # Create fresh directories
        chunks_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created fresh directories for chunks and embeddings")
        
    except Exception as e:
        logger.error(f"Error cleaning up existing data: {str(e)}")
        sys.exit(1)

def create_chunks():
    """Create chunks from text files."""
    try:
        text_processor = TextProcessor()
        texts_dir = Path("data/texts")
        chunks_dir = Path("data/chunks")
        
        # Get all text files
        text_files = list(texts_dir.glob("*.txt"))
        logger.info(f"Found {len(text_files)} text files to process")
        
        total_chunks = 0
        # Process each text file
        for text_path in text_files:
            try:
                logger.info(f"Processing {text_path.name}")
                # Read the text file
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Create chunks
                chunks = text_processor.create_chunks(text)
                total_chunks += len(chunks)
                
                # Save chunks to file
                chunk_file = chunks_dir / f"{text_path.stem}_chunks.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    for i, chunk in enumerate(chunks, 1):
                        f.write(f"--- Chunk {i} ---\n{chunk}\n\n")
                
                logger.info(f"Created {len(chunks)} chunks for {text_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {text_path.name}: {str(e)}")
                continue
        
        logger.info(f"Total chunks created: {total_chunks}")
        return total_chunks
        
    except Exception as e:
        logger.error(f"Error during chunk creation: {str(e)}")
        sys.exit(1)

def ingest_chunks():
    """Ingest chunks into the vector store."""
    try:
        # Initialize RAG engine
        rag_engine = RAGEngine()
        
        # Update the index with the new chunks
        rag_engine.update_index()
        
        logger.info("Successfully ingested chunks into vector store")
        
    except Exception as e:
        logger.error(f"Error during chunk ingestion: {str(e)}")
        sys.exit(1)

def main():
    """Main function to orchestrate the entire process."""
    try:
        # Step 1: Clean up existing data
        logger.info("Step 1: Cleaning up existing data")
        clean_existing_data()
        
        # Step 2: Create chunks from text files
        logger.info("Step 2: Creating chunks from text files")
        total_chunks = create_chunks()
        
        # Step 3: Ingest chunks into vector store
        logger.info("Step 3: Ingesting chunks into vector store")
        ingest_chunks()
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 