import os
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH,CHUNK_OVERLAP,CHUNK_SIZE

logger=get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"The specified data path does not exist: {DATA_PATH}")
        logger.info(f"Loading files from {DATA_PATH}")
        loader=DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents=loader.load()
        if not documents:
            raise CustomException(f"No PDF files found in the specified directory: {DATA_PATH}")
        else:
            logger.info(f"Loaded {len(documents)} documents from {DATA_PATH}")
        return documents
    except Exception as e:
        error_msj= CustomException("Failed to load PDF files", e)
        logger.error(str(error_msj))
        return []
    
def create_document_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents provided for chunking.")
        logger.info(f"Splitting {len(documents)} documents into ")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"generated {len(text_chunks)} text chunks from documents")
        return text_chunks
    except Exception as e:
        error_msj= CustomException("Failed to create document chunks", e)
        logger.error(str(error_msj))
        return []