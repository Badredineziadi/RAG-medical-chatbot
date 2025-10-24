from langchain_community.vectorstores import FAISS
import os
from app.components.embeddings import get_embeddings_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH
logger= get_logger(__name__)

def load_vector_store():
    try:
        if os.path.exists(DB_FAISS_PATH):
            embedding_model = get_embeddings_model()
            logger.info("Loading FAISS vector store ...")
            return FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model,
                allow_dangerous_deserialization=True)
        else:
            logger.warning("No vector store found ..")
    except Exception as e:
        error_message= CustomException("Failed to load vector store", e)
        logger.error(str(error_message))

def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to save to vector store")
        logger.info("Generating the new vector store ...")
        embedding_model= get_embeddings_model()
        db= FAISS.from_documents(text_chunks, embedding_model)
        logger.info("Saving the vector store to disk ...")
        db.save_local(DB_FAISS_PATH)
        logger.info("Vector store saved successfully.")
        return db
    except Exception as e:
        error_message= CustomException("Failed to save vector store", e)
        logger.error(str(error_message))

        