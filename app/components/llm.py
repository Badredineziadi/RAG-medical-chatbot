from langchain_huggingface import HuggingFaceEndpoint

from app.config.config import HF_TOKEN,HUGGINGFACE_REPO_ID

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger= get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("loading LLM model from huggingface")
        llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID, 
    huggingfacehub_api_token=HF_TOKEN,
    task="conversational",
    temperature=0.3,            # must be here
    max_new_tokens=256,          # must be here
    return_full_text=False  # optional if required
)
        logger.info("LLM model loaded successfully")
        return llm
    except Exception as e:
        error_message= CustomException("Error loading the llm",e)
        logger.error(str(error_message))
